from __future__ import annotations

import argparse
import csv
import gc
import json
import math
import subprocess
from collections import defaultdict
from datetime import datetime, timezone
from pathlib import Path
from typing import Dict, Mapping, Optional, Sequence, Tuple

import matplotlib.pyplot as plt
import numpy as np
import torch
from PIL import Image, ImageDraw
from torch import Tensor, nn

from trkh.core.utils import set_seed
from trkh.inference.inference import load_model
from trkh.models.model import classification_logits_from_features
from trkh.tools import audit_cropr_token_selector_preflight as preflight
from trkh.tools import audit_foveal_aggregated_attention_pair as common
from trkh.tools import audit_inattentive_token_fusion_pair as fusion_pair
from trkh.tools import audit_inattentive_token_fusion_preflight as fusion_preflight
from trkh.tools.audit_ceconv_stem_residual_readiness import (
    _batched_gradcam,
    _heat_overlay,
    _normalize_maps,
)
from trkh.tools.audit_counterfactual_illumination_disagreement_readiness import (
    _classification_metrics,
)


METHOD = "cropr_token_selector_a0_pair_audit"
PAIR_METHOD = "cropr_token_selector_a0_pair"
FOCUS_CLASS = 1
RESTRICTED_NEGATIVE_CLASSES = (0, 2, 4)
FOLD = 0
EXPECTED_HOLDOUT_ROWS = 1_843
EXPECTED_FIT_ROWS = 7_372
EXPECTED_HOLDOUT_COUNTS = [380, 109, 393, 503, 458]
EXPECTED_FIT_COUNTS = [1_561, 432, 1_527, 2_017, 1_835]
EXPECTED_FIT_INDEX_SHA256 = (
    "22edca99022fe2dcd0287a08b5f6b7c0d699603b904f65c82b51fdea7f31ce5d"
)
EXPECTED_HOLDOUT_INDEX_SHA256 = (
    "a628686b491c8b8f10bbf1782e6c84a923f21c8b53617cf0b0325260e97e97ae"
)
EXPECTED_PARAMETER_COUNT = 8_301_856
LOCKED_DECLARATION_SHA256 = (
    "2e0993752d58d99ea429bfefe1e2bfe6fa949e45aea1a26cc4bdfee97d4db21c"
)
LOCKED_FOLD_SUMMARY_SHA256 = (
    "2f938b11573073ed957c2522e1a170e6043522f305a787620d1af5f6dbd66763"
)
LOCKED_PROTOCOL_SHA256 = (
    "42e912a6bdc320d98a9acbb79b6d48c1e372a5df861f279af1c6988b3ff6a847"
)
LOCKED_PREFLIGHT_SUMMARY_SHA256 = (
    "12902d6930ec89fd658c2b86e42a30f07ea44643a2fd67a8a1fddb5b7f1e873c"
)
LOCKED_PAIR_MANIFEST_SHA256 = (
    "09a89cbe4cf4f87e6ce2ac52dd6affd80b732357bc53c1bb75979f7a6180cc1e"
)
LOCKED_CURRENT_COMMAND_SHA256 = (
    "36b9aa1a21b765829acf4c8321be147bd76297de4ccdb8a40e6dee8e37940faf"
)
LOCKED_COMMAND_HISTORY_SHA256 = (
    "39bd2879ce66fddf36a953021ea1e40f8d9de6cb4334b9b825011b2b8dc98f53"
)
FAILED_AUDIT_HEAD = "347c25dac82867021863d8648fece48732a18c98"
FAILED_AUDIT_DIR = Path("runs/audit_cropr_token_selector_a0_pair_20260717")
LOCKED_FAILED_SUMMARY_SHA256 = (
    "6404590b0a25049d41238b019bba6da0ef84aea35184f68c588d922f4c04f9cb"
)
LOCKED_FAILURE_RECORD_SHA256 = (
    "5dc48089553d747656d6e014df71bcbc8ef0fd4c0adea53b5b61301f1a6c7311"
)
LOCKED_FAILED_MANIFEST_SHA256 = (
    "af4ff08eebccd75d13ec5d7764fe4e0db0966ff44fffa5f0ae68b53f68fcbdc0"
)
RUNTIME_PATHS = (
    "trkh/models/cropr_token_selector.py",
    "trkh/models/model.py",
    "trkh/core/config.py",
    "trkh/training/train.py",
    "scripts/run_trkh_5class_attention_views_v8.ps1",
    "scripts/run_trkh_cropr_token_selector_a0.ps1",
    "configs/trkh_cropr_a0_fitonly_20260716.yaml",
    "docs/TRKH_5CLASS_CROPR_TOKEN_SELECTOR_READINESS_PROTOCOL_20260716.md",
)


def parse_args(argv: Optional[Sequence[str]] = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Locked train-only holdout and XAI audit for the matched Cropr "
            "token-selector A0 pair. Official validation and test are forbidden."
        )
    )
    parser.add_argument(
        "--control-checkpoint",
        type=Path,
        default=Path(
            "runs/probe_cropr_a0_native_control_5e_20260716/checkpoints/last.pt"
        ),
    )
    parser.add_argument(
        "--candidate-checkpoint",
        type=Path,
        default=Path(
            "runs/probe_cropr_a0_learned_candidate_5e_20260716/checkpoints/last.pt"
        ),
    )
    parser.add_argument(
        "--control-run-dir",
        type=Path,
        default=Path("runs/probe_cropr_a0_native_control_5e_20260716"),
    )
    parser.add_argument(
        "--candidate-run-dir",
        type=Path,
        default=Path("runs/probe_cropr_a0_learned_candidate_5e_20260716"),
    )
    parser.add_argument(
        "--pair-manifest",
        type=Path,
        default=Path("runs/cropr_a0_pair_manifest_20260716.json"),
    )
    parser.add_argument(
        "--fold-data",
        type=Path,
        default=Path("runs/yolof_cidt_fold0_trainonly_20260716/data.yaml"),
    )
    parser.add_argument(
        "--fold-summary",
        type=Path,
        default=Path("runs/yolof_cidt_fold0_trainonly_20260716/summary.json"),
    )
    parser.add_argument(
        "--declaration",
        type=Path,
        default=Path(
            "runs/audit_cidt_readiness_full_train_20260714/"
            "predictions_all_conditions.csv"
        ),
    )
    parser.add_argument(
        "--preflight-summary",
        type=Path,
        default=Path("runs/audit_cropr_token_selector_preflight_20260716/summary.json"),
    )
    parser.add_argument(
        "--protocol",
        type=Path,
        default=Path(
            "docs/TRKH_5CLASS_CROPR_TOKEN_SELECTOR_READINESS_PROTOCOL_20260716.md"
        ),
    )
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--batch-size", type=int, default=32)
    parser.add_argument("--xai-batch-size", type=int, default=2)
    parser.add_argument("--num-workers", type=int, default=4)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--finalize-visual-review", action="store_true")
    parser.add_argument("--visual-review-result", choices=("pass", "fail"))
    parser.add_argument("--visual-review-note", type=str, default="")
    parser.add_argument("--expected-summary-sha256", type=str, default="")
    return parser.parse_args(argv)


def _prepare_output(path: Path) -> Path:
    resolved = Path(path).resolve()
    if resolved.exists() and any(resolved.iterdir()):
        raise FileExistsError(f"Cropr pair audit output is not empty: {resolved}")
    resolved.mkdir(parents=True, exist_ok=True)
    return resolved


def _git_value(*args: str) -> str:
    return subprocess.check_output(
        ["git", *args], text=True, encoding="utf-8"
    ).strip()


def _approx(value: object, expected: float, tolerance: float = 1e-12) -> bool:
    try:
        return math.isclose(
            float(value), float(expected), rel_tol=0.0, abs_tol=tolerance
        )
    except (TypeError, ValueError):
        return False


def _cli_value(values: Sequence[str], flag: str) -> Optional[str]:
    positions = [index for index, value in enumerate(values) if value == flag]
    if len(positions) != 1 or positions[0] + 1 >= len(values):
        return None
    return str(values[positions[0] + 1])


def _normalize_train_args(values: Sequence[object]) -> list[str]:
    normalized = [str(value) for value in values]
    run_index = normalized.index("--run-name")
    normalized[run_index + 1] = "<RUN>"
    if normalized.count("--disable-cropr-token-selector-routing") > 1:
        raise ValueError("Cropr routing-disable flag occurs more than once.")
    return [
        value
        for value in normalized
        if value != "--disable-cropr-token-selector-routing"
    ]


def _without_key(values: Mapping[str, object], key: str) -> Dict[str, object]:
    result = dict(values)
    result.pop(key, None)
    return result


def _verify_failed_attempt(
    *,
    current_head: str,
    require_correction_commit: bool = True,
) -> Dict[str, object]:
    output_dir = FAILED_AUDIT_DIR.resolve()
    summary_path = output_dir / "summary.json"
    failure_path = output_dir / "failure_record.json"
    manifest_path = output_dir / "artifact_manifest.json"
    checks: Dict[str, bool] = {
        "failed_summary_sha256": summary_path.is_file()
        and common._sha256(summary_path) == LOCKED_FAILED_SUMMARY_SHA256,
        "failure_record_sha256": failure_path.is_file()
        and common._sha256(failure_path) == LOCKED_FAILURE_RECORD_SHA256,
        "failed_manifest_sha256": manifest_path.is_file()
        and common._sha256(manifest_path) == LOCKED_FAILED_MANIFEST_SHA256,
    }
    if not all(checks.values()):
        return {
            "checks": checks,
            "all_checks_pass": False,
            "failed_checks": sorted(
                name for name, passed in checks.items() if not passed
            ),
            "output_dir": str(output_dir),
            "replay_artifacts": {},
        }
    failure = common._load_json(failure_path)
    manifest = common._load_json(manifest_path)
    manifest_rows = manifest.get("files")
    failure_rows = failure.get("artifacts")
    if not isinstance(manifest_rows, list) or not isinstance(failure_rows, list):
        raise ValueError("Cropr failed-attempt manifests lack artifact rows.")
    for row in manifest_rows:
        if not isinstance(row, Mapping):
            raise ValueError("Cropr failed artifact manifest row is invalid.")
        relative = str(row.get("path", ""))
        path = output_dir / relative
        checks[f"manifest_{relative}_exists"] = path.is_file()
        checks[f"manifest_{relative}_bytes"] = path.is_file() and int(
            path.stat().st_size
        ) == int(row.get("bytes", -1))
        checks[f"manifest_{relative}_sha256"] = path.is_file() and common._sha256(
            path
        ) == str(row.get("sha256", ""))
    replay_artifacts = {
        str(row.get("name", "")): dict(row)
        for row in failure_rows
        if isinstance(row, Mapping)
    }
    replay_names = {
        "predictions_all_conditions.csv",
        "selector_rows_all_conditions.csv",
        "perturbation_rows.csv",
        "event_manifest.csv",
    }
    checks.update(
        {
            "failure_method": failure.get("method")
            == "cropr_token_selector_a0_pair_failed_attempt",
            "failure_status": failure.get("status")
            == "preserved_before_xai_standard_forward_correction",
            "failure_head": failure.get("audit_git_head") == FAILED_AUDIT_HEAD,
            "failure_summary_sha256": failure.get("summary_sha256")
            == LOCKED_FAILED_SUMMARY_SHA256,
            "failure_scope_xai_only": failure.get("defect", {}).get("scope")
            == "xai_only",
            "replay_artifact_inventory": replay_names.issubset(replay_artifacts),
            "failed_validation_unused": failure.get("official_validation_used")
            is False,
            "failed_test_unused": failure.get("test_data_used") is False,
            "failed_raw_data_unmodified": failure.get("raw_data_modified") is False,
            "failed_current_best_unchanged": failure.get(
                "current_best_command_updated"
            )
            is False,
        }
    )
    changed_paths = {
        value.strip().replace("\\", "/")
        for value in _git_value(
            "diff", "--name-only", FAILED_AUDIT_HEAD, current_head
        ).splitlines()
        if value.strip()
    }
    if require_correction_commit:
        allowed = {
            "scripts/run_trkh_cropr_token_selector_a0_audit.ps1",
            "tests/test_audit_cropr_token_selector_pair.py",
            "trkh/tools/audit_cropr_token_selector_pair.py",
        }
        checks.update(
            {
                "correction_commit_is_new": current_head != FAILED_AUDIT_HEAD,
                "correction_delta_nonempty": bool(changed_paths),
                "correction_delta_scoped": changed_paths.issubset(allowed),
            }
        )
    return {
        "checks": checks,
        "all_checks_pass": all(checks.values()),
        "failed_checks": sorted(
            name for name, passed in checks.items() if not passed
        ),
        "output_dir": str(output_dir),
        "summary_sha256": common._sha256(summary_path),
        "failure_record_sha256": common._sha256(failure_path),
        "artifact_manifest_sha256": common._sha256(manifest_path),
        "failed_audit_head": FAILED_AUDIT_HEAD,
        "correction_audit_head": current_head,
        "changed_paths": sorted(changed_paths),
        "replay_artifacts": replay_artifacts,
    }


def _compare_failed_artifacts(
    failed_attempt: Mapping[str, object], *, output_dir: Path
) -> Dict[str, object]:
    expected = failed_attempt.get("replay_artifacts")
    if not isinstance(expected, Mapping):
        raise ValueError("Cropr failed attempt lacks replay artifact metadata.")
    names = (
        "predictions_all_conditions.csv",
        "selector_rows_all_conditions.csv",
        "perturbation_rows.csv",
        "event_manifest.csv",
    )
    checks: Dict[str, bool] = {}
    observed: Dict[str, object] = {}
    for name in names:
        row = expected.get(name)
        if not isinstance(row, Mapping):
            raise ValueError(f"Cropr failed attempt lacks replay target {name}.")
        path = Path(output_dir) / name
        exists = path.is_file()
        size = int(path.stat().st_size) if exists else -1
        digest = common._sha256(path) if exists else ""
        checks[f"replay_{name}_exists"] = exists
        checks[f"replay_{name}_bytes_exact"] = size == int(row.get("bytes", -1))
        checks[f"replay_{name}_sha256_exact"] = digest == str(
            row.get("sha256", "")
        )
        observed[name] = {"bytes": size, "sha256": digest}
    return {
        "checks": checks,
        "all_checks_pass": all(checks.values()),
        "failed_checks": sorted(
            name for name, passed in checks.items() if not passed
        ),
        "artifacts": observed,
    }


def _run_provenance(
    *,
    control_run: Path,
    candidate_run: Path,
    control_checkpoint: Path,
    candidate_checkpoint: Path,
    pair_manifest_path: Path,
    preflight_path: Path,
    fold_summary_path: Path,
    protocol_path: Path,
    require_clean_worktree: bool = True,
) -> Dict[str, object]:
    pair_manifest = common._load_json(pair_manifest_path)
    preflight_summary = common._load_json(preflight_path)
    fold_summary = common._load_json(fold_summary_path)
    pair_head = str(pair_manifest.get("git_head", ""))
    current_head = _git_value("rev-parse", "HEAD")
    current_upstream = _git_value("rev-parse", "@{upstream}")
    runtime_diff = subprocess.run(
        ["git", "diff", "--quiet", pair_head, current_head, "--", *RUNTIME_PATHS],
        check=False,
    )
    tracked_status = _git_value("status", "--short", "--untracked-files=no")
    preflight_gate = preflight_summary.get("gate", {})
    checks: Dict[str, bool] = {
        "preflight_summary_hash": common._sha256(preflight_path)
        == LOCKED_PREFLIGHT_SUMMARY_SHA256,
        "preflight_method": preflight_summary.get("method")
        == "cropr_token_selector_a0_preflight",
        "preflight_automated_pass": bool(preflight_gate.get("automated_pass")),
        "preflight_pair_permission": bool(
            preflight_gate.get("formal_pair_permission")
        ),
        "preflight_validation_forbidden": preflight_summary.get("validation_used")
        is False,
        "preflight_test_forbidden": preflight_summary.get("test_used") is False,
        "pair_manifest_hash": common._sha256(pair_manifest_path)
        == LOCKED_PAIR_MANIFEST_SHA256,
        "pair_manifest_method": pair_manifest.get("method") == PAIR_METHOD,
        "pair_manifest_causal_difference": pair_manifest.get(
            "sole_causal_difference"
        )
        == "cropr_token_selector_routing_false_vs_true",
        "pair_manifest_holdout_not_loaded": pair_manifest.get(
            "holdout_loaded_during_training"
        )
        is False,
        "pair_manifest_validation_forbidden": pair_manifest.get(
            "official_validation_used"
        )
        is False,
        "pair_manifest_test_forbidden": pair_manifest.get("test_used") is False,
        "pair_manifest_commands_unchanged": pair_manifest.get(
            "current_best_command_updated"
        )
        is False,
        "pair_manifest_preflight_path": Path(
            str(pair_manifest.get("preflight_summary", ""))
        ).resolve()
        == preflight_path.resolve(),
        "pair_manifest_control_path": Path(
            str(pair_manifest.get("control_run", ""))
        ).resolve()
        == control_run.resolve(),
        "pair_manifest_candidate_path": Path(
            str(pair_manifest.get("candidate_run", ""))
        ).resolve()
        == candidate_run.resolve(),
        "fold_summary_hash": common._sha256(fold_summary_path)
        == LOCKED_FOLD_SUMMARY_SHA256,
        "protocol_hash": common._sha256(protocol_path) == LOCKED_PROTOCOL_SHA256,
        "fold_fit_rows": int(fold_summary.get("fit_rows", -1)) == EXPECTED_FIT_ROWS,
        "fold_holdout_rows": int(fold_summary.get("holdout_rows", -1))
        == EXPECTED_HOLDOUT_ROWS,
        "fold_fit_counts": fold_summary.get("fit_class_counts")
        == EXPECTED_FIT_COUNTS,
        "fold_holdout_counts": fold_summary.get("holdout_class_counts")
        == EXPECTED_HOLDOUT_COUNTS,
        "fold_zero_source_overlap": int(fold_summary.get("source_overlap", -1))
        == 0,
        "fold_raw_unmodified": fold_summary.get("raw_data_modified") is False,
        "fold_test_is_compatibility_mirror": fold_summary.get(
            "test_mirrors_holdout"
        )
        is True,
        "tracked_worktree_clean": tracked_status == ""
        or not require_clean_worktree,
        "head_pushed": current_head == current_upstream,
        "runtime_files_unchanged_since_pair": runtime_diff.returncode == 0,
        "current_command_unchanged": common._sha256(
            Path("docs/TRKH_CURRENT_BEST_FULL_TRAIN_COMMANDS_20260706.txt")
        )
        == LOCKED_CURRENT_COMMAND_SHA256,
        "command_history_unchanged": common._sha256(
            Path("docs/TRKH_CURRENT_BEST_COMMAND_UPDATE_HISTORY.txt")
        )
        == LOCKED_COMMAND_HISTORY_SHA256,
    }
    exact_launcher = {
        "scheduler_total_epochs": 5,
        "batch_size": 32,
        "grad_accum_steps": 2,
        "seed": 42,
        "disable_balanced_epoch_sampling": True,
        "learning_rate": 2.5e-4,
        "min_learning_rate": 1e-6,
        "warmup_epochs": 1,
        "weight_decay": 0.05,
        "classification_loss": "ldam_focal",
        "metric_learning_loss_weight": 0.04,
        "attention_view_loss_weight": 0.0,
        "attention_crop_probability": 0.0,
        "attention_drop_probability": 0.0,
        "teacher_focus_binary_loss_weight": 0.0,
        "bbox_spatial_fusion": True,
        "data_cartography": True,
        "skip_final_test": True,
        "no_pretrained": True,
        "resume_checkpoint": "",
        "token_prune_foreground_weight": 0.35,
        "cropr_token_selector": True,
    }
    run_payloads: Dict[str, object] = {}
    occurrences: Dict[str, Mapping[str, object]] = {}
    normalized_args: Dict[str, list[str]] = {}
    resolved_payloads: Dict[str, Mapping[str, object]] = {}
    checkpoint_paths = {
        "control": control_checkpoint,
        "candidate": candidate_checkpoint,
    }
    for role, run_dir in (("control", control_run), ("candidate", candidate_run)):
        launcher = common._load_json(run_dir / "launcher_args.json")
        resolved = common._load_json(run_dir / "resolved_config.json")
        summary = common._load_json(run_dir / "summary.json")
        history_rows = list(
            csv.DictReader(
                (run_dir / "history.csv").open(
                    "r", encoding="utf-8-sig", newline=""
                )
            )
        )
        occurrence_path = run_dir / "data_cartography_train_occurrence_hashes.json"
        occurrence = common._load_json(occurrence_path)
        occurrences[role] = occurrence
        train_args = [str(value) for value in launcher.get("train_args", [])]
        normalized_args[role] = _normalize_train_args(train_args)
        train_config = resolved.get("train_config", {})
        model_config = resolved.get("model_config", {})
        augmentation_config = resolved.get("augmentation_config", {})
        if not all(
            isinstance(value, Mapping)
            for value in (train_config, model_config, augmentation_config)
        ):
            raise ValueError(f"{role} resolved config lacks a mapping section.")
        resolved_payloads[role] = {
            "train": train_config,
            "model": model_config,
            "augmentation": augmentation_config,
        }
        role_checks: Dict[str, bool] = {}
        for key, expected in exact_launcher.items():
            observed = launcher.get(key)
            role_checks[f"launcher_{key}"] = (
                _approx(observed, expected)
                if isinstance(expected, float)
                else observed == expected
            )
        expected_routing = role == "candidate"
        checkpoint_path = checkpoint_paths[role]
        manifest_hash_key = f"{role}_last_checkpoint_sha256"
        role_checks.update(
            {
                "launcher_fit_only_data": str(launcher.get("data_yaml", "")).replace(
                    "\\", "/"
                )
                == "configs/trkh_cropr_a0_fitonly_20260716.yaml",
                "launcher_routing_role": bool(
                    launcher.get("cropr_token_selector_routing", False)
                )
                == expected_routing,
                "train_args_epochs": _cli_value(train_args, "--epochs") == "5",
                "train_args_patience": _cli_value(train_args, "--patience") == "3",
                "train_args_cropr_once": train_args.count(
                    "--cropr-token-selector"
                )
                == 1,
                "train_args_routing_role": train_args.count(
                    "--disable-cropr-token-selector-routing"
                )
                == int(not expected_routing),
                "resolved_routing_role": bool(
                    model_config.get("cropr_token_selector_routing", False)
                )
                == expected_routing,
                "resolved_cropr_enabled": model_config.get("cropr_token_selector")
                is True,
                "resolved_token_pruning": model_config.get("token_pruning") is True,
                "resolved_prune_layers": str(
                    model_config.get("token_prune_layers", "")
                )
                == "2,5",
                "resolved_keep_rates": str(model_config.get("token_keep_rates", ""))
                == "0.85,0.65",
                "resolved_foreground_weight": _approx(
                    model_config.get("token_prune_foreground_weight"), 0.35
                ),
                "resolved_epochs": int(train_config.get("epochs", -1)) == 5,
                "resolved_patience": int(
                    train_config.get("early_stopping_patience", -1)
                )
                == 3,
                "resolved_scheduler_epochs": int(
                    train_config.get("scheduler_total_epochs", -1)
                )
                == 5,
                "resolved_pairwise_margin_weight": _approx(
                    train_config.get("pairwise_margin_loss_weight"), 0.04
                ),
                "resolved_metric_learning_weight": _approx(
                    train_config.get("metric_learning_loss_weight"), 0.04
                ),
                "resolved_model_ema": train_config.get("model_ema") is True,
                "resolved_model_ema_decay": _approx(
                    train_config.get("model_ema_decay"), 0.995
                ),
                "five_history_epochs": len(history_rows) == 5,
                "last_epoch_exact": int(summary.get("last_epoch", -1)) == 5,
                "stop_completed": summary.get("stop_reason") == "completed",
                "test_summary_absent": summary.get("test_summary") is None,
                "architecture_trace_completed": summary.get(
                    "architecture_trace", {}
                ).get("status")
                == "completed",
                "parameter_count_exact": int(summary.get("parameter_count", -1))
                == EXPECTED_PARAMETER_COUNT,
                "resolved_resume_not_loaded": not bool(
                    resolved.get("resume", {}).get("loaded", True)
                ),
                "last_checkpoint_exists": checkpoint_path.is_file(),
                "last_checkpoint_hash_matches_manifest": common._sha256(
                    checkpoint_path
                )
                == str(pair_manifest.get(manifest_hash_key, "")),
            }
        )
        checks.update(
            {f"{role}_{name}": bool(value) for name, value in role_checks.items()}
        )
        run_payloads[role] = {
            "launcher_args_sha256": common._sha256(run_dir / "launcher_args.json"),
            "resolved_config_sha256": common._sha256(run_dir / "resolved_config.json"),
            "summary_sha256": common._sha256(run_dir / "summary.json"),
            "history_sha256": common._sha256(run_dir / "history.csv"),
            "occurrence_sha256": common._sha256(occurrence_path),
            "last_checkpoint_sha256": common._sha256(checkpoint_path),
            "summary": summary,
            "checks": role_checks,
        }
    checks.update(
        {
            "normalized_train_args_equal_except_routing_role": normalized_args[
                "control"
            ]
            == normalized_args["candidate"],
            "resolved_train_configs_equal": resolved_payloads["control"]["train"]
            == resolved_payloads["candidate"]["train"],
            "resolved_augmentation_configs_equal": resolved_payloads["control"][
                "augmentation"
            ]
            == resolved_payloads["candidate"]["augmentation"],
            "resolved_model_configs_equal_except_routing": _without_key(
                resolved_payloads["control"]["model"],
                "cropr_token_selector_routing",
            )
            == _without_key(
                resolved_payloads["candidate"]["model"],
                "cropr_token_selector_routing",
            ),
            "occurrence_epoch_records_equal": occurrences["control"].get(
                "epochs", []
            )
            == occurrences["candidate"].get("epochs", []),
            "occurrence_five_epochs": len(
                occurrences["control"].get("epochs", [])
            )
            == len(occurrences["candidate"].get("epochs", []))
            == 5,
            "occurrence_rows_exact": all(
                int(row.get("occurrences", -1)) == EXPECTED_FIT_ROWS
                and int(row.get("unique_sample_indices", -1)) == EXPECTED_FIT_ROWS
                and row.get("class_counts") == EXPECTED_FIT_COUNTS
                for row in occurrences["control"].get("epochs", [])
            ),
        }
    )
    return {
        "checks": checks,
        "all_checks_pass": all(checks.values()),
        "failed_checks": sorted(
            name for name, passed in checks.items() if not bool(passed)
        ),
        "pair_git_head": pair_head,
        "audit_git_head": current_head,
        "audit_git_upstream": current_upstream,
        "runtime_paths": list(RUNTIME_PATHS),
        "tracked_worktree_clean_required": bool(require_clean_worktree),
        "tracked_worktree_clean_observed": tracked_status == "",
        "pair_manifest": pair_manifest,
        "pair_manifest_sha256": common._sha256(pair_manifest_path),
        "preflight_summary_sha256": common._sha256(preflight_path),
        "runs": run_payloads,
        "occurrence": occurrences,
    }


def _forward_trace(
    model: nn.Module,
    images: Tensor,
    metadata: Mapping[str, Tensor],
) -> Tuple[Tensor, Mapping[str, object]]:
    features = model.forward_features(
        images,
        image_valid_mask=metadata.get("image_mask"),
        bbox_token_prior=metadata.get("bbox"),
        return_attention=False,
        return_trace=True,
    )
    if torch.is_tensor(metadata.get("bbox")):
        features["bbox"] = metadata["bbox"]
    return classification_logits_from_features(model, features), features


def _serialize_numbers(values: Tensor, *, precision: int = 10) -> str:
    return fusion_preflight._serialize_numbers(values, precision=precision)


def _parse_ints(value: object) -> list[int]:
    text = str(value).strip()
    return [] if not text else [int(item) for item in text.split(";")]


def _scatter_active(values: Tensor, indices: Tensor, *, total: int = 256) -> Tensor:
    if values.ndim != 2 or tuple(values.shape) != tuple(indices.shape):
        raise ValueError("Cropr scatter requires aligned [B,N] values and indices.")
    output = values.new_zeros((int(values.size(0)), int(total)))
    return output.scatter(1, indices.long(), values)


def _selector_trace_conditions(
    *,
    control: nn.Module,
    candidate: nn.Module,
    predictions: Mapping[str, Mapping[str, Sequence[Mapping[str, object]]]],
    base_dataset,
    transform,
    holdout_rows,
    output_dir: Path,
    device: torch.device,
    batch_size: int,
    num_workers: int,
    seed: int,
    hard_negative_sample_indices: Sequence[int],
) -> Tuple[Dict[str, object], list[Dict[str, object]]]:
    rows: list[Dict[str, object]] = []
    loaders: Dict[str, object] = {}
    local_indices = list(range(len(base_dataset)))
    hard_negative = {int(value) for value in hard_negative_sample_indices}
    amp_enabled = device.type == "cuda"
    for condition_index, (condition, brightness, contrast) in enumerate(
        common.CONDITIONS
    ):
        condition_dataset = common._SelectedConditionDataset(
            base_dataset,
            local_indices,
            corruption=common._condition_corruption(condition, brightness, contrast),
            transform=transform,
        )
        loader, loader_summary = common._make_loader(
            dataset=condition_dataset,
            batch_size=batch_size,
            num_workers=num_workers,
            context=f"cropr_pair_{condition}_selector_trace",
            seed=seed + 300 + condition_index,
        )
        loaders[condition] = loader_summary
        standard_by_role = {
            role: {
                int(row["local_index"]): row
                for row in predictions[role][condition]
            }
            for role in ("control", "candidate")
        }
        processed = 0
        with torch.inference_mode():
            for images_cpu, targets_cpu, metadata_cpu in loader:
                images = images_cpu.to(device=device, non_blocking=True)
                metadata = common._metadata_to_device(metadata_cpu, device)
                sample_indices = metadata_cpu.get("sample_index")
                bboxes = metadata_cpu.get("bbox")
                if not torch.is_tensor(sample_indices) or not torch.is_tensor(bboxes):
                    raise ValueError("Cropr selector trace lacks sample_index/bbox.")
                overlaps = torch.stack(
                    [
                        fusion_preflight._bbox_patch_overlap(bbox)
                        for bbox in bboxes
                    ],
                    dim=0,
                ).to(device=device)
                for role, model in (("control", control), ("candidate", candidate)):
                    with torch.autocast(
                        device_type=device.type,
                        dtype=torch.bfloat16,
                        enabled=amp_enabled,
                    ):
                        trace_logits, features = _forward_trace(model, images, metadata)
                    trace_logits_cpu = trace_logits.detach().float().cpu()
                    pruning = features.get("trace", {}).get("pruning")
                    if not isinstance(pruning, list) or len(pruning) != 2:
                        raise ValueError("Cropr trace does not contain two prune stages.")
                    previous = torch.arange(
                        256, device=device, dtype=torch.long
                    ).unsqueeze(0).expand(int(images.size(0)), -1)
                    for stage_index, layer in enumerate((2, 5)):
                        stage = pruning[stage_index]
                        required = (
                            "kept_indices",
                            "dropped_indices",
                            "attention",
                            "auxiliary_logits",
                            "normalized_entropy",
                            "score_variance",
                            "native_scores",
                            "cropr_scores",
                            "cropr_routing",
                        )
                        if any(
                            key not in stage or not torch.is_tensor(stage[key])
                            for key in required
                        ):
                            raise ValueError(
                                f"Cropr stage {layer} lacks a required trace tensor."
                            )
                        kept = stage["kept_indices"].long()
                        dropped = stage["dropped_indices"].long()
                        partition_valid = fusion_preflight._partition_valid(
                            previous.detach().cpu(),
                            kept.detach().cpu(),
                            dropped.detach().cpu(),
                        )
                        attention = stage["attention"].float()
                        auxiliary_logits = stage["auxiliary_logits"].float()
                        active_overlap = overlaps.gather(1, previous)
                        cropr_mass = (attention * active_overlap).sum(dim=1) / attention.sum(
                            dim=1
                        ).clamp_min(1e-12)
                        native_attention = stage["native_scores"].float()
                        if tuple(native_attention.shape) != tuple(active_overlap.shape):
                            raise ValueError(
                                f"Cropr stage {layer} native-score layout differs."
                            )
                        native_attention = native_attention / native_attention.sum(
                            dim=1, keepdim=True
                        ).clamp_min(1e-12)
                        native_mass = (
                            native_attention * active_overlap
                        ).sum(dim=1) / native_attention.sum(dim=1).clamp_min(1e-12)
                        selected_overlap = overlaps.gather(1, kept).sum(dim=1)
                        object_recall = selected_overlap / overlaps.sum(
                            dim=1
                        ).clamp_min(1e-12)
                        auxiliary_predictions = auxiliary_logits.argmax(dim=1)
                        auxiliary_margin = auxiliary_logits[:, FOCUS_CLASS] - auxiliary_logits[
                            :, list(RESTRICTED_NEGATIVE_CLASSES)
                        ].amax(dim=1)
                        for position, local_value in enumerate(sample_indices.tolist()):
                            local_index = int(local_value)
                            source = holdout_rows[local_index]
                            target = int(targets_cpu[position].item())
                            standard = standard_by_role[role][local_index]
                            standard_logits = torch.tensor(
                                [
                                    float(standard[f"logit_{class_index}"])
                                    for class_index in range(5)
                                ]
                            )
                            trace_error = float(
                                (
                                    standard_logits
                                    - trace_logits_cpu[position]
                                ).abs().amax().item()
                            )
                            row: Dict[str, object] = {
                                "role": role,
                                "condition": condition,
                                "layer": layer,
                                "local_index": local_index,
                                "sample_index": int(source.sample_index),
                                "source_stem": source.source_stem,
                                "object_index": int(
                                    base_dataset.samples[
                                        local_index
                                    ].primary_object_index
                                ),
                                "target": target,
                                "standard_prediction": int(standard["prediction"]),
                                "trace_prediction": int(
                                    trace_logits_cpu[position].argmax().item()
                                ),
                                "standard_trace_maximum_error": trace_error,
                                "standard_trace_prediction_match": int(
                                    standard["prediction"]
                                )
                                == int(trace_logits_cpu[position].argmax().item()),
                                "auxiliary_prediction": int(
                                    auxiliary_predictions[position].item()
                                ),
                                "auxiliary_class1_margin": float(
                                    auxiliary_margin[position].item()
                                ),
                                "normalized_entropy": float(
                                    stage["normalized_entropy"][position].float().item()
                                ),
                                "score_variance": float(
                                    stage["score_variance"][position].float().item()
                                ),
                                "cropr_bbox_mass": float(cropr_mass[position].item()),
                                "native_bbox_mass": float(native_mass[position].item()),
                                "foreground_mass_gain": float(
                                    (cropr_mass[position] - native_mass[position]).item()
                                ),
                                "selected_object_patch_recall": float(
                                    object_recall[position].item()
                                ),
                                "cropr_routing": bool(
                                    stage["cropr_routing"].item()
                                ),
                                "partition_valid": bool(partition_valid[position].item()),
                                "hard_negative": int(source.sample_index)
                                in hard_negative,
                                "kept_indices": _serialize_numbers(kept[position]),
                                "dropped_indices": _serialize_numbers(dropped[position]),
                            }
                            for class_index in range(5):
                                row[f"auxiliary_logit_{class_index}"] = float(
                                    auxiliary_logits[position, class_index].item()
                                )
                            rows.append(row)
                        previous = kept
                processed += int(images.size(0))
                if processed % 512 < int(images.size(0)) or processed == len(
                    condition_dataset
                ):
                    print(
                        json.dumps(
                            {
                                "method": METHOD,
                                "phase": "selector_trace",
                                "condition": condition,
                                "processed": processed,
                                "rows": len(condition_dataset),
                            }
                        ),
                        flush=True,
                    )
    expected_rows = 2 * len(common.CONDITIONS) * 2 * EXPECTED_HOLDOUT_ROWS
    if len(rows) != expected_rows:
        raise ValueError(f"Cropr selector trace rows differ: {len(rows)}.")
    path = output_dir / "selector_rows_all_conditions.csv"
    _write_rows(path, rows)
    summary = _selector_summary(rows)
    replay_rows = list(csv.DictReader(path.open("r", encoding="utf-8", newline="")))
    replay_summary = _selector_summary(replay_rows)
    replay_error = _maximum_tree_error(summary, replay_summary)
    return {
        "conditions": summary["conditions"],
        "route_stability": summary["route_stability"],
        "standard_trace": summary["standard_trace"],
        "rows": len(rows),
        "rows_csv": str(path.resolve()),
        "rows_csv_sha256": common._sha256(path),
        "replay_maximum_error": replay_error,
        "replay_exact": math.isfinite(replay_error) and replay_error <= 1e-12,
        "loader": loaders,
    }, rows


def _write_rows(path: Path, rows: Sequence[Mapping[str, object]]) -> None:
    if not rows:
        raise ValueError(f"No rows to write: {path}")
    with Path(path).open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)


def _maximum_tree_error(left: object, right: object) -> float:
    if isinstance(left, Mapping) and isinstance(right, Mapping):
        if set(left) != set(right):
            return math.inf
        return max(
            (_maximum_tree_error(left[key], right[key]) for key in left),
            default=0.0,
        )
    if isinstance(left, list) and isinstance(right, list):
        if len(left) != len(right):
            return math.inf
        return max(
            (_maximum_tree_error(a, b) for a, b in zip(left, right)),
            default=0.0,
        )
    if isinstance(left, bool) or isinstance(right, bool):
        return 0.0 if bool(left) == bool(right) else math.inf
    try:
        return abs(float(left) - float(right))
    except (TypeError, ValueError):
        return 0.0 if left == right else math.inf


def _set_jaccard(left: Sequence[int], right: Sequence[int]) -> float:
    left_set, right_set = set(left), set(right)
    return len(left_set.intersection(right_set)) / max(
        1, len(left_set.union(right_set))
    )


def _selector_summary(rows: Sequence[Mapping[str, object]]) -> Dict[str, object]:
    return _selector_summary_with_expected(rows, expected_rows=EXPECTED_HOLDOUT_ROWS)


def _selector_summary_with_expected(
    rows: Sequence[Mapping[str, object]], *, expected_rows: int
) -> Dict[str, object]:
    grouped: Dict[Tuple[str, str, int], list[Mapping[str, object]]] = defaultdict(
        list
    )
    for row in rows:
        grouped[(str(row["condition"]), str(row["role"]), int(row["layer"]))].append(
            row
        )
    conditions: Dict[str, object] = {}
    standard_errors: list[float] = []
    standard_matches = 0
    for condition, _, _ in common.CONDITIONS:
        role_payload: Dict[str, object] = {}
        for role in ("control", "candidate"):
            layers: Dict[str, object] = {}
            for layer in (2, 5):
                selected = grouped[(condition, role, layer)]
                if len(selected) != int(expected_rows):
                    raise ValueError(
                        f"Cropr selector group differs for {condition}/{role}/{layer}."
                    )
                targets = np.asarray(
                    [int(row["target"]) for row in selected], dtype=np.int64
                )
                predictions = np.asarray(
                    [int(row["auxiliary_prediction"]) for row in selected],
                    dtype=np.int64,
                )
                metrics = _classification_metrics(targets, predictions, num_classes=5)
                entropy = np.asarray(
                    [float(row["normalized_entropy"]) for row in selected],
                    dtype=np.float64,
                )
                variance = np.asarray(
                    [float(row["score_variance"]) for row in selected],
                    dtype=np.float64,
                )
                cropr_mass = np.asarray(
                    [float(row["cropr_bbox_mass"]) for row in selected],
                    dtype=np.float64,
                )
                native_mass = np.asarray(
                    [float(row["native_bbox_mass"]) for row in selected],
                    dtype=np.float64,
                )
                object_recall = np.asarray(
                    [
                        float(row["selected_object_patch_recall"])
                        for row in selected
                    ],
                    dtype=np.float64,
                )
                auroc_labels: list[int] = []
                auroc_scores: list[float] = []
                for row in selected:
                    target = int(row["target"])
                    hard_negative = str(row["hard_negative"]).casefold() in {
                        "true",
                        "1",
                    }
                    if target == FOCUS_CLASS:
                        auroc_labels.append(1)
                        auroc_scores.append(float(row["auxiliary_class1_margin"]))
                    elif hard_negative:
                        auroc_labels.append(0)
                        auroc_scores.append(float(row["auxiliary_class1_margin"]))
                trace_errors = [
                    float(row["standard_trace_maximum_error"]) for row in selected
                ]
                trace_matches = [
                    str(row["standard_trace_prediction_match"]).casefold()
                    in {"true", "1"}
                    for row in selected
                ]
                standard_errors.extend(trace_errors)
                standard_matches += sum(int(value) for value in trace_matches)
                layers[str(layer)] = {
                    "auxiliary": metrics,
                    "auxiliary_class1_hard_negative_auroc": common._binary_auroc(
                        auroc_labels, auroc_scores
                    ),
                    "auxiliary_auroc_rows": len(auroc_labels),
                    "normalized_entropy_mean": float(entropy.mean()),
                    "high_entropy_fraction_gt_0p995": float((entropy > 0.995).mean()),
                    "score_variance_mean": float(variance.mean()),
                    "collapsed_variance_fraction_lte_1e_8": float(
                        (variance <= 1e-8).mean()
                    ),
                    "cropr_bbox_mass_mean": float(cropr_mass.mean()),
                    "native_bbox_mass_mean": float(native_mass.mean()),
                    "foreground_mass_gain_mean": float(
                        (cropr_mass - native_mass).mean()
                    ),
                    "selected_object_patch_recall_mean": float(
                        object_recall.mean()
                    ),
                    "partition_valid_fraction": float(
                        np.mean(
                            [
                                str(row["partition_valid"]).casefold()
                                in {"true", "1"}
                                for row in selected
                            ]
                        )
                    ),
                    "routing_fraction": float(
                        np.mean(
                            [
                                str(row["cropr_routing"]).casefold()
                                in {"true", "1"}
                                for row in selected
                            ]
                        )
                    ),
                    "standard_trace_maximum_error": max(trace_errors),
                    "standard_trace_prediction_matches": sum(
                        int(value) for value in trace_matches
                    ),
                }
            role_payload[role] = {"layers": layers}
        conditions[condition] = role_payload
    clean_maps: Dict[Tuple[str, int, int], list[int]] = {}
    for role in ("control", "candidate"):
        for layer in (2, 5):
            for row in grouped[("clean", role, layer)]:
                clean_maps[(role, layer, int(row["local_index"]))] = _parse_ints(
                    row["kept_indices"]
                )
    route_stability: Dict[str, object] = {}
    for condition, _, _ in common.CONDITIONS[1:]:
        role_payload = {}
        for role in ("control", "candidate"):
            layer_payload = {}
            for layer in (2, 5):
                values = [
                    _set_jaccard(
                        clean_maps[(role, layer, int(row["local_index"]))],
                        _parse_ints(row["kept_indices"]),
                    )
                    for row in grouped[(condition, role, layer)]
                ]
                layer_payload[str(layer)] = {
                    "mean_jaccard": float(np.mean(values)),
                    "minimum_jaccard": float(np.min(values)),
                }
            role_payload[role] = {"layers": layer_payload}
        route_stability[condition] = role_payload
    return {
        "conditions": conditions,
        "route_stability": route_stability,
        "standard_trace": {
            "maximum_error": max(standard_errors),
            "prediction_matches": standard_matches,
            "expected_matches": len(rows),
        },
    }


def _perturbation_audit(
    *,
    control: nn.Module,
    candidate: nn.Module,
    base_dataset,
    transform,
    holdout_rows,
    hard_negative_sample_indices: Sequence[int],
    output_dir: Path,
    device: torch.device,
    batch_size: int,
    num_workers: int,
    seed: int,
) -> Dict[str, object]:
    hard_negative = {int(value) for value in hard_negative_sample_indices}
    selected = [
        local_index
        for local_index, row in enumerate(holdout_rows)
        if int(row.target) == FOCUS_CLASS or int(row.sample_index) in hard_negative
    ]
    rows: list[Dict[str, object]] = []
    loaders: Dict[str, object] = {}
    for mode_index, mode in enumerate(("clean", "object", "far_background")):
        dataset = common._RegionPerturbationDataset(
            base_dataset, selected, mode=mode, transform=transform
        )
        loader, loader_summary = common._make_loader(
            dataset=dataset,
            batch_size=batch_size,
            num_workers=num_workers,
            context=f"cropr_pair_{mode}_perturbation",
            seed=seed + 700 + mode_index,
        )
        loaders[mode] = loader_summary
        with torch.inference_mode():
            for images_cpu, targets_cpu, metadata_cpu in loader:
                images = images_cpu.to(device=device, non_blocking=True)
                metadata = common._metadata_to_device(metadata_cpu, device)
                sample_indices = metadata_cpu.get("sample_index")
                if not torch.is_tensor(sample_indices):
                    raise ValueError("Cropr perturbation lacks sample_index.")
                for role, model in (("control", control), ("candidate", candidate)):
                    with torch.autocast(
                        device_type=device.type,
                        dtype=torch.bfloat16,
                        enabled=device.type == "cuda",
                    ):
                        logits, _ = common._forward_classification_with_metadata(
                            model, images, metadata, device=device
                        )
                    logits = logits.detach().float().cpu()
                    margins = common._margin(logits)
                    predictions = logits.argmax(dim=1)
                    for position, local_value in enumerate(sample_indices.tolist()):
                        local_index = int(local_value)
                        source = holdout_rows[local_index]
                        rows.append(
                            {
                                "role": role,
                                "mode": mode,
                                "local_index": local_index,
                                "sample_index": int(source.sample_index),
                                "target": int(targets_cpu[position].item()),
                                "prediction": int(predictions[position].item()),
                                "class1_restricted_margin": float(
                                    margins[position].item()
                                ),
                                "hard_negative": int(source.sample_index)
                                in hard_negative,
                            }
                        )
    path = output_dir / "perturbation_rows.csv"
    _write_rows(path, rows)
    summary = _perturbation_summary(rows)
    replay = _perturbation_summary(
        list(csv.DictReader(path.open("r", encoding="utf-8", newline="")))
    )
    error = _maximum_tree_error(summary, replay)
    return {
        **summary,
        "rows_csv": str(path.resolve()),
        "rows_csv_sha256": common._sha256(path),
        "replay_maximum_error": error,
        "replay_exact": math.isfinite(error) and error <= 1e-12,
        "loaders": loaders,
    }


def _perturbation_summary(rows: Sequence[Mapping[str, object]]) -> Dict[str, object]:
    grouped: Dict[Tuple[str, str], list[Mapping[str, object]]] = defaultdict(list)
    for row in rows:
        grouped[(str(row["role"]), str(row["mode"]))].append(row)
    roles: Dict[str, object] = {}
    for role in ("control", "candidate"):
        values = {}
        for mode in ("clean", "object", "far_background"):
            selected = grouped[(role, mode)]
            values[mode] = {
                int(row["local_index"]): row for row in selected
            }
        clean_order = sorted(values["clean"])
        clean_margin = np.asarray(
            [
                float(values["clean"][index]["class1_restricted_margin"])
                for index in clean_order
            ],
            dtype=np.float64,
        )
        role_payload: Dict[str, object] = {}
        for mode in ("clean", "object", "far_background"):
            margins = np.asarray(
                [
                    float(values[mode][index]["class1_restricted_margin"])
                    for index in clean_order
                ],
                dtype=np.float64,
            )
            restricted_fp = sum(
                int(
                    str(values[mode][index]["hard_negative"]).casefold()
                    in {"true", "1"}
                    and int(values[mode][index]["prediction"]) == FOCUS_CLASS
                )
                for index in clean_order
            )
            role_payload[mode] = {
                "mean_absolute_margin_change_from_clean": float(
                    np.abs(margins - clean_margin).mean()
                ),
                "restricted_false_positives": restricted_fp,
            }
        object_effect = float(
            role_payload["object"]["mean_absolute_margin_change_from_clean"]
        )
        background_effect = float(
            role_payload["far_background"][
                "mean_absolute_margin_change_from_clean"
            ]
        )
        role_payload["object_minus_background"] = object_effect - background_effect
        role_payload["object_more_causal"] = object_effect > background_effect
        role_payload["restricted_fp_no_increase"] = bool(
            int(role_payload["object"]["restricted_false_positives"])
            <= int(role_payload["clean"]["restricted_false_positives"])
            and int(
                role_payload["far_background"]["restricted_false_positives"]
            )
            <= int(role_payload["clean"]["restricted_false_positives"])
        )
        roles[role] = role_payload
    return {"rows": len(rows), "cohort_rows": len(rows) // 6, "roles": roles}


def _representative_requests(
    *,
    dataset,
    holdout_rows,
    positive_sample_indices: Sequence[int],
) -> list[Dict[str, object]]:
    requests = common._representative_requests(
        dataset=dataset,
        holdout_rows=holdout_rows,
        positive_sample_indices=positive_sample_indices,
    )
    tiny_local = min(
        range(len(dataset)),
        key=lambda index: (
            common._sample_bbox(dataset, index)[2]
            * common._sample_bbox(dataset, index)[3],
            index,
        ),
    )
    source = holdout_rows[tiny_local]
    requests.append(
        {
            "condition": "clean",
            "local_index": tiny_local,
            "sample_index": int(source.sample_index),
            "source_stem": source.source_stem,
            "object_index": int(dataset.samples[tiny_local].primary_object_index),
            "target": int(source.target),
            "control_prediction": -1,
            "candidate_prediction": -1,
            "categories": ["representative_tiny_object"],
        }
    )
    return requests


def _sparse_token_gradcam(
    activations: Tensor,
    gradients: Tensor,
    *,
    prefix_count: int,
    patch_indices: Tensor,
) -> Tensor:
    patch_activations = activations[:, prefix_count:].float()
    patch_gradients = gradients[:, prefix_count:].float()
    if int(patch_activations.size(1)) != int(patch_indices.size(1)):
        raise ValueError("Sparse token Grad-CAM token/index counts differ.")
    weights = patch_gradients.mean(dim=1, keepdim=True)
    active = torch.relu((weights * patch_activations).sum(dim=-1))
    dense = _scatter_active(active, patch_indices)
    return _normalize_maps(dense.reshape(-1, 16, 16))


def _collect_xai_maps(
    *,
    role: str,
    model: nn.Module,
    base_dataset,
    transform,
    requests: Sequence[Mapping[str, object]],
    standard_rows: Mapping[Tuple[str, int], Mapping[str, object]],
    device: torch.device,
    batch_size: int,
    num_workers: int,
    seed: int,
    mean: Sequence[float],
    std: Sequence[float],
) -> Tuple[Dict[Tuple[str, int], Dict[str, object]], Dict[str, object]]:
    for parameter in model.parameters():
        parameter.requires_grad_(False)
    model.eval()
    records: Dict[Tuple[str, int], Dict[str, object]] = {}
    loaders: Dict[str, object] = {}
    by_condition: Dict[str, list[int]] = defaultdict(list)
    for request in requests:
        by_condition[str(request["condition"])].append(int(request["local_index"]))
    condition_specs = {
        name: (brightness, contrast)
        for name, brightness, contrast in common.CONDITIONS
    }
    for condition_index, (condition, selected) in enumerate(by_condition.items()):
        unique_indices = sorted(set(selected))
        brightness, contrast = condition_specs[condition]
        condition_dataset = common._SelectedConditionDataset(
            base_dataset,
            unique_indices,
            corruption=common._condition_corruption(condition, brightness, contrast),
            transform=transform,
        )
        loader, loader_summary = common._make_loader(
            dataset=condition_dataset,
            batch_size=batch_size,
            num_workers=num_workers,
            context=f"cropr_{role}_{condition}_xai",
            seed=seed + 900 + condition_index,
        )
        loaders[condition] = loader_summary
        for images_cpu, _, metadata_cpu in loader:
            images = images_cpu.to(device=device, non_blocking=True).requires_grad_(True)
            metadata = common._metadata_to_device(metadata_cpu, device)
            sample_indices = metadata_cpu.get("sample_index")
            crop_bbox_cpu = metadata_cpu.get("crop_bbox")
            if not torch.is_tensor(sample_indices) or not torch.is_tensor(crop_bbox_cpu):
                raise ValueError("Cropr XAI lacks sample_index/crop_bbox.")
            captured: Dict[str, Tensor] = {}

            def stem_hook(_module, _inputs, output):
                if not torch.is_tensor(output):
                    raise TypeError("Cropr stem hook requires tensor output.")
                output.retain_grad()
                captured["stem"] = output

            def block2_hook(_module, _inputs, output):
                tokens = output[0] if isinstance(output, tuple) else output
                if not torch.is_tensor(tokens):
                    raise TypeError("Cropr block-2 hook requires tensor output.")
                tokens.retain_grad()
                captured["block2"] = tokens

            def block5_hook(_module, _inputs, output):
                tokens = output[0] if isinstance(output, tuple) else output
                if not torch.is_tensor(tokens):
                    raise TypeError("Cropr block-5 hook requires tensor output.")
                tokens.retain_grad()
                captured["block5"] = tokens

            handles = (
                model.stem.register_forward_hook(stem_hook),
                model.blocks[1].register_forward_hook(block2_hook),
                model.blocks[4].register_forward_hook(block5_hook),
            )
            model.zero_grad(set_to_none=True)
            try:
                with torch.autocast(
                    device_type=device.type,
                    dtype=torch.bfloat16,
                    enabled=device.type == "cuda",
                ):
                    logits, _ = common._forward_classification_with_metadata(
                        model, images, metadata, device=device
                    )
                logits[:, FOCUS_CLASS].float().sum().backward()
            finally:
                for handle in handles:
                    handle.remove()
            stem = captured.get("stem")
            block2 = captured.get("block2")
            block5 = captured.get("block5")
            if any(
                value is None or value.grad is None
                for value in (stem, block2, block5)
            ):
                raise RuntimeError("Cropr XAI gradients were not retained.")
            with torch.inference_mode(), torch.autocast(
                device_type=device.type,
                dtype=torch.bfloat16,
                enabled=device.type == "cuda",
            ):
                trace_logits, trace_features = _forward_trace(
                    model, images.detach(), metadata
                )
            pruning = trace_features.get("trace", {}).get("pruning")
            if not isinstance(pruning, list) or len(pruning) != 2:
                raise ValueError("Cropr XAI lacks two pruning traces.")
            prefix_count = int(model.num_prefix_tokens)
            full_indices = torch.arange(
                256, device=device, dtype=torch.long
            ).unsqueeze(0).expand(int(images.size(0)), -1)
            stage1_indices = pruning[0]["kept_indices"].long().clone()
            with torch.no_grad():
                stem_heat = _batched_gradcam(
                    stem,
                    stem.grad,
                    size=(int(images.size(-2)), int(images.size(-1))),
                ).cpu()
                block2_heat = _sparse_token_gradcam(
                    block2,
                    block2.grad,
                    prefix_count=prefix_count,
                    patch_indices=full_indices,
                ).cpu()
                block5_heat = _sparse_token_gradcam(
                    block5,
                    block5.grad,
                    prefix_count=prefix_count,
                    patch_indices=stage1_indices,
                ).cpu()
                cropr2_raw = _scatter_active(
                    pruning[0]["attention"].float(), full_indices
                ).reshape(-1, 16, 16)
                cropr5_raw = _scatter_active(
                    pruning[1]["attention"].float(), stage1_indices
                ).reshape(-1, 16, 16)
                cropr2_heat = _normalize_maps(cropr2_raw).cpu()
                cropr5_heat = _normalize_maps(cropr5_raw).cpu()
            foreground = {
                "stem": common._bbox_foreground_mass(stem_heat, crop_bbox_cpu),
                "block2": common._bbox_foreground_mass(block2_heat, crop_bbox_cpu),
                "block5": common._bbox_foreground_mass(block5_heat, crop_bbox_cpu),
                "cropr2": common._bbox_foreground_mass(
                    cropr2_raw.detach().cpu(), crop_bbox_cpu
                ),
                "cropr5": common._bbox_foreground_mass(
                    cropr5_raw.detach().cpu(), crop_bbox_cpu
                ),
            }
            logits_cpu = logits.detach().float().cpu()
            trace_logits_cpu = trace_logits.detach().float().cpu()
            for position, local_value in enumerate(sample_indices.tolist()):
                local_index = int(local_value)
                standard = standard_rows[(condition, local_index)]
                standard_logits = torch.tensor(
                    [float(standard[f"logit_{index}"]) for index in range(5)]
                )
                records[(condition, local_index)] = {
                    "rgb": common._rgb_from_tensor(
                        images_cpu[position], mean=mean, std=std
                    ),
                    "bbox": crop_bbox_cpu[position].float().numpy(),
                    "stem_gradcam": stem_heat[position].numpy(),
                    "block2_gradcam": block2_heat[position].numpy(),
                    "block5_gradcam": block5_heat[position].numpy(),
                    "cropr_layer2": cropr2_heat[position].numpy(),
                    "cropr_layer5": cropr5_heat[position].numpy(),
                    "stem_foreground_mass": float(foreground["stem"][position]),
                    "block2_foreground_mass": float(
                        foreground["block2"][position]
                    ),
                    "block5_foreground_mass": float(
                        foreground["block5"][position]
                    ),
                    "cropr_layer2_foreground_mass": float(
                        foreground["cropr2"][position]
                    ),
                    "cropr_layer5_foreground_mass": float(
                        foreground["cropr5"][position]
                    ),
                    "prediction": int(logits_cpu[position].argmax().item()),
                    "standard_prediction": int(standard["prediction"]),
                    "standard_prediction_match": int(
                        logits_cpu[position].argmax().item()
                    )
                    == int(standard["prediction"]),
                    "standard_logit_maximum_error": float(
                        (standard_logits - logits_cpu[position]).abs().amax().item()
                    ),
                    "trace_standard_logit_maximum_error": float(
                        (standard_logits - trace_logits_cpu[position])
                        .abs()
                        .amax()
                        .item()
                    ),
                    "trace_standard_prediction_match": int(
                        trace_logits_cpu[position].argmax().item()
                    )
                    == int(standard["prediction"]),
                }
            del trace_features, trace_logits, logits, images
            gc.collect()
            if device.type == "cuda":
                torch.cuda.empty_cache()
    expected = {
        (str(request["condition"]), int(request["local_index"]))
        for request in requests
    }
    if set(records) != expected:
        raise ValueError(f"Cropr {role} XAI records differ from requests.")
    map_names = (
        "stem_gradcam",
        "block2_gradcam",
        "block5_gradcam",
        "cropr_layer2",
        "cropr_layer5",
    )
    finite = all(
        all(np.isfinite(np.asarray(row[name])).all() for name in map_names)
        for row in records.values()
    )
    return records, {"loaders": loaders, "finite": bool(finite), "rows": len(records)}


def _bbox_overlay(rgb: np.ndarray, bbox: np.ndarray) -> Image.Image:
    image = Image.fromarray(np.asarray(rgb, dtype=np.uint8)).convert("RGB")
    draw = ImageDraw.Draw(image)
    center_x, center_y, width, height = [float(value) for value in bbox]
    left = (center_x - width / 2.0) * image.width
    top = (center_y - height / 2.0) * image.height
    right = (center_x + width / 2.0) * image.width
    bottom = (center_y + height / 2.0) * image.height
    draw.rectangle((left, top, right, bottom), outline=(0, 255, 80), width=3)
    return image


def _render_xai_pages(
    *,
    requests: Sequence[Mapping[str, object]],
    control_maps: Mapping[Tuple[str, int], Mapping[str, object]],
    candidate_maps: Mapping[Tuple[str, int], Mapping[str, object]],
    output_dir: Path,
) -> Dict[str, object]:
    pages_dir = output_dir / "xai_pages"
    pages_dir.mkdir(parents=True, exist_ok=False)
    pages: list[str] = []
    represented: set[str] = set()
    for page_index, start in enumerate(range(0, len(requests), 2), start=1):
        chunk = requests[start : start + 2]
        figure, axes = plt.subplots(
            len(chunk) * 2,
            6,
            figsize=(18, max(6.4, 5.8 * len(chunk))),
            squeeze=False,
        )
        for request_index, request in enumerate(chunk):
            key = (str(request["condition"]), int(request["local_index"]))
            categories = [str(value) for value in request.get("categories", [])]
            represented.update(categories)
            for role_index, (role, maps) in enumerate(
                (("control", control_maps), ("candidate", candidate_maps))
            ):
                row_index = request_index * 2 + role_index
                record = maps[key]
                panels = (
                    _bbox_overlay(record["rgb"], record["bbox"]),
                    _heat_overlay(record["rgb"], record["stem_gradcam"]),
                    _heat_overlay(record["rgb"], record["block2_gradcam"]),
                    _heat_overlay(record["rgb"], record["block5_gradcam"]),
                    _heat_overlay(record["rgb"], record["cropr_layer2"]),
                    _heat_overlay(record["rgb"], record["cropr_layer5"]),
                )
                titles = (
                    f"{role} input",
                    f"stem fg={record['stem_foreground_mass']:.3f}",
                    f"block2 fg={record['block2_foreground_mass']:.3f}",
                    f"block5 fg={record['block5_foreground_mass']:.3f}",
                    f"Cropr-L2 fg={record['cropr_layer2_foreground_mass']:.3f}",
                    f"Cropr-L5 fg={record['cropr_layer5_foreground_mass']:.3f}",
                )
                for column, (panel, title) in enumerate(zip(panels, titles)):
                    axes[row_index, column].imshow(panel)
                    axes[row_index, column].set_title(title, fontsize=8)
                    axes[row_index, column].axis("off")
                axes[row_index, 0].set_xlabel(
                    (
                        f"{request['condition']} local={request['local_index']} "
                        f"target={request['target']} pred={record['prediction']}\n"
                        f"{','.join(categories)}"
                    ),
                    fontsize=7,
                )
        figure.tight_layout()
        page = pages_dir / f"page_{page_index:03d}.png"
        figure.savefig(page, dpi=150, bbox_inches="tight")
        plt.close(figure)
        pages.append(str(page.resolve()))
    return {
        "page_count": len(pages),
        "pages": pages,
        "page_sha256": {path: common._sha256(Path(path)) for path in pages},
        "represented_categories": sorted(represented),
    }


def _xai_summary(
    *,
    requests: Sequence[Mapping[str, object]],
    events: Sequence[Mapping[str, object]],
    control_maps: Mapping[Tuple[str, int], Mapping[str, object]],
    candidate_maps: Mapping[Tuple[str, int], Mapping[str, object]],
    render: Mapping[str, object],
) -> Dict[str, object]:
    keys = {
        (str(request["condition"]), int(request["local_index"]))
        for request in requests
    }
    event_keys = {
        (str(event["condition"]), int(event["local_index"])) for event in events
    }
    required_representatives = {
        "representative_close",
        "representative_wide",
        "representative_partial",
        "representative_edge",
        "representative_tiny_object",
        "representative_lighting_dim",
        "representative_lighting_bright",
        "representative_low_contrast",
    }
    represented = set(str(value) for value in render["represented_categories"])
    map_names = (
        "stem_foreground_mass",
        "block2_foreground_mass",
        "block5_foreground_mass",
        "cropr_layer2_foreground_mass",
        "cropr_layer5_foreground_mass",
    )
    means = {
        role: {
            name: float(np.mean([float(maps[key][name]) for key in sorted(keys)]))
            for name in map_names
        }
        for role, maps in (("control", control_maps), ("candidate", candidate_maps))
    }
    finite = all(
        math.isfinite(value)
        for role_payload in means.values()
        for value in role_payload.values()
    )
    return {
        "request_rows": len(keys),
        "event_rows": len(event_keys),
        "all_event_rows_covered": event_keys.issubset(keys),
        "all_representatives_covered": required_representatives.issubset(
            represented
        ),
        "required_representatives": sorted(required_representatives),
        "foreground_mass_means": means,
        "maps_finite": finite,
        "control_standard_prediction_matches": sum(
            int(bool(control_maps[key]["standard_prediction_match"])) for key in keys
        ),
        "candidate_standard_prediction_matches": sum(
            int(bool(candidate_maps[key]["standard_prediction_match"])) for key in keys
        ),
        "control_standard_logit_maximum_error": max(
            float(control_maps[key]["standard_logit_maximum_error"]) for key in keys
        ),
        "candidate_standard_logit_maximum_error": max(
            float(candidate_maps[key]["standard_logit_maximum_error"])
            for key in keys
        ),
        "control_trace_standard_prediction_matches": sum(
            int(bool(control_maps[key]["trace_standard_prediction_match"]))
            for key in keys
        ),
        "candidate_trace_standard_prediction_matches": sum(
            int(bool(candidate_maps[key]["trace_standard_prediction_match"]))
            for key in keys
        ),
        "control_trace_standard_logit_maximum_error": max(
            float(control_maps[key]["trace_standard_logit_maximum_error"])
            for key in keys
        ),
        "candidate_trace_standard_logit_maximum_error": max(
            float(candidate_maps[key]["trace_standard_logit_maximum_error"])
            for key in keys
        ),
        "render": dict(render),
    }


def _gate_checks(
    *,
    failed_attempt: Mapping[str, object],
    correction_replay: Mapping[str, object],
    provenance: Mapping[str, object],
    comparisons: Mapping[str, object],
    selector: Mapping[str, object],
    perturbation: Mapping[str, object],
    xai: Mapping[str, object],
    prediction_replay: Mapping[str, object],
) -> Dict[str, bool]:
    clean = comparisons["clean"]
    delta = clean["delta"]
    transitions = clean["transitions"]
    control_confusion = clean["control"]["confusion_matrix"]
    candidate_confusion = clean["candidate"]["confusion_matrix"]
    control_tp = int(control_confusion[FOCUS_CLASS][FOCUS_CLASS])
    candidate_tp = int(candidate_confusion[FOCUS_CLASS][FOCUS_CLASS])
    restricted_control = int(transitions["restricted_focus_fp_control"])
    restricted_reduction = int(transitions["restricted_focus_fp_reduction"])
    precision_deltas = [
        float(comparisons[name]["delta"]["class1_precision"])
        for name, _, _ in common.CONDITIONS
    ]
    f1_deltas = [
        float(comparisons[name]["delta"]["class1_f1"])
        for name, _, _ in common.CONDITIONS
    ]
    macro_deltas = [
        float(comparisons[name]["delta"]["macro_f1"])
        for name, _, _ in common.CONDITIONS
    ]
    clean_selector = selector["conditions"]["clean"]
    candidate_layers = clean_selector["candidate"]["layers"]
    control_layers = clean_selector["control"]["layers"]
    foreground_gains = [
        float(candidate_layers[str(layer)]["foreground_mass_gain_mean"])
        for layer in (2, 5)
    ]
    recall_deltas = [
        float(
            candidate_layers[str(layer)]["selected_object_patch_recall_mean"]
        )
        - float(control_layers[str(layer)]["selected_object_patch_recall_mean"])
        for layer in (2, 5)
    ]
    shifted_names = [name for name, _, _ in common.CONDITIONS[1:]]
    route_stability = selector["route_stability"]
    conditions = selector["conditions"]
    total_selector_rows = 2 * len(common.CONDITIONS) * 2 * EXPECTED_HOLDOUT_ROWS
    total_xai_rows = int(xai["request_rows"])
    checks = {
        "failed_attempt_preserved": bool(failed_attempt["all_checks_pass"]),
        "unaffected_artifact_replay_exact": bool(
            correction_replay["all_checks_pass"]
        ),
        "provenance": bool(provenance["all_checks_pass"]),
        "prediction_replay_exact": bool(prediction_replay["metrics_exact"]),
        "selector_replay_exact": bool(selector["replay_exact"]),
        "perturbation_replay_exact": bool(perturbation["replay_exact"]),
        "selector_rows_exact": int(selector["rows"]) == total_selector_rows,
        "clean_macro_f1_delta_gte_0": float(delta["macro_f1"]) >= 0.0,
        "clean_class1_f1_delta_gte_0p015": float(delta["class1_f1"])
        >= 0.015,
        "clean_class1_precision_delta_gte_0p025": float(
            delta["class1_precision"]
        )
        >= 0.025,
        "clean_class1_tp_no_more_than_2_lower": candidate_tp >= control_tp - 2,
        "clean_class1_recall_delta_gte_minus_0p010": float(
            delta["class1_recall"]
        )
        >= -0.010,
        "clean_restricted_fp_reduction_gte_4": restricted_reduction >= 4,
        "clean_restricted_fp_reduction_gte_10pct": restricted_control > 0
        and restricted_reduction / restricted_control >= 0.10,
        "clean_corrections_gt_harms": int(transitions["candidate_correction"])
        > int(transitions["candidate_harm"]),
        "clean_nonfocus_f1_drop_lte_0p015": float(
            clean["maximum_nonfocus_f1_drop"]
        )
        <= 0.015,
        "mean_condition_class1_precision_delta_gte_0p015": float(
            np.mean(precision_deltas)
        )
        >= 0.015,
        "condition_class1_f1_no_delta_below_minus_0p010": min(f1_deltas)
        >= -0.010,
        "condition_macro_f1_no_delta_below_minus_0p015": min(macro_deltas)
        >= -0.015,
        "auxiliary_layer2_macro_f1_gte_0p60": float(
            candidate_layers["2"]["auxiliary"]["macro_f1"]
        )
        >= 0.60,
        "auxiliary_layer5_macro_f1_gte_0p60": float(
            candidate_layers["5"]["auxiliary"]["macro_f1"]
        )
        >= 0.60,
        "auxiliary_layer5_class1_f1_gte_0p40": float(
            candidate_layers["5"]["auxiliary"]["per_class_f1"][FOCUS_CLASS]
        )
        >= 0.40,
        "candidate_entropy_not_collapsed": all(
            float(candidate_layers[str(layer)]["high_entropy_fraction_gt_0p995"])
            <= 0.05
            for layer in (2, 5)
        ),
        "candidate_score_variance_not_collapsed": all(
            float(
                candidate_layers[str(layer)][
                    "collapsed_variance_fraction_lte_1e_8"
                ]
            )
            <= 0.05
            for layer in (2, 5)
        ),
        "candidate_foreground_gain_one_layer_gte_0p020": max(foreground_gains)
        >= 0.020,
        "candidate_foreground_gain_other_layer_gte_minus_0p010": min(
            foreground_gains
        )
        >= -0.010,
        "selected_object_recall_delta_gte_minus_0p010": min(recall_deltas)
        >= -0.010,
        "selected_set_shift_jaccard_gte_0p65": all(
            float(
                route_stability[name]["candidate"]["layers"][str(layer)][
                    "mean_jaccard"
                ]
            )
            >= 0.65
            for name in shifted_names
            for layer in (2, 5)
        ),
        "foreground_gain_shift_reversal_lte_0p020": all(
            float(foreground_gains[layer_index])
            - float(
                conditions[name]["candidate"]["layers"][str(layer)][
                    "foreground_mass_gain_mean"
                ]
            )
            <= 0.020
            for name in shifted_names
            for layer_index, layer in enumerate((2, 5))
        ),
        "auxiliary_margin_auroc_clean_gte_0p65": float(
            candidate_layers["5"]["auxiliary_class1_hard_negative_auroc"]
        )
        >= 0.65,
        "auxiliary_margin_auroc_shifted_gte_0p58": all(
            float(
                conditions[name]["candidate"]["layers"]["5"][
                    "auxiliary_class1_hard_negative_auroc"
                ]
            )
            >= 0.58
            for name in shifted_names
        ),
        "selector_standard_trace_parity": float(
            selector["standard_trace"]["maximum_error"]
        )
        <= 1e-6
        and int(selector["standard_trace"]["prediction_matches"])
        == total_selector_rows,
        "selector_partitions_valid": all(
            float(
                conditions[name][role]["layers"][str(layer)][
                    "partition_valid_fraction"
                ]
            )
            == 1.0
            for name, _, _ in common.CONDITIONS
            for role in ("control", "candidate")
            for layer in (2, 5)
        ),
        "selector_routing_roles_exact": all(
            float(
                conditions[name]["control"]["layers"][str(layer)][
                    "routing_fraction"
                ]
            )
            == 0.0
            and float(
                conditions[name]["candidate"]["layers"][str(layer)][
                    "routing_fraction"
                ]
            )
            == 1.0
            for name, _, _ in common.CONDITIONS
            for layer in (2, 5)
        ),
        "object_perturbation_more_causal_than_background": bool(
            perturbation["roles"]["candidate"]["object_more_causal"]
        ),
        "perturbation_restricted_fp_no_increase": bool(
            perturbation["roles"]["candidate"]["restricted_fp_no_increase"]
        ),
        "xai_all_events_covered": bool(xai["all_event_rows_covered"]),
        "xai_all_representatives_covered": bool(
            xai["all_representatives_covered"]
        ),
        "xai_maps_finite": bool(xai["maps_finite"]),
        "xai_standard_prediction_matches": int(
            xai["control_standard_prediction_matches"]
        )
        == total_xai_rows
        and int(xai["candidate_standard_prediction_matches"])
        == total_xai_rows,
        "xai_standard_logit_parity": float(
            xai["control_standard_logit_maximum_error"]
        )
        <= 1e-6
        and float(xai["candidate_standard_logit_maximum_error"]) <= 1e-6,
        "xai_trace_standard_prediction_matches": int(
            xai["control_trace_standard_prediction_matches"]
        )
        == total_xai_rows
        and int(xai["candidate_trace_standard_prediction_matches"])
        == total_xai_rows,
        "xai_trace_standard_logit_parity": float(
            xai["control_trace_standard_logit_maximum_error"]
        )
        <= 1e-6
        and float(xai["candidate_trace_standard_logit_maximum_error"])
        <= 1e-6,
    }
    return checks


def _finalize_visual_review(args: argparse.Namespace) -> Dict[str, object]:
    output_dir = Path(args.output_dir).resolve()
    summary_path = output_dir / "summary.json"
    if not summary_path.is_file():
        raise FileNotFoundError(f"Cropr pair summary does not exist: {summary_path}")
    if args.visual_review_result is None or not str(args.visual_review_note).strip():
        raise ValueError("Visual finalization requires a result and nonempty note.")
    expected = str(args.expected_summary_sha256).strip().lower()
    if len(expected) != 64:
        raise ValueError("Visual finalization requires --expected-summary-sha256.")
    observed = common._sha256(summary_path)
    if observed != expected:
        raise ValueError(
            f"Cropr pair summary changed before review: {observed} != {expected}"
        )
    review_path = output_dir / "visual_review.json"
    if review_path.exists():
        raise FileExistsError(f"Cropr pair visual review already exists: {review_path}")
    summary = common._load_json(summary_path)
    render = summary.get("xai", {}).get("render", {})
    pages = render.get("pages", [])
    expected_hashes = render.get("page_sha256", {})
    if not pages or not isinstance(expected_hashes, Mapping):
        raise ValueError("Cropr pair review lacks XAI pages or hashes.")
    observed_hashes = {}
    for value in pages:
        path = Path(str(value))
        if not path.is_file():
            raise FileNotFoundError(f"Cropr pair XAI page is missing: {path}")
        page_hash = common._sha256(path)
        if page_hash != str(expected_hashes.get(str(path.resolve()), "")):
            raise ValueError(f"Cropr pair XAI page hash differs: {path}")
        observed_hashes[str(path.resolve())] = page_hash
    passed = args.visual_review_result == "pass"
    review = {
        "created_at_utc": datetime.now(timezone.utc).isoformat(),
        "result": args.visual_review_result,
        "passed": passed,
        "note": str(args.visual_review_note).strip(),
        "pre_review_summary_sha256": observed,
        "page_count": len(pages),
        "pages_sha256": observed_hashes,
    }
    common._write_json(review_path, review)
    gate = summary["gate"]
    gate["visual_review_completed"] = True
    gate["visual_review_passed"] = passed
    gate["pair_pass"] = bool(gate["automated_pass"] and passed)
    gate["official_validation_protocol_permission"] = bool(gate["pair_pass"])
    gate["official_validation_permission"] = False
    gate["test_permission"] = False
    summary["visual_review"] = review
    common._write_json(summary_path, summary)
    preflight._write_manifest(output_dir)
    return summary


def run_audit(args: argparse.Namespace) -> Dict[str, object]:
    if int(args.seed) != 42 or int(args.batch_size) != 32:
        raise ValueError("Cropr pair audit is locked to seed=42 and batch_size=32.")
    if int(args.xai_batch_size) <= 0 or int(args.xai_batch_size) > 2:
        raise ValueError("Cropr pair XAI batch size must be in [1,2].")
    if not torch.cuda.is_available():
        raise RuntimeError("Cropr pair audit requires CUDA.")
    declaration = Path(args.declaration).resolve()
    fold_summary_path = Path(args.fold_summary).resolve()
    protocol_path = Path(args.protocol).resolve()
    preflight_path = Path(args.preflight_summary).resolve()
    pair_manifest_path = Path(args.pair_manifest).resolve()
    if common._sha256(declaration) != LOCKED_DECLARATION_SHA256:
        raise ValueError("Cropr pair declaration hash differs from protocol.")
    if common._sha256(fold_summary_path) != LOCKED_FOLD_SUMMARY_SHA256:
        raise ValueError("Cropr pair fold summary hash differs from protocol.")
    if common._sha256(protocol_path) != LOCKED_PROTOCOL_SHA256:
        raise ValueError("Cropr pair protocol hash differs.")
    if common._sha256(preflight_path) != LOCKED_PREFLIGHT_SUMMARY_SHA256:
        raise ValueError("Cropr pair preflight summary hash differs.")
    if common._sha256(pair_manifest_path) != LOCKED_PAIR_MANIFEST_SHA256:
        raise ValueError("Cropr pair manifest hash differs.")
    current_head = _git_value("rev-parse", "HEAD")
    failed_attempt = _verify_failed_attempt(current_head=current_head)
    if not bool(failed_attempt["all_checks_pass"]):
        raise RuntimeError(
            "Cropr failed-attempt preservation checks failed: "
            + ", ".join(
                str(value) for value in failed_attempt["failed_checks"]
            )
        )
    control_checkpoint_path = Path(args.control_checkpoint).resolve()
    candidate_checkpoint_path = Path(args.candidate_checkpoint).resolve()
    provenance = _run_provenance(
        control_run=Path(args.control_run_dir).resolve(),
        candidate_run=Path(args.candidate_run_dir).resolve(),
        control_checkpoint=control_checkpoint_path,
        candidate_checkpoint=candidate_checkpoint_path,
        pair_manifest_path=pair_manifest_path,
        preflight_path=preflight_path,
        fold_summary_path=fold_summary_path,
        protocol_path=protocol_path,
    )
    if not bool(provenance["all_checks_pass"]):
        raise RuntimeError(
            "Cropr pair provenance failed before holdout access: "
            + ", ".join(str(value) for value in provenance["failed_checks"])
        )
    output_dir = _prepare_output(args.output_dir)
    all_rows = common._read_clean_train_rows(declaration)
    holdout_rows = [row for row in all_rows if row.fold == FOLD]
    fit_rows = [row for row in all_rows if row.fold != FOLD]
    if len(holdout_rows) != EXPECTED_HOLDOUT_ROWS or len(fit_rows) != EXPECTED_FIT_ROWS:
        raise ValueError("Cropr pair fit/holdout row counts differ.")
    if common._ordered_index_sha256(
        [row.sample_index for row in fit_rows]
    ) != EXPECTED_FIT_INDEX_SHA256:
        raise ValueError("Cropr pair fit ordered-index hash differs.")
    if common._ordered_index_sha256(
        [row.sample_index for row in holdout_rows]
    ) != EXPECTED_HOLDOUT_INDEX_SHA256:
        raise ValueError("Cropr pair holdout ordered-index hash differs.")

    set_seed(int(args.seed), deterministic=True)
    torch.set_float32_matmul_precision("highest")
    torch.backends.cuda.matmul.allow_tf32 = False
    torch.backends.cudnn.allow_tf32 = False
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False
    device = torch.device("cuda")
    control, control_checkpoint, control_classes = load_model(
        control_checkpoint_path, device
    )
    candidate, candidate_checkpoint, candidate_classes = load_model(
        candidate_checkpoint_path, device
    )
    if control_classes != candidate_classes or len(control_classes) != 5:
        raise ValueError("Cropr pair checkpoint class orders differ.")
    control_config = control_checkpoint.get("model_config", {})
    candidate_config = candidate_checkpoint.get("model_config", {})
    if not bool(control_config.get("cropr_token_selector", False)) or not bool(
        candidate_config.get("cropr_token_selector", False)
    ):
        raise ValueError("Cropr pair checkpoint lacks the selector.")
    if bool(control_config.get("cropr_token_selector_routing", True)):
        raise ValueError("Cropr control checkpoint unexpectedly routes with Cropr.")
    if not bool(candidate_config.get("cropr_token_selector_routing", False)):
        raise ValueError("Cropr candidate checkpoint does not route with Cropr.")
    if common._eval_semantics(control_checkpoint) != common._eval_semantics(
        candidate_checkpoint
    ):
        raise ValueError("Cropr pair evaluation semantics differ.")

    base_dataset, transform, dataset_summary = common._build_holdout_dataset(
        control_checkpoint,
        fold_data=Path(args.fold_data),
        holdout_rows=holdout_rows,
    )
    predictions, prediction_loaders = fusion_pair._predict_conditions(
        control=control,
        candidate=candidate,
        base_dataset=base_dataset,
        transform=transform,
        holdout_rows=holdout_rows,
        device=device,
        batch_size=int(args.batch_size),
        num_workers=int(args.num_workers),
        seed=int(args.seed),
    )
    prediction_path = output_dir / "predictions_all_conditions.csv"
    common._write_predictions(prediction_path, predictions)
    comparisons = common._comparisons(predictions)
    prediction_replay = fusion_pair._prediction_replay(
        prediction_path, comparisons
    )
    prediction_replay.pop("reconstructed")
    selectivity = common._selectivity(rows=all_rows, predictions=predictions)
    selector, selector_rows = _selector_trace_conditions(
        control=control,
        candidate=candidate,
        predictions=predictions,
        base_dataset=base_dataset,
        transform=transform,
        holdout_rows=holdout_rows,
        output_dir=output_dir,
        device=device,
        batch_size=int(args.batch_size),
        num_workers=int(args.num_workers),
        seed=int(args.seed),
        hard_negative_sample_indices=selectivity["hard_negative_sample_indices"],
    )
    perturbation = _perturbation_audit(
        control=control,
        candidate=candidate,
        base_dataset=base_dataset,
        transform=transform,
        holdout_rows=holdout_rows,
        hard_negative_sample_indices=selectivity["hard_negative_sample_indices"],
        output_dir=output_dir,
        device=device,
        batch_size=int(args.batch_size),
        num_workers=int(args.num_workers),
        seed=int(args.seed),
    )
    events = common._build_event_manifest(predictions)
    event_path = output_dir / "event_manifest.csv"
    common._write_event_manifest(event_path, events)
    correction_replay = _compare_failed_artifacts(
        failed_attempt, output_dir=output_dir
    )
    if not bool(correction_replay["all_checks_pass"]):
        raise RuntimeError(
            "Cropr unaffected-artifact replay failed before corrected XAI: "
            + ", ".join(
                str(value) for value in correction_replay["failed_checks"]
            )
        )
    representatives = _representative_requests(
        dataset=base_dataset,
        holdout_rows=holdout_rows,
        positive_sample_indices=selectivity["positive_sample_indices"],
    )
    requests = common._merge_xai_requests(events, representatives)
    standard_maps = {
        role: {
            (condition, int(row["local_index"])): row
            for condition, _, _ in common.CONDITIONS
            for row in predictions[role][condition]
        }
        for role in ("control", "candidate")
    }
    semantics = dataset_summary["semantics"]
    mean = tuple(float(value) for value in semantics["input_mean"])
    std = tuple(float(value) for value in semantics["input_std"])
    control_maps, control_xai = _collect_xai_maps(
        role="control",
        model=control,
        base_dataset=base_dataset,
        transform=transform,
        requests=requests,
        standard_rows=standard_maps["control"],
        device=device,
        batch_size=int(args.xai_batch_size),
        num_workers=int(args.num_workers),
        seed=int(args.seed),
        mean=mean,
        std=std,
    )
    candidate_maps, candidate_xai = _collect_xai_maps(
        role="candidate",
        model=candidate,
        base_dataset=base_dataset,
        transform=transform,
        requests=requests,
        standard_rows=standard_maps["candidate"],
        device=device,
        batch_size=int(args.xai_batch_size),
        num_workers=int(args.num_workers),
        seed=int(args.seed),
        mean=mean,
        std=std,
    )
    render = _render_xai_pages(
        requests=requests,
        control_maps=control_maps,
        candidate_maps=candidate_maps,
        output_dir=output_dir,
    )
    xai = _xai_summary(
        requests=requests,
        events=events,
        control_maps=control_maps,
        candidate_maps=candidate_maps,
        render=render,
    )
    xai["control_collection"] = control_xai
    xai["candidate_collection"] = candidate_xai
    checks = _gate_checks(
        failed_attempt=failed_attempt,
        correction_replay=correction_replay,
        provenance=provenance,
        comparisons=comparisons,
        selector=selector,
        perturbation=perturbation,
        xai=xai,
        prediction_replay=prediction_replay,
    )
    summary: Dict[str, object] = {
        "method": METHOD,
        "created_at_utc": datetime.now(timezone.utc).isoformat(),
        "inputs": {
            "control_checkpoint": str(control_checkpoint_path),
            "control_checkpoint_sha256": common._sha256(control_checkpoint_path),
            "candidate_checkpoint": str(candidate_checkpoint_path),
            "candidate_checkpoint_sha256": common._sha256(
                candidate_checkpoint_path
            ),
            "declaration": str(declaration),
            "declaration_sha256": common._sha256(declaration),
            "fold_data": str(Path(args.fold_data).resolve()),
            "fold_summary": str(fold_summary_path),
            "preflight_summary": str(preflight_path),
            "pair_manifest": str(pair_manifest_path),
            "protocol": str(protocol_path),
        },
        "dataset": dataset_summary,
        "failed_attempt": failed_attempt,
        "correction_replay": correction_replay,
        "provenance": provenance,
        "prediction_loaders": prediction_loaders,
        "comparisons": comparisons,
        "prediction_replay": prediction_replay,
        "selectivity": selectivity,
        "selector": selector,
        "perturbation": perturbation,
        "events": {
            "rows": len(events),
            "path": str(event_path.resolve()),
            "sha256": common._sha256(event_path),
        },
        "xai": xai,
        "predictions": {
            "path": str(prediction_path.resolve()),
            "sha256": common._sha256(prediction_path),
            "rows": 2 * len(common.CONDITIONS) * EXPECTED_HOLDOUT_ROWS,
        },
        "selector_trace_rows_retained_in_memory": len(selector_rows),
        "official_validation_used": False,
        "test_data_used": False,
        "current_best_command_updated": False,
        "gate": {
            "automated_checks": checks,
            "failed_automated_checks": sorted(
                name for name, passed in checks.items() if not bool(passed)
            ),
            "automated_pass": all(checks.values()),
            "visual_review_completed": False,
            "visual_review_passed": False,
            "pair_pass": False,
            "official_validation_protocol_permission": False,
            "official_validation_permission": False,
            "test_permission": False,
        },
    }
    summary_path = output_dir / "summary.json"
    common._write_json(summary_path, summary)
    summary_hash = common._sha256(summary_path)
    common._write_json(
        output_dir / "visual_review_required.json",
        {
            "status": "pending",
            "required": True,
            "page_count": int(render["page_count"]),
            "summary_sha256": summary_hash,
            "finalize_command": (
                "python -m trkh.tools.audit_cropr_token_selector_pair "
                f"--output-dir \"{output_dir}\" --finalize-visual-review "
                f"--expected-summary-sha256 {summary_hash} "
                "--visual-review-result pass|fail --visual-review-note \"...\""
            ),
        },
    )
    preflight._write_manifest(output_dir)
    return summary


def main(argv: Optional[Sequence[str]] = None) -> None:
    args = parse_args(argv)
    if args.finalize_visual_review:
        summary = _finalize_visual_review(args)
    else:
        summary = run_audit(args)
    print(
        json.dumps(
            {
                "method": summary["method"],
                "automated_pass": summary["gate"]["automated_pass"],
                "failed_checks": summary["gate"]["failed_automated_checks"],
                "visual_review_completed": summary["gate"][
                    "visual_review_completed"
                ],
                "pair_pass": summary["gate"]["pair_pass"],
            },
            indent=2,
            sort_keys=True,
        ),
        flush=True,
    )


if __name__ == "__main__":
    main()
