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

import numpy as np
import torch
from PIL import Image, ImageDraw
from torch import Tensor, nn

from trkh.core.utils import set_seed
from trkh.inference.inference import load_model
from trkh.tools import audit_foveal_aggregated_attention_pair as common
from trkh.tools import audit_inattentive_token_fusion_preflight as fusion


METHOD = "inattentive_token_fusion_a0_pair_audit"
PAIR_METHOD = "inattentive_token_fusion_a0_pair"
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
EXPECTED_COHORT_INDEX_SHA256 = (
    "c96cf626348b9ec880d764f2476c01219eb350e26a7deebde675e12c346a33c5"
)
LOCKED_DECLARATION_SHA256 = (
    "2e0993752d58d99ea429bfefe1e2bfe6fa949e45aea1a26cc4bdfee97d4db21c"
)
LOCKED_FOLD_SUMMARY_SHA256 = (
    "2f938b11573073ed957c2522e1a170e6043522f305a787620d1af5f6dbd66763"
)
LOCKED_PROTOCOL_SHA256 = (
    "d3f8de6fbef7ad5d88ca33e5db71f9c5ca587f32c1ad5ee4797fd400aaf3ee4b"
)
FAILED_AUDIT_HEAD = "262c0ec72b813781a666b6e5be04acfb6298a111"
LOCKED_FAILED_ATTEMPT_MANIFEST_SHA256 = (
    "0bb4d7c50880592400dfa61aa1c0d586f1bf5cfa7bc50e75150d278907d5bdce"
)
RUNTIME_PATHS = (
    "trkh/models/model.py",
    "trkh/core/config.py",
    "trkh/training/train.py",
    "scripts/run_trkh_5class_attention_views_v8.ps1",
    "docs/TRKH_5CLASS_INATTENTIVE_TOKEN_FUSION_READINESS_PROTOCOL_20260716.md",
)


def parse_args(argv: Optional[Sequence[str]] = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Locked train-only post-run audit for the EViT-style inattentive-token "
            "fusion A0 pair. Official validation and test are forbidden."
        )
    )
    parser.add_argument(
        "--control-checkpoint",
        type=Path,
        default=Path(
            "runs/probe_inattentive_fusion_a0_control_5e_20260716/"
            "checkpoints/best.pt"
        ),
    )
    parser.add_argument(
        "--candidate-checkpoint",
        type=Path,
        default=Path(
            "runs/probe_inattentive_fusion_a0_candidate_5e_20260716/"
            "checkpoints/best.pt"
        ),
    )
    parser.add_argument(
        "--control-run-dir",
        type=Path,
        default=Path("runs/probe_inattentive_fusion_a0_control_5e_20260716"),
    )
    parser.add_argument(
        "--candidate-run-dir",
        type=Path,
        default=Path("runs/probe_inattentive_fusion_a0_candidate_5e_20260716"),
    )
    parser.add_argument(
        "--pair-manifest",
        type=Path,
        default=Path("runs/inattentive_fusion_a0_pair_manifest_20260716.json"),
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
        default=Path("runs/audit_inattentive_fusion_preflight_20260716/summary.json"),
    )
    parser.add_argument(
        "--protocol",
        type=Path,
        default=Path(
            "docs/TRKH_5CLASS_INATTENTIVE_TOKEN_FUSION_READINESS_PROTOCOL_20260716.md"
        ),
    )
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--batch-size", type=int, default=32)
    parser.add_argument("--xai-batch-size", type=int, default=2)
    parser.add_argument("--num-workers", type=int, default=4)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--failed-attempt-manifest", type=Path)
    parser.add_argument("--finalize-visual-review", action="store_true")
    parser.add_argument("--visual-review-result", choices=("pass", "fail"))
    parser.add_argument("--visual-review-note", type=str, default="")
    parser.add_argument("--expected-summary-sha256", type=str, default="")
    return parser.parse_args(argv)


def _prepare_output(path: Path) -> Path:
    resolved = Path(path).resolve()
    if resolved.exists() and any(resolved.iterdir()):
        raise FileExistsError(f"Fusion pair audit output is not empty: {resolved}")
    resolved.mkdir(parents=True, exist_ok=True)
    return resolved


def _git_value(*args: str) -> str:
    return subprocess.check_output(
        ["git", *args], text=True, encoding="utf-8"
    ).strip()


def _approx(value: object, expected: float, tolerance: float = 1e-12) -> bool:
    try:
        return abs(float(value) - float(expected)) <= tolerance
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
    if normalized.count("--inattentive-token-fusion") > 1:
        raise ValueError("Fusion flag occurs more than once in the train arguments.")
    return [value for value in normalized if value != "--inattentive-token-fusion"]


def _evidence_head(preflight: Mapping[str, object]) -> str:
    replay = preflight.get("postflight_replay")
    if isinstance(replay, Mapping):
        replay_git = replay.get("git")
        if isinstance(replay_git, Mapping):
            return str(replay_git.get("head", ""))
    git = preflight.get("git")
    return str(git.get("head", "")) if isinstance(git, Mapping) else ""


def _verify_failed_attempt(
    manifest_path: Path, *, current_head: str
) -> Dict[str, object]:
    resolved = Path(manifest_path).resolve()
    if common._sha256(resolved) != LOCKED_FAILED_ATTEMPT_MANIFEST_SHA256:
        raise ValueError("Failed pair-audit manifest hash differs.")
    manifest = common._load_json(resolved)
    artifacts = manifest.get("artifacts")
    if not isinstance(artifacts, list):
        raise ValueError("Failed pair-audit manifest lacks artifacts.")
    expected_names = {
        "context_rows_all_conditions.csv",
        "event_manifest.csv",
        "failure_record.json",
        "predictions_all_conditions.csv",
        "tiny_edge_cohort.json",
    }
    artifact_map = {
        str(row.get("name", "")): row
        for row in artifacts
        if isinstance(row, Mapping)
    }
    changed_paths = {
        value.strip().replace("\\", "/")
        for value in _git_value(
            "diff", "--name-only", FAILED_AUDIT_HEAD, current_head
        ).splitlines()
        if value.strip()
    }
    allowed_correction_paths = {
        "scripts/run_trkh_inattentive_token_fusion_a0.ps1",
        "tests/test_audit_inattentive_token_fusion_pair.py",
        "trkh/tools/audit_inattentive_token_fusion_pair.py",
    }
    checks: Dict[str, bool] = {
        "manifest_method": manifest.get("method")
        == "inattentive_token_fusion_a0_pair_failed_attempt_manifest",
        "failed_head_exact": manifest.get("audit_git_head") == FAILED_AUDIT_HEAD,
        "artifact_inventory_exact": set(artifact_map) == expected_names,
        "failed_output_has_no_summary": not (resolved.parent / "summary.json").exists(),
        "raw_data_unmodified": manifest.get("raw_data_modified") is False,
        "validation_unused": manifest.get("validation_used") is False,
        "test_unused": manifest.get("test_used") is False,
        "correction_commit_is_new": current_head != FAILED_AUDIT_HEAD,
        "correction_delta_nonempty": bool(changed_paths),
        "correction_delta_scoped": changed_paths.issubset(
            allowed_correction_paths
        ),
    }
    for name in sorted(expected_names):
        row = artifact_map.get(name, {})
        path = resolved.parent / name
        checks[f"source_{name}_exists"] = path.is_file()
        checks[f"source_{name}_bytes"] = path.is_file() and int(
            path.stat().st_size
        ) == int(row.get("bytes", -1))
        checks[f"source_{name}_sha256"] = path.is_file() and common._sha256(
            path
        ) == str(row.get("sha256", ""))
    return {
        "manifest": str(resolved),
        "manifest_sha256": common._sha256(resolved),
        "failed_output_dir": str(resolved.parent),
        "failed_audit_head": FAILED_AUDIT_HEAD,
        "correction_audit_head": current_head,
        "changed_paths": sorted(changed_paths),
        "checks": checks,
        "all_checks_pass": all(checks.values()),
        "failed_checks": sorted(
            name for name, passed in checks.items() if not passed
        ),
        "artifacts": artifact_map,
    }


def _compare_correction_artifacts(
    failed_attempt: Mapping[str, object], *, output_dir: Path
) -> Dict[str, object]:
    expected = failed_attempt.get("artifacts")
    if not isinstance(expected, Mapping):
        raise ValueError("Failed-attempt evidence lacks artifact hashes.")
    names = (
        "context_rows_all_conditions.csv",
        "event_manifest.csv",
        "predictions_all_conditions.csv",
        "tiny_edge_cohort.json",
    )
    checks: Dict[str, bool] = {}
    observed: Dict[str, object] = {}
    for name in names:
        row = expected.get(name)
        if not isinstance(row, Mapping):
            raise ValueError(f"Failed-attempt evidence lacks {name}.")
        path = output_dir / name
        exists = path.is_file()
        size = int(path.stat().st_size) if exists else -1
        digest = common._sha256(path) if exists else ""
        checks[f"replay_{name}_exists"] = exists
        checks[f"replay_{name}_bytes_exact"] = size == int(
            row.get("bytes", -1)
        )
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
        "replayed_artifacts": observed,
    }


def _run_provenance(
    *,
    control_run: Path,
    candidate_run: Path,
    pair_manifest_path: Path,
    preflight_path: Path,
    preflight: Mapping[str, object],
    fold_summary_path: Path,
    fold_summary: Mapping[str, object],
    protocol_path: Path,
) -> Dict[str, object]:
    pair_manifest = common._load_json(pair_manifest_path)
    pair_head = str(pair_manifest.get("git_head", ""))
    current_head = _git_value("rev-parse", "HEAD")
    current_upstream = _git_value("rev-parse", "@{upstream}")
    runtime_diff = subprocess.run(
        ["git", "diff", "--quiet", pair_head, current_head, "--", *RUNTIME_PATHS],
        check=False,
    )
    tracked_status = _git_value("status", "--short", "--untracked-files=no")
    preflight_gate = preflight.get("gate")
    preflight_gate = preflight_gate if isinstance(preflight_gate, Mapping) else {}
    checks: Dict[str, bool] = {
        "preflight_automated_pass": bool(preflight_gate.get("automated_pass")),
        "preflight_visual_pass": bool(preflight_gate.get("visual_review_passed")),
        "preflight_pair_permission": bool(
            preflight_gate.get("formal_pair_permission")
        ),
        "pair_manifest_method": pair_manifest.get("method") == PAIR_METHOD,
        "pair_manifest_head_matches_preflight": pair_head
        == _evidence_head(preflight),
        "pair_manifest_control_path": Path(
            str(pair_manifest.get("control_run", ""))
        ).resolve()
        == control_run.resolve(),
        "pair_manifest_candidate_path": Path(
            str(pair_manifest.get("candidate_run", ""))
        ).resolve()
        == candidate_run.resolve(),
        "pair_manifest_causal_difference": pair_manifest.get(
            "sole_causal_difference"
        )
        == "inattentive_token_fusion_false_vs_true",
        "pair_manifest_validation_forbidden": pair_manifest.get(
            "official_validation_used"
        )
        is False,
        "pair_manifest_test_forbidden": pair_manifest.get("test_used") is False,
        "pair_manifest_commands_unchanged": pair_manifest.get(
            "current_best_command_updated"
        )
        is False,
        "preflight_summary_path": Path(
            str(pair_manifest.get("preflight_summary", ""))
        ).resolve()
        == preflight_path.resolve(),
        "fold_summary_hash": common._sha256(fold_summary_path)
        == LOCKED_FOLD_SUMMARY_SHA256,
        "protocol_hash": common._sha256(protocol_path) == LOCKED_PROTOCOL_SHA256,
        "fold_fit_rows": int(fold_summary.get("fit_rows", -1))
        == EXPECTED_FIT_ROWS,
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
        "tracked_worktree_clean": tracked_status == "",
        "head_pushed": current_head == current_upstream,
        "runtime_files_unchanged_since_pair": runtime_diff.returncode == 0,
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
    }
    run_payloads: Dict[str, object] = {}
    occurrences: Dict[str, Mapping[str, object]] = {}
    normalized_args: Dict[str, list[str]] = {}
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
        role_checks: Dict[str, bool] = {}
        for key, expected in exact_launcher.items():
            observed = launcher.get(key)
            role_checks[f"launcher_{key}"] = (
                _approx(observed, expected)
                if isinstance(expected, float)
                else observed == expected
            )
        train_config = resolved.get("train_config")
        model_config = resolved.get("model_config")
        if not isinstance(train_config, Mapping) or not isinstance(
            model_config, Mapping
        ):
            raise ValueError(f"{role} resolved config lacks train/model config.")
        expected_fusion = role == "candidate"
        role_checks.update(
            {
                "launcher_train_args_epochs": _cli_value(train_args, "--epochs")
                == "5",
                "launcher_train_args_patience": _cli_value(
                    train_args, "--patience"
                )
                == "3",
                "launcher_fusion_role": bool(
                    launcher.get("inattentive_token_fusion", False)
                )
                == expected_fusion,
                "train_args_fusion_role": train_args.count(
                    "--inattentive-token-fusion"
                )
                == int(expected_fusion),
                "resolved_fusion_role": bool(
                    model_config.get("inattentive_token_fusion", False)
                )
                == expected_fusion,
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
                "resolved_token_pruning": model_config.get("token_pruning")
                is True,
                "resolved_prune_layers": str(
                    model_config.get("token_prune_layers", "")
                )
                == "2,5",
                "resolved_keep_rates": str(
                    model_config.get("token_keep_rates", "")
                )
                == "0.85,0.65",
                "five_history_epochs": len(history_rows) == 5,
                "last_epoch_exact": int(summary.get("last_epoch", -1)) == 5,
                "stop_completed": summary.get("stop_reason") == "completed",
                "test_summary_absent": summary.get("test_summary") is None,
                "architecture_trace_completed": summary.get(
                    "architecture_trace", {}
                ).get("status")
                == "completed",
                "parameter_count_exact": int(summary.get("parameter_count", -1))
                == 7_245_590,
                "resolved_resume_not_loaded": not bool(
                    resolved.get("resume", {}).get("loaded", True)
                ),
                "best_checkpoint_exists": (
                    run_dir / "checkpoints" / "best.pt"
                ).is_file(),
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
            "summary": summary,
            "checks": role_checks,
        }
    control_epochs = occurrences["control"].get("epochs", [])
    candidate_epochs = occurrences["candidate"].get("epochs", [])
    checks["normalized_train_args_equal_except_fusion_role"] = (
        normalized_args["control"] == normalized_args["candidate"]
    )
    checks["occurrence_epoch_records_equal"] = control_epochs == candidate_epochs
    checks["occurrence_five_epochs"] = (
        len(control_epochs) == len(candidate_epochs) == 5
    )
    checks["occurrence_rows_exact"] = all(
        int(row.get("occurrences", -1)) == EXPECTED_FIT_ROWS
        and int(row.get("unique_sample_indices", -1)) == EXPECTED_FIT_ROWS
        and row.get("class_counts") == EXPECTED_FIT_COUNTS
        for row in control_epochs
    )
    return {
        "checks": checks,
        "all_checks_pass": all(checks.values()),
        "failed_checks": sorted(name for name, passed in checks.items() if not passed),
        "pair_git_head": pair_head,
        "audit_git_head": current_head,
        "audit_git_upstream": current_upstream,
        "runtime_paths": list(RUNTIME_PATHS),
        "pair_manifest": pair_manifest,
        "pair_manifest_sha256": common._sha256(pair_manifest_path),
        "preflight_summary_sha256": common._sha256(preflight_path),
        "runs": run_payloads,
        "occurrence": occurrences,
    }


def _predict_conditions(
    *,
    control: nn.Module,
    candidate: nn.Module,
    base_dataset,
    transform,
    holdout_rows,
    device: torch.device,
    batch_size: int,
    num_workers: int,
    seed: int,
) -> Tuple[Dict[str, Dict[str, list[Dict[str, object]]]], Dict[str, object]]:
    output: Dict[str, Dict[str, list[Dict[str, object]]]] = {
        "control": {},
        "candidate": {},
    }
    loader_summaries: Dict[str, object] = {}
    local_indices = list(range(len(base_dataset)))
    amp_enabled = device.type == "cuda"
    for condition_index, (condition, brightness, contrast) in enumerate(
        common.CONDITIONS
    ):
        condition_dataset = common._SelectedConditionDataset(
            base_dataset,
            local_indices,
            corruption=common._condition_corruption(
                condition, brightness, contrast
            ),
            transform=transform,
        )
        loader, loader_summary = common._make_loader(
            dataset=condition_dataset,
            batch_size=batch_size,
            num_workers=num_workers,
            context=f"inattentive_fusion_pair_{condition}_standard",
            seed=seed + 100 + condition_index,
        )
        loader_summaries[condition] = loader_summary
        role_rows = {"control": [], "candidate": []}
        processed = 0
        with torch.inference_mode():
            for images_cpu, targets_cpu, metadata_cpu in loader:
                images = images_cpu.to(device=device, non_blocking=True)
                metadata = common._metadata_to_device(metadata_cpu, device)
                sample_indices = metadata_cpu.get("sample_index")
                if not torch.is_tensor(sample_indices):
                    raise ValueError("Pair prediction metadata lacks sample_index.")
                for role, model in (("control", control), ("candidate", candidate)):
                    with torch.autocast(
                        device_type=device.type,
                        dtype=torch.bfloat16,
                        enabled=amp_enabled,
                    ):
                        logits, _ = common._forward_classification_with_metadata(
                            model, images, metadata, device=device
                        )
                    logits = logits.float()
                    probabilities = logits.softmax(dim=1)
                    predictions = probabilities.argmax(dim=1)
                    margins = common._margin(logits)
                    for position, local_value in enumerate(sample_indices.tolist()):
                        local_index = int(local_value)
                        source = holdout_rows[local_index]
                        target = int(targets_cpu[position].item())
                        if target != int(source.target):
                            raise ValueError("Pair prediction target order differs.")
                        row: Dict[str, object] = {
                            "role": role,
                            "condition": condition,
                            "local_index": local_index,
                            "sample_index": int(source.sample_index),
                            "source_stem": source.source_stem,
                            "object_index": int(
                                base_dataset.samples[
                                    local_index
                                ].primary_object_index
                            ),
                            "target": target,
                            "prediction": int(predictions[position].item()),
                            "class1_restricted_margin": float(
                                margins[position].item()
                            ),
                        }
                        for class_index in range(5):
                            row[f"logit_{class_index}"] = float(
                                logits[position, class_index].item()
                            )
                            row[f"prob_{class_index}"] = float(
                                probabilities[position, class_index].item()
                            )
                        role_rows[role].append(row)
                processed += int(images.size(0))
                if processed % 512 < int(images.size(0)) or processed == len(
                    condition_dataset
                ):
                    print(
                        json.dumps(
                            {
                                "method": METHOD,
                                "phase": "standard_predictions",
                                "condition": condition,
                                "processed": processed,
                                "rows": len(condition_dataset),
                            }
                        ),
                        flush=True,
                    )
        for role in ("control", "candidate"):
            observed = [int(row["local_index"]) for row in role_rows[role]]
            if observed != local_indices:
                raise ValueError(f"Prediction order differs for {role}/{condition}.")
            output[role][condition] = role_rows[role]
    return output, loader_summaries


def _load_locked_cohort(
    preflight: Mapping[str, object], *, output_dir: Path
) -> Dict[str, object]:
    data = preflight.get("data")
    if not isinstance(data, Mapping):
        raise ValueError("Preflight summary lacks data provenance.")
    cohort = data.get("cohort")
    if not isinstance(cohort, Mapping):
        raise ValueError("Preflight summary lacks the frozen tiny/edge cohort.")
    source_path = Path(str(cohort.get("path", ""))).resolve()
    expected_hash = str(cohort.get("sha256", ""))
    if not source_path.is_file() or common._sha256(source_path) != expected_hash:
        raise ValueError("Frozen tiny/edge cohort artifact changed after preflight.")
    payload = common._load_json(source_path)
    if payload.get("computed_before_model_scoring") is not True:
        raise ValueError("Tiny/edge cohort was not frozen before model scoring.")
    if payload.get("ordered_sample_index_sha256") != EXPECTED_COHORT_INDEX_SHA256:
        raise ValueError("Tiny/edge cohort ordered-index hash differs.")
    if int(payload.get("cohort_count", -1)) != 1_123:
        raise ValueError("Tiny/edge cohort count differs from the locked preflight.")
    copied_path = output_dir / "tiny_edge_cohort.json"
    copied_path.write_bytes(source_path.read_bytes())
    if common._sha256(copied_path) != expected_hash:
        raise RuntimeError("Copied tiny/edge cohort hash differs.")
    payload["path"] = str(copied_path.resolve())
    payload["sha256"] = expected_hash
    return payload


def _cohort_comparison(
    predictions: Mapping[str, Mapping[str, Sequence[Mapping[str, object]]]],
    cohort: Mapping[str, object],
) -> Dict[str, object]:
    selected = set(int(value) for value in cohort["local_holdout_indices"])
    control = [
        row
        for row in predictions["control"]["clean"]
        if int(row["local_index"]) in selected
    ]
    candidate = [
        row
        for row in predictions["candidate"]["clean"]
        if int(row["local_index"]) in selected
    ]
    if len(control) != len(candidate) or len(control) != int(
        cohort["cohort_count"]
    ):
        raise ValueError("Tiny/edge prediction coverage differs from the frozen cohort.")
    if [int(row["local_index"]) for row in control] != [
        int(row["local_index"]) for row in candidate
    ]:
        raise ValueError("Tiny/edge control/candidate order differs.")
    result = common._comparison(
        control_rows=control,
        candidate_rows=candidate,
        num_classes=5,
        focus_class=FOCUS_CLASS,
    )
    result["rows"] = len(control)
    result["ordered_sample_index_sha256"] = common._ordered_index_sha256(
        [int(row["sample_index"]) for row in control]
    )
    return result


def _context_condition_summary(
    rows: Sequence[Mapping[str, object]],
) -> Dict[str, object]:
    if not rows:
        raise ValueError("Cannot summarize empty fusion context rows.")

    def mean(name: str) -> float:
        values = np.asarray([float(row[name]) for row in rows], dtype=np.float64)
        finite = values[np.isfinite(values)]
        return float(finite.mean()) if finite.size else math.nan

    cohort_mask = np.asarray(
        [bool(row["tiny_edge_cohort"]) for row in rows], dtype=bool
    )
    nonzero = np.asarray(
        [bool(row["dropped_object_mass_nonzero"]) for row in rows], dtype=bool
    )
    cosines = np.asarray(
        [
            float(row["clean_context_cosine"])
            for row in rows
            if row["clean_context_cosine"] not in (None, "")
        ],
        dtype=np.float64,
    )
    return {
        "rows": len(rows),
        "finite_rows": sum(int(bool(row["finite_row"])) for row in rows),
        "standard_trace_maximum_error": float(
            max(float(row["standard_trace_maximum_error"]) for row in rows)
        ),
        "standard_trace_prediction_matches": sum(
            int(bool(row["standard_trace_prediction_match"])) for row in rows
        ),
        "partitions_valid_fraction": mean("partitions_valid"),
        "stage1_positive_finite_mass_fraction": mean(
            "stage1_positive_finite_mass"
        ),
        "stage2_positive_finite_mass_fraction": mean(
            "stage2_positive_finite_mass"
        ),
        "maximum_mass_replay_error": float(
            max(
                max(
                    float(row["stage1_mass_replay_error"]),
                    float(row["stage2_mass_replay_error"]),
                )
                for row in rows
            )
        ),
        "context_norm_mean": mean("context_norm"),
        "weighted_object_fraction_mean": mean("weighted_object_fraction"),
        "raw_dropped_object_fraction_mean": mean(
            "raw_dropped_object_fraction"
        ),
        "weighted_object_gain_mean": mean("weighted_object_gain"),
        "outside_context_fraction_mean": mean("outside_context_fraction"),
        "tiny_edge_rows": int(cohort_mask.sum()),
        "tiny_edge_nonzero_object_mass_fraction": float(
            nonzero[cohort_mask].mean()
        ),
        "clean_context_cosine_mean": (
            float(cosines.mean()) if cosines.size else None
        ),
        "clean_context_cosine_minimum": (
            float(cosines.min()) if cosines.size else None
        ),
    }


def _trace_context_conditions(
    *,
    candidate: nn.Module,
    predictions: Mapping[str, Mapping[str, Sequence[Mapping[str, object]]]],
    base_dataset,
    transform,
    holdout_rows,
    cohort: Mapping[str, object],
    output_dir: Path,
    device: torch.device,
    batch_size: int,
    num_workers: int,
    seed: int,
) -> Tuple[Dict[str, object], Sequence[Mapping[str, object]]]:
    cohort_indices = set(int(value) for value in cohort["local_holdout_indices"])
    clean_context = torch.empty(EXPECTED_HOLDOUT_ROWS, 256, dtype=torch.float32)
    clean_seen = torch.zeros(EXPECTED_HOLDOUT_ROWS, dtype=torch.bool)
    all_rows: list[Dict[str, object]] = []
    summaries: Dict[str, object] = {}
    loaders: Dict[str, object] = {}
    amp_enabled = device.type == "cuda"
    for condition_index, (condition, brightness, contrast) in enumerate(
        common.CONDITIONS
    ):
        condition_dataset = common._SelectedConditionDataset(
            base_dataset,
            list(range(len(base_dataset))),
            corruption=common._condition_corruption(
                condition, brightness, contrast
            ),
            transform=transform,
        )
        loader, loader_summary = common._make_loader(
            dataset=condition_dataset,
            batch_size=batch_size,
            num_workers=num_workers,
            context=f"inattentive_fusion_pair_{condition}_trace",
            seed=seed + 300 + condition_index,
        )
        loaders[condition] = loader_summary
        standard_rows = {
            int(row["local_index"]): row
            for row in predictions["candidate"][condition]
        }
        condition_rows: list[Dict[str, object]] = []
        processed = 0
        with torch.inference_mode():
            for images_cpu, targets_cpu, metadata_cpu in loader:
                images = images_cpu.to(device=device, non_blocking=True)
                metadata = common._metadata_to_device(metadata_cpu, device)
                with torch.autocast(
                    device_type=device.type,
                    dtype=torch.bfloat16,
                    enabled=amp_enabled,
                ):
                    trace_logits, features = fusion._forward(
                        candidate, images, metadata, return_trace=True
                    )
                trace_logits = trace_logits.float().cpu()
                pruning = features.get("trace", {}).get("pruning")
                contexts = features.get("inattentive_context")
                if not isinstance(pruning, list) or len(pruning) != 2:
                    raise ValueError("Candidate trace lacks two fusion prune stages.")
                if not torch.is_tensor(contexts) or tuple(contexts.shape[1:]) != (
                    1,
                    256,
                ):
                    raise ValueError("Candidate context does not have shape [B,1,256].")
                contexts_cpu = contexts.detach().float().cpu()
                sample_indices = metadata_cpu.get("sample_index")
                bboxes = metadata_cpu.get("bbox")
                if not torch.is_tensor(sample_indices) or not torch.is_tensor(bboxes):
                    raise ValueError("Context trace metadata lacks sample_index/bbox.")
                for position, local_value in enumerate(sample_indices.tolist()):
                    local_index = int(local_value)
                    source = holdout_rows[local_index]
                    target = int(targets_cpu[position].item())
                    if target != int(source.target):
                        raise ValueError("Context trace target order differs.")
                    standard = standard_rows[local_index]
                    standard_logits = torch.tensor(
                        [float(standard[f"logit_{index}"]) for index in range(5)]
                    )
                    error = float(
                        (standard_logits - trace_logits[position]).abs().amax().item()
                    )
                    overlap = fusion._bbox_patch_overlap(bboxes[position])
                    lineage = fusion._fusion_lineage(pruning, position, overlap)
                    stage1, stage2 = lineage["stages"]
                    geometry = fusion._bbox_geometry(bboxes[position])
                    context_vector = contexts_cpu[position, 0]
                    if condition == "clean":
                        clean_context[local_index] = context_vector
                        clean_seen[local_index] = True
                        cosine: object = ""
                    else:
                        if not bool(clean_seen[local_index]):
                            raise RuntimeError("Clean context cache is incomplete.")
                        cosine = float(
                            torch.nn.functional.cosine_similarity(
                                clean_context[local_index].unsqueeze(0),
                                context_vector.unsqueeze(0),
                                dim=1,
                            ).item()
                        )
                    finite_row = bool(
                        torch.isfinite(trace_logits[position]).all()
                        and torch.isfinite(context_vector).all()
                        and math.isfinite(float(lineage["lineage_total"]))
                        and math.isfinite(float(lineage["weighted_object_fraction"]))
                        and math.isfinite(float(lineage["outside_fraction"]))
                    )
                    row: Dict[str, object] = {
                        "condition": condition,
                        "local_holdout_index": local_index,
                        "sample_index": int(source.sample_index),
                        "source_stem": source.source_stem,
                        "object_index": int(
                            base_dataset.samples[local_index].primary_object_index
                        ),
                        "target": target,
                        "control_prediction": int(
                            predictions["control"][condition][local_index]["prediction"]
                        ),
                        "candidate_prediction": int(standard["prediction"]),
                        "trace_prediction": int(
                            trace_logits[position].argmax().item()
                        ),
                        "standard_trace_maximum_error": error,
                        "standard_trace_prediction_match": int(
                            standard["prediction"]
                        )
                        == int(trace_logits[position].argmax().item()),
                        "bbox_area": float(geometry["area"]),
                        "bbox_edge_gap": float(geometry["edge_gap"]),
                        "tiny_edge_cohort": local_index in cohort_indices,
                        "partitions_valid": bool(lineage["partitions_valid"]),
                        "stage1_context_mass": float(stage1["reported_mass"]),
                        "stage2_context_mass": float(stage2["reported_mass"]),
                        "stage1_positive_finite_mass": bool(
                            stage1["finite_nonnegative"]
                            and float(stage1["reported_mass"]) > 0.0
                        ),
                        "stage2_positive_finite_mass": bool(
                            stage2["finite_nonnegative"]
                            and float(stage2["reported_mass"]) > 0.0
                        ),
                        "stage1_mass_replay_error": float(
                            stage1["mass_replay_error"]
                        ),
                        "stage2_mass_replay_error": float(
                            stage2["mass_replay_error"]
                        ),
                        "context_norm": float(context_vector.norm().item()),
                        "effective_context_lineage_mass": float(
                            lineage["lineage_total"]
                        ),
                        "dropped_object_attention_mass": float(
                            lineage["object_mass"]
                        ),
                        "dropped_object_mass_nonzero": float(
                            lineage["object_mass"]
                        )
                        > 1e-12,
                        "weighted_object_fraction": float(
                            lineage["weighted_object_fraction"]
                        ),
                        "raw_dropped_object_fraction": float(
                            lineage["raw_object_fraction"]
                        ),
                        "weighted_object_gain": float(
                            lineage["weighted_object_fraction"]
                            - lineage["raw_object_fraction"]
                        ),
                        "outside_context_fraction": float(
                            lineage["outside_fraction"]
                        ),
                        "clean_context_cosine": cosine,
                        "finite_row": finite_row,
                        "stage1_kept_indices": fusion._serialize_numbers(
                            stage1["kept"]
                        ),
                        "stage1_dropped_indices": fusion._serialize_numbers(
                            stage1["dropped"]
                        ),
                        "stage1_fusion_weights": fusion._serialize_numbers(
                            stage1["weights"]
                        ),
                        "stage2_kept_indices": fusion._serialize_numbers(
                            stage2["kept"]
                        ),
                        "stage2_dropped_indices": fusion._serialize_numbers(
                            stage2["dropped"]
                        ),
                        "stage2_fusion_weights": fusion._serialize_numbers(
                            stage2["weights"]
                        ),
                        "stage2_previous_context_attention": float(
                            stage2["previous_mass"]
                        ),
                    }
                    condition_rows.append(row)
                    all_rows.append(row)
                processed += int(images.size(0))
                if processed % 512 < int(images.size(0)) or processed == len(
                    condition_dataset
                ):
                    print(
                        json.dumps(
                            {
                                "method": METHOD,
                                "phase": "candidate_context_trace",
                                "condition": condition,
                                "processed": processed,
                                "rows": len(condition_dataset),
                            }
                        ),
                        flush=True,
                    )
        summaries[condition] = _context_condition_summary(condition_rows)
    if not bool(clean_seen.all()):
        raise RuntimeError("Clean context cache did not cover the holdout.")
    rows_path = output_dir / "context_rows_all_conditions.csv"
    fusion._write_rows_csv(rows_path, all_rows)
    return {
        "conditions": summaries,
        "loader": loaders,
        "rows": len(all_rows),
        "rows_csv": str(rows_path.resolve()),
        "rows_csv_sha256": common._sha256(rows_path),
    }, all_rows


def _prediction_replay(
    prediction_path: Path,
    expected: Mapping[str, object],
) -> Dict[str, object]:
    reconstructed: Dict[str, Dict[str, list[Dict[str, object]]]] = {
        "control": {name: [] for name, _, _ in common.CONDITIONS},
        "candidate": {name: [] for name, _, _ in common.CONDITIONS},
    }
    with prediction_path.open("r", encoding="utf-8", newline="") as handle:
        for row in csv.DictReader(handle):
            reconstructed[str(row["role"])][str(row["condition"])].append(row)
    replayed = common._comparisons(reconstructed)

    def maximum_error(left: object, right: object) -> float:
        if isinstance(left, Mapping) and isinstance(right, Mapping):
            if set(left) != set(right):
                return math.inf
            return max(
                (maximum_error(left[key], right[key]) for key in left), default=0.0
            )
        if isinstance(left, list) and isinstance(right, list):
            if len(left) != len(right):
                return math.inf
            return max(
                (maximum_error(a, b) for a, b in zip(left, right)), default=0.0
            )
        try:
            return abs(float(left) - float(right))
        except (TypeError, ValueError):
            return 0.0 if left == right else math.inf

    error = maximum_error(expected, replayed)
    replay_events = common._build_event_manifest(reconstructed)
    return {
        "rows": sum(
            len(reconstructed[role][condition])
            for role in ("control", "candidate")
            for condition, _, _ in common.CONDITIONS
        ),
        "maximum_metric_error": float(error),
        "metrics_exact": math.isfinite(error) and error <= 1e-12,
        "event_rows": len(replay_events),
        "reconstructed": reconstructed,
    }


def _representative_requests(
    *,
    dataset,
    holdout_rows,
    positive_sample_indices: Sequence[int],
    cohort: Mapping[str, object],
) -> list[Dict[str, object]]:
    requests = common._representative_requests(
        dataset=dataset,
        holdout_rows=holdout_rows,
        positive_sample_indices=positive_sample_indices,
    )
    cohort_indices = [int(value) for value in cohort["local_holdout_indices"]]
    tiny_local = min(
        cohort_indices,
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


def _collect_stem_gradcam(
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
    amp_enabled = device.type == "cuda"
    for condition_index, (condition, selected) in enumerate(by_condition.items()):
        brightness, contrast = condition_specs[condition]
        dataset = common._SelectedConditionDataset(
            base_dataset,
            sorted(set(selected)),
            corruption=common._condition_corruption(
                condition, brightness, contrast
            ),
            transform=transform,
        )
        loader, loader_summary = common._make_loader(
            dataset=dataset,
            batch_size=batch_size,
            num_workers=num_workers,
            context=f"inattentive_fusion_{role}_{condition}_gradcam",
            seed=seed + 700 + condition_index,
        )
        loaders[condition] = loader_summary
        for images_cpu, _targets, metadata_cpu in loader:
            images = images_cpu.to(device=device, non_blocking=True).requires_grad_(
                True
            )
            metadata = common._metadata_to_device(metadata_cpu, device)
            sample_indices = metadata_cpu.get("sample_index")
            bboxes = metadata_cpu.get("bbox")
            if not torch.is_tensor(sample_indices) or not torch.is_tensor(bboxes):
                raise ValueError("Grad-CAM metadata lacks sample_index/bbox.")
            captured: Dict[str, Tensor] = {}

            def stem_hook(_module, _inputs, output):
                if not torch.is_tensor(output):
                    raise TypeError("Stem Grad-CAM hook requires a tensor output.")
                output.retain_grad()
                captured["stem"] = output

            handle = model.stem.register_forward_hook(stem_hook)
            model.zero_grad(set_to_none=True)
            try:
                with torch.autocast(
                    device_type=device.type,
                    dtype=torch.bfloat16,
                    enabled=amp_enabled,
                ):
                    features = model.forward_features(
                        images,
                        image_valid_mask=metadata.get("image_mask"),
                        bbox_token_prior=metadata.get("bbox"),
                    )
                    if torch.is_tensor(metadata.get("bbox")):
                        features["bbox"] = metadata["bbox"]
                    logits = common.classification_logits_from_features(
                        model, features
                    )
                logits[:, FOCUS_CLASS].float().sum().backward()
            finally:
                handle.remove()
            stem = captured.get("stem")
            if stem is None or stem.grad is None:
                raise RuntimeError("Stem Grad-CAM gradient was not retained.")
            heat = common._batched_gradcam(
                stem,
                stem.grad,
                size=(int(images.size(-2)), int(images.size(-1))),
            ).cpu()
            foreground = common._bbox_foreground_mass(heat, bboxes)
            probabilities = logits.detach().float().softmax(dim=1).cpu()
            for position, local_value in enumerate(sample_indices.tolist()):
                local_index = int(local_value)
                key = (condition, local_index)
                standard = standard_rows[key]
                prediction = int(probabilities[position].argmax().item())
                records[key] = {
                    "rgb": common._rgb_from_tensor(
                        images_cpu[position], mean=mean, std=std
                    ),
                    "bbox": bboxes[position].detach().float().cpu().numpy(),
                    "stem_gradcam": heat[position].numpy(),
                    "stem_foreground_mass": float(foreground[position]),
                    "prediction": prediction,
                    "standard_prediction": int(standard["prediction"]),
                    "standard_prediction_match": prediction
                    == int(standard["prediction"]),
                }
            del features, logits, images
            gc.collect()
            if device.type == "cuda":
                torch.cuda.empty_cache()
    expected = {
        (str(request["condition"]), int(request["local_index"]))
        for request in requests
    }
    if set(records) != expected:
        raise ValueError(f"{role} Grad-CAM records differ from requests.")
    finite = all(
        np.isfinite(np.asarray(record["stem_gradcam"])).all()
        for record in records.values()
    )
    return records, {
        "rows": len(records),
        "loaders": loaders,
        "finite": bool(finite),
        "standard_prediction_matches": sum(
            int(bool(record["standard_prediction_match"]))
            for record in records.values()
        ),
    }


def _caption_gradcam(
    record: Mapping[str, object], *, title: str
) -> Image.Image:
    overlay = common._heat_overlay(
        np.asarray(record["rgb"]), np.asarray(record["stem_gradcam"])
    ).convert("RGB")
    draw = ImageDraw.Draw(overlay)
    center_x, center_y, width, height = [
        float(value) for value in np.asarray(record["bbox"]).reshape(-1)[:4]
    ]
    image_width, image_height = overlay.size
    rectangle = (
        int(round((center_x - width / 2.0) * image_width)),
        int(round((center_y - height / 2.0) * image_height)),
        int(round((center_x + width / 2.0) * image_width)),
        int(round((center_y + height / 2.0) * image_height)),
    )
    draw.rectangle(rectangle, outline=(20, 225, 230), width=4)
    header_height = 58
    canvas = Image.new("RGB", (image_width, image_height + header_height), (248, 248, 248))
    canvas.paste(overlay, (0, header_height))
    header = ImageDraw.Draw(canvas)
    header.text((6, 6), title[:105], fill=(10, 10, 10))
    header.text(
        (6, 27),
        f"stem fg={float(record['stem_foreground_mass']):.3f}",
        fill=(25, 25, 25),
    )
    header.text((6, 44), "cyan=bbox orange=class1 Grad-CAM", fill=(45, 45, 45))
    return canvas


def _pair_visual_row(row: Mapping[str, object]) -> Dict[str, object]:
    visual_row = dict(row)
    # Jaccard compared preflight off/on pruning; no such pair exists post-train.
    visual_row["second_prune_jaccard"] = math.nan
    return visual_row


def _render_xai_pages(
    *,
    base_dataset,
    transform,
    requests: Sequence[Mapping[str, object]],
    trace_rows: Sequence[Mapping[str, object]],
    control_maps: Mapping[Tuple[str, int], Mapping[str, object]],
    candidate_maps: Mapping[Tuple[str, int], Mapping[str, object]],
    output_dir: Path,
    mean: Sequence[float],
    std: Sequence[float],
) -> Dict[str, object]:
    trace_map = {
        (str(row["condition"]), int(row["local_holdout_index"])): row
        for row in trace_rows
    }
    condition_specs = {
        name: (brightness, contrast)
        for name, brightness, contrast in common.CONDITIONS
    }
    row_panels: list[Tuple[Mapping[str, object], Sequence[Image.Image]]] = []
    for request in requests:
        condition = str(request["condition"])
        local_index = int(request["local_index"])
        key = (condition, local_index)
        trace = trace_map[key]
        categories = ",".join(str(value) for value in request.get("categories", []))
        prefix = (
            f"{condition} idx={request['sample_index']} y={request['target']} "
            f"c={trace['control_prediction']} f={trace['candidate_prediction']}"
        )
        control_panel = _caption_gradcam(
            control_maps[key], title=f"control {prefix} {categories}"
        )
        candidate_panel = _caption_gradcam(
            candidate_maps[key], title=f"fusion {prefix} {categories}"
        )
        brightness, contrast = condition_specs[condition]
        dataset = common._SelectedConditionDataset(
            base_dataset,
            [local_index],
            corruption=common._condition_corruption(
                condition, brightness, contrast
            ),
            transform=transform,
        )
        image_tensor, _label, metadata = dataset[0]
        fusion_panel = fusion._draw_fusion_overlay(
            image_tensor,
            metadata["bbox"],
            _pair_visual_row(trace),
            mean=mean,
            std=std,
            title=f"context {prefix} {categories}",
        )
        row_panels.append(
            (request, (control_panel, candidate_panel, fusion_panel))
        )
    if not row_panels:
        raise ValueError("No pair XAI panels were rendered.")
    rows_per_page = 4
    page_paths: list[str] = []
    page_hashes: Dict[str, str] = {}
    for page_number, start in enumerate(
        range(0, len(row_panels), rows_per_page), start=1
    ):
        page_rows = row_panels[start : start + rows_per_page]
        width, height = page_rows[0][1][0].size
        page = Image.new(
            "RGB", (3 * width, len(page_rows) * height), (235, 235, 235)
        )
        for row_index, (_request, panels) in enumerate(page_rows):
            for column, panel in enumerate(panels):
                page.paste(panel, (column * width, row_index * height))
        path = output_dir / f"fusion_pair_xai_contact_sheet_{page_number:03d}.png"
        page.save(path, format="PNG", optimize=True)
        resolved = str(path.resolve())
        page_paths.append(resolved)
        page_hashes[resolved] = common._sha256(path)
    represented = {
        str(category)
        for request in requests
        for category in request.get("categories", [])
    }
    return {
        "rows": len(row_panels),
        "pages": page_paths,
        "page_sha256": page_hashes,
        "page_count": len(page_paths),
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
    control_mass = np.asarray(
        [float(control_maps[key]["stem_foreground_mass"]) for key in sorted(keys)],
        dtype=np.float64,
    )
    candidate_mass = np.asarray(
        [float(candidate_maps[key]["stem_foreground_mass"]) for key in sorted(keys)],
        dtype=np.float64,
    )
    return {
        "request_rows": len(keys),
        "event_rows": len(event_keys),
        "all_event_rows_covered": event_keys.issubset(keys),
        "all_representatives_covered": required_representatives.issubset(
            represented
        ),
        "required_representatives": sorted(required_representatives),
        "control_stem_foreground_mass": float(control_mass.mean()),
        "candidate_stem_foreground_mass": float(candidate_mass.mean()),
        "stem_foreground_mass_delta": float(
            candidate_mass.mean() - control_mass.mean()
        ),
        "maps_finite": bool(
            np.isfinite(control_mass).all() and np.isfinite(candidate_mass).all()
        ),
        "control_standard_prediction_matches": sum(
            int(bool(control_maps[key]["standard_prediction_match"]))
            for key in keys
        ),
        "candidate_standard_prediction_matches": sum(
            int(bool(candidate_maps[key]["standard_prediction_match"]))
            for key in keys
        ),
        "render": dict(render),
    }


def _gate_checks(
    *,
    provenance: Mapping[str, object],
    comparisons: Mapping[str, object],
    cohort: Mapping[str, object],
    selectivity: Mapping[str, object],
    context: Mapping[str, object],
    perturbation: Mapping[str, object],
    xai: Mapping[str, object],
    replay: Mapping[str, object],
    correction_replay: Optional[Mapping[str, object]],
) -> Dict[str, bool]:
    clean = comparisons["clean"]
    clean_delta = clean["delta"]
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
    selectivity_conditions = selectivity["conditions"]
    clean_selectivity = float(
        selectivity_conditions["clean"]["candidate"]["auroc"]
    )
    shifted_names = [name for name, _, _ in common.CONDITIONS[1:]]
    context_conditions = context["conditions"]
    clean_context = context_conditions["clean"]
    total_xai_rows = int(xai["request_rows"])
    checks: Dict[str, bool] = {
        "provenance_and_training": bool(provenance["all_checks_pass"]),
        "correction_replay_exact": correction_replay is None
        or bool(correction_replay["all_checks_pass"]),
        "prediction_replay_exact": bool(replay["metrics_exact"]),
        "prediction_rows_exact": int(replay["rows"])
        == 2 * len(common.CONDITIONS) * EXPECTED_HOLDOUT_ROWS,
        "clean_macro_f1_delta_gte_0": float(clean_delta["macro_f1"]) >= 0.0,
        "clean_class1_f1_delta_gte_0p005": float(clean_delta["class1_f1"])
        >= 0.005,
        "clean_class1_precision_delta_gte_0p015": float(
            clean_delta["class1_precision"]
        )
        >= 0.015,
        "clean_class1_tp_no_more_than_2_lower": candidate_tp >= control_tp - 2,
        "clean_restricted_fp_reduction_gte_4": restricted_reduction >= 4,
        "clean_restricted_fp_reduction_gte_10pct": restricted_control > 0
        and restricted_reduction / restricted_control >= 0.10,
        "clean_corrections_gt_harms": int(transitions["candidate_correction"])
        > int(transitions["candidate_harm"]),
        "clean_nonfocus_f1_drop_lte_0p015": float(
            clean["maximum_nonfocus_f1_drop"]
        )
        <= 0.015,
        "mean_condition_class1_precision_delta_gte_0p010": float(
            np.mean(precision_deltas)
        )
        >= 0.010,
        "condition_class1_f1_no_delta_below_minus_0p010": min(f1_deltas)
        >= -0.010,
        "tiny_edge_macro_f1_not_lower": float(cohort["delta"]["macro_f1"])
        >= 0.0,
        "tiny_edge_class1_f1_not_lower": float(
            cohort["delta"]["class1_f1"]
        )
        >= 0.0,
        "context_selectivity_clean_candidate_gte_0p65": clean_selectivity
        >= 0.65,
        "context_selectivity_shifted_candidate_gte_0p60": all(
            float(selectivity_conditions[name]["candidate"]["auroc"]) >= 0.60
            for name in shifted_names
        ),
        "context_selectivity_shift_drop_lte_0p10": all(
            clean_selectivity
            - float(selectivity_conditions[name]["candidate"]["auroc"])
            <= 0.10
            for name in shifted_names
        ),
        "context_rows_exact": int(context["rows"])
        == len(common.CONDITIONS) * EXPECTED_HOLDOUT_ROWS,
        "context_clean_partitions_valid": float(
            clean_context["partitions_valid_fraction"]
        )
        == 1.0,
        "context_clean_stage1_mass_gte_0p95": float(
            clean_context["stage1_positive_finite_mass_fraction"]
        )
        >= 0.95,
        "context_clean_stage2_mass_gte_0p95": float(
            clean_context["stage2_positive_finite_mass_fraction"]
        )
        >= 0.95,
        "context_clean_mass_replay_lte_1e_6": float(
            clean_context["maximum_mass_replay_error"]
        )
        <= 1e-6,
        "context_clean_tiny_edge_nonzero_gte_0p20": float(
            clean_context["tiny_edge_nonzero_object_mass_fraction"]
        )
        >= 0.20,
        "context_clean_weighted_object_gain_gte_0p010": float(
            clean_context["weighted_object_gain_mean"]
        )
        >= 0.010,
        "context_clean_outside_mass_gte_0p20": float(
            clean_context["outside_context_fraction_mean"]
        )
        >= 0.20,
        "context_all_rows_finite": all(
            int(context_conditions[name]["finite_rows"])
            == EXPECTED_HOLDOUT_ROWS
            for name, _, _ in common.CONDITIONS
        ),
        "context_standard_trace_parity": all(
            float(context_conditions[name]["standard_trace_maximum_error"])
            <= 1e-6
            and int(context_conditions[name]["standard_trace_prediction_matches"])
            == EXPECTED_HOLDOUT_ROWS
            for name, _, _ in common.CONDITIONS
        ),
        "context_shift_cosine_gte_0p65": all(
            context_conditions[name]["clean_context_cosine_mean"] is not None
            and float(context_conditions[name]["clean_context_cosine_mean"])
            >= 0.65
            for name in shifted_names
        ),
        "object_perturbation_more_causal_than_background": bool(
            perturbation["roles"]["candidate"]["object_more_causal"]
        ),
        "xai_stem_foreground_delta_gte_minus_0p05": float(
            xai["stem_foreground_mass_delta"]
        )
        >= -0.05,
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
    }
    return checks


def _finalize_visual_review(args: argparse.Namespace) -> Dict[str, object]:
    output_dir = Path(args.output_dir).resolve()
    summary_path = output_dir / "summary.json"
    if not summary_path.is_file():
        raise FileNotFoundError(f"Fusion pair summary does not exist: {summary_path}")
    if args.visual_review_result is None or not str(args.visual_review_note).strip():
        raise ValueError("Visual finalization requires a result and nonempty note.")
    expected = str(args.expected_summary_sha256).strip().lower()
    if len(expected) != 64:
        raise ValueError("Visual finalization requires --expected-summary-sha256.")
    observed = common._sha256(summary_path)
    if observed != expected:
        raise ValueError(
            f"Fusion pair summary changed before review: {observed} != {expected}"
        )
    review_path = output_dir / "visual_review.json"
    if review_path.exists():
        raise FileExistsError(f"Fusion pair visual review already exists: {review_path}")
    summary = common._load_json(summary_path)
    render = summary.get("xai", {}).get("render", {})
    pages = render.get("pages", [])
    expected_hashes = render.get("page_sha256", {})
    if not pages or not isinstance(expected_hashes, Mapping):
        raise ValueError("Fusion pair review lacks XAI pages or hashes.")
    observed_hashes = {}
    for value in pages:
        path = Path(str(value))
        if not path.is_file():
            raise FileNotFoundError(f"Fusion pair XAI page is missing: {path}")
        page_hash = common._sha256(path)
        if page_hash != str(expected_hashes.get(str(path.resolve()), "")):
            raise ValueError(f"Fusion pair XAI page hash differs: {path}")
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
    fusion._write_manifest(output_dir)
    return summary


def run_audit(args: argparse.Namespace) -> Dict[str, object]:
    if int(args.seed) != 42 or int(args.batch_size) != 32:
        raise ValueError("Fusion pair audit is locked to seed=42 and batch_size=32.")
    if int(args.xai_batch_size) <= 0 or int(args.xai_batch_size) > 2:
        raise ValueError("Fusion pair XAI batch size must be in [1,2].")
    if not torch.cuda.is_available():
        raise RuntimeError("Fusion pair audit requires CUDA.")
    current_head = _git_value("rev-parse", "HEAD")
    failed_attempt = None
    if args.failed_attempt_manifest is not None:
        failed_attempt = _verify_failed_attempt(
            Path(args.failed_attempt_manifest), current_head=current_head
        )
        if not bool(failed_attempt["all_checks_pass"]):
            raise ValueError(
                "Failed-attempt evidence did not pass correction replay preflight: "
                f"{failed_attempt['failed_checks']}"
            )
    output_dir = _prepare_output(args.output_dir)
    declaration = Path(args.declaration).resolve()
    fold_summary_path = Path(args.fold_summary).resolve()
    protocol_path = Path(args.protocol).resolve()
    preflight_path = Path(args.preflight_summary).resolve()
    pair_manifest_path = Path(args.pair_manifest).resolve()
    if common._sha256(declaration) != LOCKED_DECLARATION_SHA256:
        raise ValueError("Fusion pair declaration hash differs from the protocol.")
    if common._sha256(fold_summary_path) != LOCKED_FOLD_SUMMARY_SHA256:
        raise ValueError("Fusion pair fold summary hash differs from the protocol.")
    if common._sha256(protocol_path) != LOCKED_PROTOCOL_SHA256:
        raise ValueError("Fusion pair protocol hash differs.")
    rows = common._read_clean_train_rows(declaration)
    holdout_rows = [row for row in rows if row.fold == FOLD]
    fit_rows = [row for row in rows if row.fold != FOLD]
    if len(holdout_rows) != EXPECTED_HOLDOUT_ROWS or len(fit_rows) != EXPECTED_FIT_ROWS:
        raise ValueError("Fusion pair fit/holdout counts differ.")
    if common._ordered_index_sha256(
        [row.sample_index for row in fit_rows]
    ) != EXPECTED_FIT_INDEX_SHA256:
        raise ValueError("Fusion pair fit ordered-index hash differs.")
    if common._ordered_index_sha256(
        [row.sample_index for row in holdout_rows]
    ) != EXPECTED_HOLDOUT_INDEX_SHA256:
        raise ValueError("Fusion pair holdout ordered-index hash differs.")

    set_seed(int(args.seed), deterministic=True)
    torch.set_float32_matmul_precision("highest")
    torch.backends.cuda.matmul.allow_tf32 = False
    torch.backends.cudnn.allow_tf32 = False
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False
    device = torch.device("cuda")
    control_checkpoint_path = Path(args.control_checkpoint).resolve()
    candidate_checkpoint_path = Path(args.candidate_checkpoint).resolve()
    control, control_checkpoint, control_classes = load_model(
        control_checkpoint_path, device
    )
    candidate, candidate_checkpoint, candidate_classes = load_model(
        candidate_checkpoint_path, device
    )
    if control_classes != candidate_classes or len(control_classes) != 5:
        raise ValueError("Fusion pair checkpoint class orders differ.")
    control_config = control_checkpoint.get("model_config", {})
    candidate_config = candidate_checkpoint.get("model_config", {})
    if bool(control_config.get("inattentive_token_fusion", False)):
        raise ValueError("Fusion control checkpoint unexpectedly enables fusion.")
    if not bool(candidate_config.get("inattentive_token_fusion", False)):
        raise ValueError("Fusion candidate checkpoint does not enable fusion.")
    if common._eval_semantics(control_checkpoint) != common._eval_semantics(
        candidate_checkpoint
    ):
        raise ValueError("Fusion pair evaluation semantics differ.")

    base_dataset, transform, dataset_summary = common._build_holdout_dataset(
        control_checkpoint,
        fold_data=Path(args.fold_data),
        holdout_rows=holdout_rows,
    )
    preflight = common._load_json(preflight_path)
    fold_summary = common._load_json(fold_summary_path)
    provenance = _run_provenance(
        control_run=Path(args.control_run_dir).resolve(),
        candidate_run=Path(args.candidate_run_dir).resolve(),
        pair_manifest_path=pair_manifest_path,
        preflight_path=preflight_path,
        preflight=preflight,
        fold_summary_path=fold_summary_path,
        fold_summary=fold_summary,
        protocol_path=protocol_path,
    )
    cohort = _load_locked_cohort(preflight, output_dir=output_dir)
    predictions, prediction_loaders = _predict_conditions(
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
    replay = _prediction_replay(prediction_path, comparisons)
    replay.pop("reconstructed")
    selectivity = common._selectivity(rows=rows, predictions=predictions)
    cohort_comparison = _cohort_comparison(predictions, cohort)
    context, context_rows = _trace_context_conditions(
        candidate=candidate,
        predictions=predictions,
        base_dataset=base_dataset,
        transform=transform,
        holdout_rows=holdout_rows,
        cohort=cohort,
        output_dir=output_dir,
        device=device,
        batch_size=int(args.batch_size),
        num_workers=int(args.num_workers),
        seed=int(args.seed),
    )
    events = common._build_event_manifest(predictions)
    event_path = output_dir / "event_manifest.csv"
    common._write_event_manifest(event_path, events)
    representatives = _representative_requests(
        dataset=base_dataset,
        holdout_rows=holdout_rows,
        positive_sample_indices=selectivity["positive_sample_indices"],
        cohort=cohort,
    )
    requests = common._merge_xai_requests(events, representatives)
    perturbation = common._perturbation_audit(
        control=control,
        candidate=candidate,
        base_dataset=base_dataset,
        transform=transform,
        holdout_rows=holdout_rows,
        selectivity=selectivity,
        device=device,
        batch_size=int(args.batch_size),
        num_workers=int(args.num_workers),
        seed=int(args.seed),
    )
    semantics = dataset_summary["semantics"]
    mean = tuple(float(value) for value in semantics["input_mean"])
    std = tuple(float(value) for value in semantics["input_std"])
    standard_maps = {
        role: {
            (condition, int(row["local_index"])): row
            for condition, _, _ in common.CONDITIONS
            for row in predictions[role][condition]
        }
        for role in ("control", "candidate")
    }
    control_maps, control_xai = _collect_stem_gradcam(
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
    candidate_maps, candidate_xai = _collect_stem_gradcam(
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
        base_dataset=base_dataset,
        transform=transform,
        requests=requests,
        trace_rows=context_rows,
        control_maps=control_maps,
        candidate_maps=candidate_maps,
        output_dir=output_dir,
        mean=mean,
        std=std,
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
    correction_replay = None
    if failed_attempt is not None:
        artifact_replay = _compare_correction_artifacts(
            failed_attempt, output_dir=output_dir
        )
        correction_replay = {
            **failed_attempt,
            "artifact_replay": artifact_replay,
            "all_checks_pass": bool(failed_attempt["all_checks_pass"])
            and bool(artifact_replay["all_checks_pass"]),
            "failed_checks": [
                *failed_attempt["failed_checks"],
                *artifact_replay["failed_checks"],
            ],
        }
    checks = _gate_checks(
        provenance=provenance,
        comparisons=comparisons,
        cohort=cohort_comparison,
        selectivity=selectivity,
        context=context,
        perturbation=perturbation,
        xai=xai,
        replay=replay,
        correction_replay=correction_replay,
    )
    summary: Dict[str, object] = {
        "method": METHOD,
        "created_at_utc": datetime.now(timezone.utc).isoformat(),
        "inputs": {
            "control_checkpoint": str(control_checkpoint_path),
            "control_checkpoint_sha256": common._sha256(
                control_checkpoint_path
            ),
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
            "failed_attempt_manifest": (
                str(Path(args.failed_attempt_manifest).resolve())
                if args.failed_attempt_manifest is not None
                else None
            ),
        },
        "dataset": dataset_summary,
        "provenance": provenance,
        "prediction_loaders": prediction_loaders,
        "comparisons": comparisons,
        "prediction_replay": replay,
        "selectivity": selectivity,
        "tiny_edge_cohort": cohort_comparison,
        "context": context,
        "perturbation": perturbation,
        "events": {
            "rows": len(events),
            "path": str(event_path.resolve()),
            "sha256": common._sha256(event_path),
        },
        "xai": xai,
        "correction_replay": correction_replay,
        "predictions": {
            "path": str(prediction_path.resolve()),
            "sha256": common._sha256(prediction_path),
            "rows": 2 * len(common.CONDITIONS) * EXPECTED_HOLDOUT_ROWS,
        },
        "validation_predictions_used": False,
        "test_data_used": False,
        "current_best_command_updated": False,
        "gate": {
            "automated_checks": checks,
            "failed_automated_checks": sorted(
                name for name, passed in checks.items() if not passed
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
                "python -m trkh.tools.audit_inattentive_token_fusion_pair "
                f"--output-dir \"{output_dir}\" --finalize-visual-review "
                f"--expected-summary-sha256 {summary_hash} "
                "--visual-review-result pass|fail --visual-review-note \"...\""
            ),
        },
    )
    fusion._write_manifest(output_dir)
    return summary


def main(argv: Optional[Sequence[str]] = None) -> None:
    args = parse_args(argv)
    summary = (
        _finalize_visual_review(args)
        if bool(args.finalize_visual_review)
        else run_audit(args)
    )
    print(json.dumps(summary["gate"], indent=2, sort_keys=True), flush=True)
    if bool(args.finalize_visual_review):
        if not bool(summary["gate"]["pair_pass"]):
            raise SystemExit(2)
    elif not bool(summary["gate"]["automated_pass"]):
        raise SystemExit(2)


if __name__ == "__main__":
    main()
