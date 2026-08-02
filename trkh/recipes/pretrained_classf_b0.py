from __future__ import annotations

import argparse
import csv
import hashlib
import importlib.metadata
import json
import math
import os
import re
import subprocess
import sys
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Dict, Mapping, Optional, Sequence

import torch

from trkh.core.config import load_data_spec
from trkh.core.utils import build_optimizer_param_groups, load_checkpoint
from trkh.data.dataset import ClassificationFolderDataset
from trkh.models.model import create_model
from trkh.tools.validate_canonical_classf import (
    CANONICAL_FILE_SHA256,
    CANONICAL_IMAGE_TREE_SHA256,
    validate_canonical_classf,
)
from trkh.training.train import build_configs, parse_args as parse_train_args


PROTOCOL_ID = "TRKH_PRETRAINED_CLASSF_B0_20260731"
B1_NATURAL_PROTOCOL_ID = "TRKH_PRETRAINED_CLASSF_B1_NATURAL_20260731"
B1_MARGIN0_PROTOCOL_ID = "TRKH_PRETRAINED_CLASSF_B1_MARGIN0_20260731"
B2_TEMPERED_P05_PROTOCOL_ID = "TRKH_PRETRAINED_CLASSF_B2_TEMPERED_P05_20260731"
B9_B2_REFERENCE_COMPLETION_PROTOCOL_ID = (
    "TRKH_PRETRAINED_CLASSF_B9_B2_REFERENCE_COMPLETION_20260801"
)
B10_GROUP_TEMPERED_P05_PROTOCOL_ID = (
    "TRKH_PRETRAINED_CLASSF_B10_GROUP_TEMPERED_P05_20260802"
)
HYBRID_V2_GENERIC_PROTOCOL_ID = (
    "TRKH_PRETRAINED_CLASSF_HYBRID_V2_GENERIC_20260802"
)
HYBRID_V2_LOCAL_SURFACE_PROTOCOL_ID = (
    "TRKH_PRETRAINED_CLASSF_HYBRID_V2_LOCAL_SURFACE_20260802"
)
HYBRID_V2_LOCAL_SURFACE_RANDOMINIT_PROTOCOL_ID = (
    "TRKH_PRETRAINED_CLASSF_HYBRID_V2_LOCAL_SURFACE_RANDOMINIT_20260802"
)
PRMR_R1_CONTROL_PROTOCOL_ID = "TRKH_PRETRAINED_CLASSF_PRMR_R1_CONTROL_20260801"
PRMR_R1_PROTOCOL_ID = "TRKH_PRETRAINED_CLASSF_PRMR_R1_20260801"
DINO_MODEL_NAME = "vit_small_patch16_dinov3.lvd1689m"
DINO_SHA256 = "2a1ec16ae28ffa07bc0ead0241ee7df9fc26451fe6f9f839b7b3afa0a906b040"
DINO_SOURCE_URL = "https://huggingface.co/timm/vit_small_patch16_dinov3.lvd1689m"
DINO_SOURCE_REVISION = "3bf4720a82ec2066db88137180ff1f83a675cef0"
DINO_SOURCE_LICENSE = "dinov3-license"
EXPECTED_PARAMETER_COUNT = 21_588_869
HYBRID_V2_GENERIC_EXPECTED_PARAMETER_COUNT = 21_618_859
HYBRID_V2_LOCAL_SURFACE_EXPECTED_PARAMETER_COUNT = 21_618_694
EXPECTED_CLASS_NAMES = (
    "Xoai_Song_Chua_KhoDap",
    "Xoai_Song_ChuaNhe_CoNguyCo",
    "Xoai_Chin_NgotThanh_DeDap",
    "Xoai_ChinGia_NgotGat_KhongVanChuyen",
    "Xoai_Hu_KhongAnDuoc",
)
EXPECTED_DEVELOPMENT_COUNTS = {
    "train": [1987, 497, 1326, 2080, 2388],
    "val": [558, 158, 380, 494, 889],
}


@dataclass(frozen=True)
class Stage:
    epochs: int
    scheduler_total_epochs: int
    warmup_epochs: int
    patience: int
    max_train_batches: int
    max_val_batches: int


@dataclass(frozen=True)
class Experiment:
    key: str
    protocol_id: str
    run_prefix: str
    balanced_epoch_sampling: bool
    tempered_class_sampling_power: float | None
    ldam_max_margin: float
    single_semantic_delta_from_b0: str | None
    tempered_leakage_group_sampling: bool = False
    research_role: str = "unclassified"
    promotion_eligible: bool = False
    reference_experiment: str | None = None
    known_probe_class1_f1: float | None = None
    required_probe_class1_f1: float | None = None
    full_train_authorized: bool = False
    model_type: str = "timm_classifier"
    dinov3_surface_hybrid_mode: str | None = None
    expected_parameter_count: int = EXPECTED_PARAMETER_COUNT
    external_initialization_used: bool = True
    extra_train_args: tuple[str, ...] = ()


EXPERIMENTS: Mapping[str, Experiment] = {
    "b0": Experiment(
        key="b0",
        protocol_id=PROTOCOL_ID,
        run_prefix="pretrained_dinov3_classf_direct",
        balanced_epoch_sampling=True,
        tempered_class_sampling_power=None,
        ldam_max_margin=0.3,
        single_semantic_delta_from_b0=None,
        research_role="matched_pretrained_backbone_control",
        promotion_eligible=False,
        full_train_authorized=True,
    ),
    "b1-natural": Experiment(
        key="b1-natural",
        protocol_id=B1_NATURAL_PROTOCOL_ID,
        run_prefix="pretrained_dinov3_classf_natural",
        balanced_epoch_sampling=False,
        tempered_class_sampling_power=None,
        ldam_max_margin=0.3,
        single_semantic_delta_from_b0="disable_balanced_epoch_sampling",
        research_role="closed_probe_ablation",
        promotion_eligible=False,
    ),
    "b1-margin0": Experiment(
        key="b1-margin0",
        protocol_id=B1_MARGIN0_PROTOCOL_ID,
        run_prefix="pretrained_dinov3_classf_margin0",
        balanced_epoch_sampling=True,
        tempered_class_sampling_power=None,
        ldam_max_margin=0.0,
        single_semantic_delta_from_b0="ldam_max_margin:0.3->0.0",
        research_role="closed_probe_ablation",
        promotion_eligible=False,
    ),
    "b2-tempered-p05": Experiment(
        key="b2-tempered-p05",
        protocol_id=B2_TEMPERED_P05_PROTOCOL_ID,
        run_prefix="pretrained_dinov3_classf_tempered_p05",
        balanced_epoch_sampling=False,
        tempered_class_sampling_power=0.5,
        ldam_max_margin=0.3,
        single_semantic_delta_from_b0="class_sampling_prior:uniform->n_c**0.5",
        research_role="closed_probe_hypothesis",
        promotion_eligible=False,
        known_probe_class1_f1=0.653409,
        required_probe_class1_f1=0.66,
    ),
    "b9-b2-reference-completion": Experiment(
        key="b9-b2-reference-completion",
        protocol_id=B9_B2_REFERENCE_COMPLETION_PROTOCOL_ID,
        run_prefix="pretrained_dinov3_classf_b9_b2_reference_completion",
        balanced_epoch_sampling=False,
        tempered_class_sampling_power=0.5,
        ldam_max_margin=0.3,
        single_semantic_delta_from_b0="class_sampling_prior:uniform->n_c**0.5",
        research_role="exploratory_canonical_reference_completion",
        promotion_eligible=False,
        reference_experiment="b2-tempered-p05",
        known_probe_class1_f1=0.653409,
        required_probe_class1_f1=0.66,
        full_train_authorized=True,
    ),
    "b10-group-tempered-p05": Experiment(
        key="b10-group-tempered-p05",
        protocol_id=B10_GROUP_TEMPERED_P05_PROTOCOL_ID,
        run_prefix="pretrained_dinov3_classf_b10_group_tempered_p05",
        balanced_epoch_sampling=False,
        tempered_class_sampling_power=0.5,
        ldam_max_margin=0.3,
        single_semantic_delta_from_b0=(
            "class_sampling_prior:uniform->n_c**0.5;"
            "within_class_sampling:image_uniform->leakage_group_uniform"
        ),
        tempered_leakage_group_sampling=True,
        research_role="train_group_decorrelation_probe",
        promotion_eligible=False,
        reference_experiment="b2-tempered-p05",
        required_probe_class1_f1=0.66,
        full_train_authorized=False,
    ),
    "hybrid-v2-generic": Experiment(
        key="hybrid-v2-generic",
        protocol_id=HYBRID_V2_GENERIC_PROTOCOL_ID,
        run_prefix="pretrained_dinov3_classf_hybrid_v2_generic",
        balanced_epoch_sampling=False,
        tempered_class_sampling_power=0.5,
        ldam_max_margin=0.3,
        single_semantic_delta_from_b0=(
            "class_sampling_prior:uniform->n_c**0.5;"
            "within_class_sampling:image_uniform->leakage_group_uniform;"
            "model:direct_dinov3->pre_final_capacity_matched_generic_adapter"
        ),
        tempered_leakage_group_sampling=True,
        research_role="hybrid_v2_capacity_matched_probe_control",
        promotion_eligible=False,
        reference_experiment="b10-group-tempered-p05",
        full_train_authorized=False,
        model_type="dinov3_surface_patch_hybrid_v2",
        dinov3_surface_hybrid_mode="generic_token_adapter",
        expected_parameter_count=HYBRID_V2_GENERIC_EXPECTED_PARAMETER_COUNT,
    ),
    "hybrid-v2-local-surface": Experiment(
        key="hybrid-v2-local-surface",
        protocol_id=HYBRID_V2_LOCAL_SURFACE_PROTOCOL_ID,
        run_prefix="pretrained_dinov3_classf_hybrid_v2_local_surface",
        balanced_epoch_sampling=False,
        tempered_class_sampling_power=0.5,
        ldam_max_margin=0.3,
        single_semantic_delta_from_b0=(
            "class_sampling_prior:uniform->n_c**0.5;"
            "within_class_sampling:image_uniform->leakage_group_uniform;"
            "model:direct_dinov3->pre_final_exact_patch_local_surface_residual"
        ),
        tempered_leakage_group_sampling=True,
        research_role="hybrid_v2_local_surface_matched_probe",
        promotion_eligible=False,
        reference_experiment="b10-group-tempered-p05",
        full_train_authorized=False,
        model_type="dinov3_surface_patch_hybrid_v2",
        dinov3_surface_hybrid_mode="local_surface",
        expected_parameter_count=HYBRID_V2_LOCAL_SURFACE_EXPECTED_PARAMETER_COUNT,
    ),
    "hybrid-v2-local-surface-randominit": Experiment(
        key="hybrid-v2-local-surface-randominit",
        protocol_id=HYBRID_V2_LOCAL_SURFACE_RANDOMINIT_PROTOCOL_ID,
        run_prefix="pretrained_dinov3_classf_hybrid_v2_local_surface_randominit",
        balanced_epoch_sampling=False,
        tempered_class_sampling_power=0.5,
        ldam_max_margin=0.3,
        single_semantic_delta_from_b0=(
            "class_sampling_prior:uniform->n_c**0.5;"
            "within_class_sampling:image_uniform->leakage_group_uniform;"
            "model:direct_dinov3->pre_final_exact_patch_local_surface_residual;"
            "initialization:locked_dinov3_checkpoint->random"
        ),
        tempered_leakage_group_sampling=True,
        research_role=(
            "matched_initialization_ablation_under_fixed_finetune_protocol"
        ),
        promotion_eligible=False,
        reference_experiment="hybrid-v2-local-surface",
        full_train_authorized=False,
        model_type="dinov3_surface_patch_hybrid_v2",
        dinov3_surface_hybrid_mode="local_surface",
        expected_parameter_count=HYBRID_V2_LOCAL_SURFACE_EXPECTED_PARAMETER_COUNT,
        external_initialization_used=False,
    ),
    "prmr-r1-control": Experiment(
        key="prmr-r1-control",
        protocol_id=PRMR_R1_CONTROL_PROTOCOL_ID,
        run_prefix="pretrained_dinov3_classf_prmr_r1_control",
        balanced_epoch_sampling=False,
        tempered_class_sampling_power=0.5,
        ldam_max_margin=0.3,
        single_semantic_delta_from_b0="class_sampling_prior:uniform->n_c**0.5",
        research_role="matched_prmr_probe_control",
        promotion_eligible=False,
        reference_experiment="b9-b2-reference-completion",
    ),
    "prmr-r1": Experiment(
        key="prmr-r1",
        protocol_id=PRMR_R1_PROTOCOL_ID,
        run_prefix="pretrained_dinov3_classf_prmr_r1",
        balanced_epoch_sampling=False,
        tempered_class_sampling_power=0.5,
        ldam_max_margin=0.3,
        single_semantic_delta_from_b0=(
            "class_sampling_prior:uniform->n_c**0.5;"
            "train_only_pairwise_margin_retention_under_relighting"
        ),
        research_role="project_novel_matched_probe",
        promotion_eligible=False,
        reference_experiment="prmr-r1-control",
        extra_train_args=(
            "--illumination-consistency-loss-weight",
            "0.15",
            "--illumination-consistency-probability",
            "0.50",
            "--illumination-consistency-brightness",
            "0.25",
            "--illumination-consistency-contrast",
            "0.10",
            "--illumination-consistency-gamma",
            "0",
            "--illumination-consistency-mode",
            "pairwise_margin_retention",
            "--illumination-consistency-focus-class",
            "1",
            "--illumination-consistency-negative-classes",
            "0,2,4",
            "--illumination-consistency-margin-retention",
            "0.80",
            "--illumination-consistency-start-epoch",
            "3",
        ),
    ),
}


STAGES: Mapping[str, Stage] = {
    "smoke": Stage(
        epochs=1,
        scheduler_total_epochs=5,
        warmup_epochs=0,
        patience=2,
        max_train_batches=4,
        max_val_batches=2,
    ),
    "probe": Stage(
        epochs=5,
        scheduler_total_epochs=5,
        warmup_epochs=1,
        patience=5,
        max_train_batches=120,
        max_val_batches=0,
    ),
    "full": Stage(
        epochs=30,
        scheduler_total_epochs=30,
        warmup_epochs=2,
        patience=6,
        max_train_batches=0,
        max_val_batches=0,
    ),
}
ACCUMULATION_BY_BATCH = {8: 6, 12: 4, 16: 3, 24: 2}


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with Path(path).open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _absolute_path_without_resolving_symlink(path: Path) -> Path:
    candidate = Path(path).expanduser()
    if not candidate.is_absolute():
        candidate = Path.cwd() / candidate
    return candidate.absolute()


def source_tree_sha256(repo_root: Path) -> str:
    repo_root = Path(repo_root).resolve()
    files = sorted((repo_root / "trkh").rglob("*.py"))
    files.extend(sorted((repo_root / "configs").rglob("*.yaml")))
    digest = hashlib.sha256()
    for path in files:
        digest.update(path.relative_to(repo_root).as_posix().encode("utf-8"))
        digest.update(b"\0")
        # Git may materialize CRLF on Windows and LF in a Kaggle archive.
        # Normalize text line endings so one commit has one portable digest.
        digest.update(path.read_bytes().replace(b"\r\n", b"\n"))
    return digest.hexdigest()


def validate_development_data_yaml(
    data_yaml: Path,
    *,
    canonical_contract: Mapping[str, object],
) -> Dict[str, object]:
    data_yaml = Path(data_yaml).resolve()
    spec = load_data_spec(
        data_yaml,
        class_name_mode="raw",
        expected_num_classes=5,
    )
    if spec.data_format != "classification_folder":
        raise ValueError("B0 development data must use classification_folder.")
    if spec.has_test_split:
        raise ValueError("B0 development YAML must omit the test split.")
    if tuple(spec.class_names) != EXPECTED_CLASS_NAMES:
        raise ValueError(
            "B0 development class order mismatch: "
            f"{list(spec.class_names)!r} != {list(EXPECTED_CLASS_NAMES)!r}."
        )
    canonical_root = Path(str(canonical_contract["root"])).resolve()
    if spec.root.resolve() != canonical_root:
        raise ValueError(
            "B0 development YAML must point to the verified canonical root: "
            f"{spec.root.resolve()} != {canonical_root}."
        )
    split_counts: Dict[str, list[int]] = {}
    split_totals: Dict[str, int] = {}
    for split, expected_counts in EXPECTED_DEVELOPMENT_COUNTS.items():
        dataset = ClassificationFolderDataset.from_data_spec(spec, split=split)
        observed_counts = dataset.class_counts(5)
        if observed_counts != expected_counts:
            raise ValueError(
                f"B0 development {split} counts mismatch: "
                f"{observed_counts} != {expected_counts}."
            )
        split_counts[split] = observed_counts
        split_totals[split] = len(dataset)
    return {
        "data_yaml": str(data_yaml),
        "root": str(spec.root.resolve()),
        "class_names": list(spec.class_names),
        "split_class_counts": split_counts,
        "split_totals": split_totals,
        "has_test_split": False,
    }


def load_canonical_attestation(
    attestation_path: Path,
    *,
    data_yaml: Path,
) -> Dict[str, object]:
    """Reuse a same-session full image hash without reading the 1 GiB tree again."""

    attestation_path = Path(attestation_path).resolve()
    payload = json.loads(attestation_path.read_text(encoding="utf-8"))
    if not isinstance(payload, dict):
        raise ValueError("Canonical attestation must contain a JSON object.")
    expected_data_yaml = Path(data_yaml).resolve()
    observed_data_yaml = Path(str(payload.get("data_yaml", ""))).resolve()
    if observed_data_yaml != expected_data_yaml:
        raise ValueError(
            "Canonical attestation data_yaml mismatch: "
            f"{observed_data_yaml} != {expected_data_yaml}."
        )
    required = {
        "contract": "TRKH_CLASS_F_CANONICAL_V1_20260730",
        "status": "passed",
        "image_tree_sha256": CANONICAL_IMAGE_TREE_SHA256,
        "image_tree_file_count": 12019,
        "file_sha256": CANONICAL_FILE_SHA256,
        "test_model_inference_performed": False,
        "test_metrics_read": False,
    }
    mismatches = {
        key: {"observed": payload.get(key), "expected": expected}
        for key, expected in required.items()
        if payload.get(key) != expected
    }
    if mismatches:
        raise ValueError(f"Canonical attestation mismatch: {mismatches}.")
    observed_file_hashes = {
        name: _sha256(expected_data_yaml.parent / name)
        for name in CANONICAL_FILE_SHA256
    }
    if observed_file_hashes != CANONICAL_FILE_SHA256:
        raise ValueError(
            "Canonical files changed after image-tree attestation: "
            f"{observed_file_hashes}."
        )
    payload["attestation_path"] = str(attestation_path)
    payload["attestation_sha256"] = _sha256(attestation_path)
    payload["reused_without_rehashing_images"] = True
    return payload


def validate_auto_resume_checkpoint(
    checkpoint_path: Path,
    *,
    training_data_yaml: Path,
    expected_train_args: Sequence[str] | None = None,
) -> Dict[str, object]:
    checkpoint_path = Path(checkpoint_path).resolve()
    checkpoint = load_checkpoint(checkpoint_path, map_location="cpu")
    if list(checkpoint.get("class_names", [])) != list(EXPECTED_CLASS_NAMES):
        raise ValueError("Auto-resume checkpoint class order is not canonical class_f.")
    checkpoint_data = checkpoint.get("data_yaml")
    if not checkpoint_data:
        raise ValueError("Auto-resume checkpoint is missing data_yaml provenance.")
    observed_data = Path(str(checkpoint_data)).resolve()
    expected_data = Path(training_data_yaml).resolve()
    if observed_data != expected_data:
        raise ValueError(
            "Auto-resume data YAML mismatch: "
            f"{observed_data} != {expected_data}."
        )
    model_config = checkpoint.get("model_config", {})
    if not isinstance(model_config, Mapping):
        raise ValueError("Auto-resume checkpoint model_config is invalid.")
    expected_parsed = (
        parse_train_args(expected_train_args)
        if expected_train_args is not None
        else None
    )
    expected_model_type = (
        str(expected_parsed.model_type)
        if expected_parsed is not None
        else "timm_classifier"
    )
    required = {
        "model_type": (
            str(model_config.get("model_type", "")),
            expected_model_type,
        ),
        "research_track": (
            str(model_config.get("research_track", "")),
            "pretrained",
        ),
        "timm_model_name": (
            str(model_config.get("timm_model_name", "")),
            DINO_MODEL_NAME,
        ),
        "pretrained_checkpoint_sha256": (
            str(model_config.get("pretrained_checkpoint_sha256", "")).lower(),
            DINO_SHA256
            if expected_parsed is None or bool(expected_parsed.pretrained)
            else "",
        ),
        "pretrained": (
            bool(model_config.get("pretrained", False)),
            True if expected_parsed is None else bool(expected_parsed.pretrained),
        ),
    }
    if expected_model_type == "dinov3_surface_patch_hybrid_v2":
        required.update(
            {
                "dinov3_surface_hybrid_mode": (
                    str(model_config.get("dinov3_surface_hybrid_mode", "")),
                    str(expected_parsed.dinov3_surface_hybrid_mode),
                ),
                "dinov3_surface_initial_gate_scale": (
                    float(
                        model_config.get(
                            "dinov3_surface_initial_gate_scale",
                            float("nan"),
                        )
                    ),
                    float(expected_parsed.dinov3_surface_initial_gate_scale),
                ),
                "dinov3_surface_max_gate_scale": (
                    float(
                        model_config.get(
                            "dinov3_surface_max_gate_scale",
                            float("nan"),
                        )
                    ),
                    float(expected_parsed.dinov3_surface_max_gate_scale),
                ),
            }
        )
    mismatches = {
        key: {"observed": observed, "expected": expected}
        for key, (observed, expected) in required.items()
        if observed != expected
    }
    if mismatches:
        raise ValueError(f"Auto-resume pretrained contract mismatch: {mismatches}.")
    train_config = checkpoint.get("train_config", {})
    if not isinstance(train_config, Mapping):
        raise ValueError("Auto-resume checkpoint train_config is invalid.")
    resume_lineage: Dict[str, object] = {}
    resume_illumination_contract: Dict[str, object] = {}
    if expected_train_args is not None:
        assert expected_parsed is not None
        expected_lineage = {
            "experiment_protocol_id": str(
                expected_parsed.experiment_protocol_id
            ),
            "source_commit": str(expected_parsed.source_commit).lower(),
            "source_tree_sha256": str(
                expected_parsed.source_tree_sha256
            ).lower(),
            "dataset_image_tree_sha256": str(
                expected_parsed.dataset_image_tree_sha256
            ).lower(),
            "recipe_train_contract_sha256": str(
                expected_parsed.recipe_train_contract_sha256
            ).lower(),
        }
        lineage_mismatches = {
            key: {
                "observed": str(train_config.get(key, "")).lower(),
                "expected": expected,
            }
            for key, expected in expected_lineage.items()
            if str(train_config.get(key, "")).lower() != expected.lower()
        }
        if lineage_mismatches:
            raise ValueError(
                "Auto-resume source/protocol/config lineage mismatch: "
                f"{lineage_mismatches}."
            )
        resume_lineage = expected_lineage
        illumination_fields = (
            "illumination_consistency_loss_weight",
            "illumination_consistency_probability",
            "illumination_consistency_brightness",
            "illumination_consistency_contrast",
            "illumination_consistency_gamma",
            "illumination_consistency_temperature",
            "illumination_consistency_mode",
            "illumination_consistency_focus_class",
            "illumination_consistency_negative_classes",
            "illumination_consistency_margin_retention",
            "illumination_consistency_start_epoch",
        )

        def _normalize_illumination_value(field: str, value: object) -> object:
            if field == "illumination_consistency_negative_classes":
                return ",".join(
                    item.strip()
                    for item in str(value).replace(";", ",").split(",")
                    if item.strip()
                )
            if field == "illumination_consistency_mode":
                return str(value).strip().lower()
            return value

        expected_illumination = {
            field: _normalize_illumination_value(
                field,
                getattr(expected_parsed, field),
            )
            for field in illumination_fields
        }
        illumination_mismatches = {
            field: {
                "observed": _normalize_illumination_value(
                    field,
                    train_config.get(field),
                ),
                "expected": expected,
            }
            for field, expected in expected_illumination.items()
            if _normalize_illumination_value(
                field,
                train_config.get(field),
            )
            != expected
        }
        if illumination_mismatches:
            raise ValueError(
                "Auto-resume PRMR/illumination semantic mismatch: "
                f"{illumination_mismatches}."
            )
        resume_illumination_contract = expected_illumination
    return {
        "checkpoint": str(checkpoint_path),
        "data_yaml": str(observed_data),
        "epoch": checkpoint.get("epoch"),
        "best_epoch": checkpoint.get("best_epoch"),
        "model_type": required["model_type"][0],
        "research_track": required["research_track"][0],
        "timm_model_name": required["timm_model_name"][0],
        "pretrained_checkpoint_sha256": required[
            "pretrained_checkpoint_sha256"
        ][0],
        "lineage": resume_lineage,
        "illumination_contract": resume_illumination_contract,
    }


def experiment_spec(experiment: str = "b0") -> Experiment:
    key = str(experiment).strip().lower()
    try:
        return EXPERIMENTS[key]
    except KeyError as error:
        raise ValueError(
            f"Unsupported experiment {experiment!r}; choose from {sorted(EXPERIMENTS)}."
        ) from error


def run_name(stage: str, run_tag: str, *, experiment: str = "b0") -> str:
    spec = experiment_spec(experiment)
    return f"{spec.run_prefix}_{stage}_{run_tag}"


def build_train_args(
    *,
    data_yaml: Path,
    dino_checkpoint: Optional[Path],
    output_dir: Path,
    stage: str,
    run_tag: str,
    batch_size: int = 24,
    num_workers: int = 4,
    eval_num_workers: int = 2,
    seed: int = 42,
    amp_init_scale: Optional[float] = None,
    amp_dtype: str = "auto",
    auto_resume: bool = False,
    source_commit: str = "",
    source_tree_sha256: str = "",
    dataset_image_tree_sha256: str = "",
    experiment: str = "b0",
    tempered_leakage_group_manifest: Optional[Path] = None,
) -> list[str]:
    stage_name = str(stage).strip().lower()
    if stage_name not in STAGES:
        raise ValueError(f"Unsupported stage: {stage!r}.")
    if int(batch_size) not in ACCUMULATION_BY_BATCH:
        raise ValueError(
            f"batch_size must be one of {sorted(ACCUMULATION_BY_BATCH)}."
        )
    grad_accum_steps = ACCUMULATION_BY_BATCH[int(batch_size)]
    if int(batch_size) * grad_accum_steps != 48:
        raise AssertionError("B0 effective batch contract drifted from 48.")
    if not str(run_tag).replace("_", "").replace("-", "").isalnum():
        raise ValueError("run_tag may contain only letters, numbers, '_' and '-'.")
    if amp_init_scale is not None and (
        not math.isfinite(float(amp_init_scale)) or float(amp_init_scale) <= 0.0
    ):
        raise ValueError("amp_init_scale must be finite and > 0.")
    resolved_amp_dtype = str(amp_dtype).strip().lower()
    if resolved_amp_dtype not in {"auto", "bf16", "fp16"}:
        raise ValueError("amp_dtype must be one of auto/bf16/fp16.")
    stage_spec = STAGES[stage_name]
    experiment_config = experiment_spec(experiment)
    if experiment_config.external_initialization_used and dino_checkpoint is None:
        raise ValueError(
            f"{experiment_config.key} requires a locked DINOv3 checkpoint."
        )

    external_initialization_args: list[str] = []
    if experiment_config.external_initialization_used:
        external_initialization_args = [
            "--pretrained",
            "--pretrained-checkpoint-path",
            str(_absolute_path_without_resolving_symlink(Path(dino_checkpoint))),
            "--pretrained-checkpoint-sha256",
            DINO_SHA256,
            "--pretrained-source-url",
            DINO_SOURCE_URL,
            "--pretrained-source-revision",
            DINO_SOURCE_REVISION,
            "--pretrained-source-license",
            DINO_SOURCE_LICENSE,
        ]

    args = [
        "--data",
        str(Path(data_yaml).resolve()),
        "--class-name-mode",
        "raw",
        "--expected-num-classes",
        "5",
        "--output-dir",
        str(Path(output_dir).resolve()),
        "--run-name",
        run_name(stage_name, run_tag, experiment=experiment_config.key),
        "--model-type",
        experiment_config.model_type,
        "--research-track",
        "pretrained",
        "--timm-model-name",
        DINO_MODEL_NAME,
        *external_initialization_args,
        "--image-size",
        "256",
        "--batch-size",
        str(int(batch_size)),
        "--grad-accum-steps",
        str(int(grad_accum_steps)),
        "--epochs",
        str(stage_spec.epochs),
        "--scheduler-total-epochs",
        str(stage_spec.scheduler_total_epochs),
        "--patience",
        str(stage_spec.patience),
        "--learning-rate",
        "1.5e-4",
        "--backbone-lr-scale",
        "0.1",
        "--min-learning-rate",
        "1e-6",
        "--warmup-epochs",
        str(stage_spec.warmup_epochs),
        "--warmup-start-factor",
        "0.1",
        "--weight-decay",
        "0.05",
        "--grad-clip-norm",
        "0.7",
        "--max-nonfinite-grad-steps",
        "4",
        "--model-ema",
        "--model-ema-decay",
        "0.995",
        "--num-workers",
        str(max(0, int(num_workers))),
        "--eval-num-workers",
        str(max(0, int(eval_num_workers))),
        "--train-image-cache-mb",
        "0",
        "--eval-image-cache-mb",
        "0",
        "--seed",
        str(int(seed)),
        "--deterministic",
        "--amp-dtype",
        resolved_amp_dtype,
        "--disable-class-weights",
        "--balanced-epoch-multiplier",
        "1.0",
        "--balanced-epoch-tolerance",
        "0.10",
        "--classification-loss",
        "ldam_focal",
        "--focal-loss-gamma",
        "1.0",
        "--focal-loss-mix",
        "0.1",
        "--label-smoothing",
        "0.02",
        "--ldam-max-margin",
        str(experiment_config.ldam_max_margin),
        "--ldam-scale",
        "18",
        "--best-metric",
        "fair_macro_f1",
        "--fair-f1-gap-target",
        "0.05",
        "--fair-f1-gap-penalty",
        "1.5",
        "--fair-f1-min-weight",
        "0.25",
        "--resize-mode",
        "pad",
        "--train-scale-min",
        "0.90",
        "--train-scale-crop-probability",
        "0.30",
        "--brightness",
        "0.05",
        "--contrast",
        "0.05",
        "--saturation",
        "0.03",
        "--hue",
        "0.01",
        "--local-exposure-probability",
        "0.15",
        "--local-exposure-strength",
        "0.20",
        "--obstacle-probability",
        "0.03",
        "--obstacle-max-area",
        "0.06",
        "--random-erasing-probability",
        "0",
        "--random-affine-degrees",
        "3",
        "--random-affine-translate",
        "0.02",
        "--random-affine-scale-min",
        "0.96",
        "--horizontal-flip-probability",
        "0.5",
        "--vertical-flip-probability",
        "0",
        "--rotate90-probability",
        "0.03",
        "--lighting-probability",
        "0",
        "--disable-class-aware-augmentation",
        "--disable-rare-class-repeat",
        "--batch-mix-probability",
        "0",
        "--mosaic-probability",
        "0",
        "--mixup-probability",
        "0",
        "--cutmix-probability",
        "0",
        "--copy-paste-probability",
        "0",
        "--targeted-copy-paste-probability",
        "0",
        "--no-pretrained-distillation",
        "--distillation-weight",
        "0",
        "--teacher-focus-binary-loss-weight",
        "0",
        "--teacher-focus-margin-loss-weight",
        "0",
        "--teacher-pairwise-margin-loss-weight",
        "0",
        "--max-train-batches",
        str(stage_spec.max_train_batches),
        "--max-val-batches",
        str(stage_spec.max_val_batches),
        "--skip-final-test",
    ]
    if experiment_config.dinov3_surface_hybrid_mode is not None:
        args.extend(
            [
                "--dinov3-surface-hybrid-mode",
                experiment_config.dinov3_surface_hybrid_mode,
                "--dinov3-surface-initial-gate-scale",
                "0.05",
                "--dinov3-surface-max-gate-scale",
                "0.25",
            ]
        )
    args.extend(
        [
            "--experiment-protocol-id",
            experiment_config.protocol_id,
            "--source-commit",
            str(source_commit).strip().lower(),
            "--source-tree-sha256",
            str(source_tree_sha256).strip().lower(),
            "--dataset-image-tree-sha256",
            str(dataset_image_tree_sha256).strip().lower(),
        ]
    )
    if experiment_config.tempered_class_sampling_power is not None:
        args.extend(
            [
                "--tempered-class-sampling-power",
                str(experiment_config.tempered_class_sampling_power),
            ]
        )
    elif not experiment_config.balanced_epoch_sampling:
        args.append("--disable-balanced-epoch-sampling")
    if experiment_config.tempered_leakage_group_sampling:
        group_manifest = (
            Path(tempered_leakage_group_manifest).resolve()
            if tempered_leakage_group_manifest is not None
            else (Path(data_yaml).resolve().parent / "manifest.csv").resolve()
        )
        args.extend(
            [
                "--tempered-leakage-group-manifest",
                str(group_manifest),
            ]
        )
    args.extend(experiment_config.extra_train_args)
    if (
        experiment_config.key == "prmr-r1"
        and stage_name == "smoke"
        and "--illumination-consistency-start-epoch" in args
    ):
        start_index = args.index("--illumination-consistency-start-epoch") + 1
        args[start_index] = "1"
    if amp_init_scale is not None:
        args.extend(["--amp-init-scale", str(float(amp_init_scale))])
    train_contract_sha256 = hashlib.sha256(
        json.dumps(
            args,
            ensure_ascii=False,
            separators=(",", ":"),
        ).encode("utf-8")
    ).hexdigest()
    args.extend(
        [
            "--recipe-train-contract-sha256",
            train_contract_sha256,
        ]
    )
    if auto_resume:
        args.append("--auto-resume")
    return args


def _git_snapshot(repo_root: Path) -> Dict[str, object]:
    try:
        commit = subprocess.run(
            ["git", "-C", str(repo_root), "rev-parse", "HEAD"],
            check=True,
            capture_output=True,
            text=True,
        ).stdout.strip()
        branch = subprocess.run(
            ["git", "-C", str(repo_root), "branch", "--show-current"],
            check=True,
            capture_output=True,
            text=True,
        ).stdout.strip()
        status = subprocess.run(
            ["git", "-C", str(repo_root), "status", "--short"],
            check=True,
            capture_output=True,
            text=True,
        ).stdout.splitlines()
        return {
            "available": True,
            "commit": commit,
            "branch": branch,
            "status_short": status,
            "clean": not status,
        }
    except (OSError, subprocess.CalledProcessError):
        return {
            "available": False,
            "commit": "",
            "branch": "",
            "status_short": [],
            "clean": False,
        }


def _pretrained_checkpoint_sha256(model: torch.nn.Module) -> str:
    provenance = getattr(model, "pretrained_provenance", {})
    if not isinstance(provenance, Mapping):
        return ""

    containers = [provenance]
    primary_backbone = provenance.get("primary_backbone")
    if isinstance(primary_backbone, Mapping):
        containers.append(primary_backbone)

    observed: set[str] = set()
    for container in containers:
        initialization = container.get("initialization")
        if not isinstance(initialization, Mapping):
            continue
        checkpoint = initialization.get("checkpoint")
        if not isinstance(checkpoint, Mapping):
            continue
        sha256 = str(checkpoint.get("sha256", "")).strip().lower()
        if sha256:
            observed.add(sha256)
    if len(observed) > 1:
        raise RuntimeError(
            "Model exposes conflicting pretrained checkpoint provenance: "
            f"{sorted(observed)}."
        )
    return next(iter(observed), "")


def _preflight_model(
    train_args: Sequence[str],
    *,
    expected_parameter_count: int = EXPECTED_PARAMETER_COUNT,
    expected_external_initialization: bool = True,
) -> Dict[str, object]:
    parsed = parse_train_args(train_args)
    model_config, _, _ = build_configs(parsed)
    model = create_model(num_classes=5, model_config=model_config)
    parameter_count = sum(int(parameter.numel()) for parameter in model.parameters())
    if parameter_count != int(expected_parameter_count):
        raise RuntimeError(
            "Pretrained five-class model parameter count drifted: "
            f"{parameter_count} != {int(expected_parameter_count)}."
        )
    groups = build_optimizer_param_groups(
        model,
        weight_decay=0.05,
        learning_rate=1.5e-4,
        backbone_lr_scale=0.1,
    )
    group_names = sorted(str(group["name"]) for group in groups)
    weight_decay_by_parameter_id = {
        id(parameter): float(group["weight_decay"])
        for group in groups
        for parameter in group["params"]
    }
    declared_no_decay = (
        set(str(name) for name in model.no_weight_decay())
        if callable(getattr(model, "no_weight_decay", None))
        else set()
    )
    no_decay_violations = [
        name
        for name, parameter in model.named_parameters()
        if parameter.requires_grad
        and (name in declared_no_decay or parameter.ndim <= 1)
        and weight_decay_by_parameter_id.get(id(parameter), float("inf")) != 0.0
    ]
    if no_decay_violations:
        raise RuntimeError(
            "DINO/timm no-weight-decay contract is inactive: "
            f"{no_decay_violations}."
        )
    backbone_lrs = sorted(
        {
            float(group["lr"])
            for group in groups
            if str(group["name"]).startswith("backbone_") and "lr" in group
        }
    )
    if (
        len(backbone_lrs) != 1
        or not math.isclose(backbone_lrs[0], 1.5e-5, rel_tol=0.0, abs_tol=1e-12)
    ):
        raise RuntimeError(f"Pretrained backbone LR split is inactive: {backbone_lrs}.")
    model.eval()
    hybrid_initialization: Dict[str, object] | None = None
    with torch.inference_mode():
        probe = torch.zeros(1, 3, 256, 256)
        if bool(getattr(model, "is_pretrained_surface_patch_hybrid_v2", False)):
            fused_tokens, fusion_trace = model.forward_features_with_fusion_trace(
                probe
            )
            logits = model.forward_head(fused_tokens)
            native_logits = model.backbone.forward_head(
                model.backbone.forward_features(probe)
            )
            exact_native_identity = bool(torch.equal(logits, native_logits))
            max_abs_logit_delta = float(
                (logits - native_logits).abs().max().cpu().item()
            )
            initial_residual_ratio_max = float(
                fusion_trace["gated_residual_norm_ratio"].max().cpu().item()
            )
            if not exact_native_identity or max_abs_logit_delta != 0.0:
                raise RuntimeError(
                    "Hybrid V2 must equal native DINO exactly at step zero; "
                    f"max_abs_logit_delta={max_abs_logit_delta}."
                )
            if initial_residual_ratio_max != 0.0:
                raise RuntimeError(
                    "Hybrid V2 step-zero residual must be zero; "
                    f"observed ratio={initial_residual_ratio_max}."
                )
            hybrid_initialization = {
                "native_logit_identity_exact": exact_native_identity,
                "max_abs_logit_delta": max_abs_logit_delta,
                "argmax_mismatch_count": int(
                    (logits.argmax(dim=1) != native_logits.argmax(dim=1))
                    .sum()
                    .cpu()
                    .item()
                ),
                "initial_gate_mean": float(
                    fusion_trace["gate_mean"].cpu().item()
                ),
                "initial_residual_norm_ratio_max": (
                    initial_residual_ratio_max
                ),
                "branch_opening_contract": (
                    "zero_output_projection_step0;projection_gradient_step1;"
                    "upstream_specialist_and_gate_after_first_update"
                ),
            }
        else:
            logits = model(probe)
    if not torch.is_tensor(logits) or tuple(logits.shape) != (1, 5):
        raise RuntimeError(f"Unexpected DINOv3 output shape: {getattr(logits, 'shape', None)}.")
    observed_sha = _pretrained_checkpoint_sha256(model)
    expected_sha = DINO_SHA256 if expected_external_initialization else ""
    if observed_sha != expected_sha:
        raise RuntimeError(
            "Model external-initialization provenance is wrong: "
            f"observed_sha={observed_sha!r}, expected_sha={expected_sha!r}."
        )
    observed_external_initialization = bool(
        getattr(model, "pretrained_provenance", {}).get(
            "external_initialization_used",
            getattr(model, "is_pretrained_timm_classifier", False),
        )
    )
    if observed_external_initialization != bool(expected_external_initialization):
        raise RuntimeError(
            "Model external-initialization flag is wrong: "
            f"observed={observed_external_initialization}, "
            f"expected={bool(expected_external_initialization)}."
        )
    trainable_parameter_count = int(
        sum(parameter.numel() for parameter in model.parameters() if parameter.requires_grad)
    )
    result = {
        "parameter_count": parameter_count,
        "expected_parameter_count": int(expected_parameter_count),
        "trainable_parameter_count": trainable_parameter_count,
        "frozen_parameter_count": int(parameter_count - trainable_parameter_count),
        "optimizer_groups": group_names,
        "declared_no_decay_parameters": sorted(declared_no_decay),
        "no_decay_violations": no_decay_violations,
        "backbone_learning_rates": backbone_lrs,
        "output_shape": list(logits.shape),
        "pretrained_checkpoint_sha256": observed_sha,
        "external_initialization_used": observed_external_initialization,
    }
    if hybrid_initialization is not None:
        result["hybrid_initialization"] = hybrid_initialization
        telemetry = getattr(model, "parameter_telemetry", None)
        if callable(telemetry):
            result["hybrid_parameter_telemetry"] = telemetry()
    return result


def _best_history_row(run_dir: Path) -> Dict[str, object]:
    summary_path = run_dir / "summary.json"
    history_path = run_dir / "history.csv"
    if not summary_path.is_file() or not history_path.is_file():
        return {}
    summary = json.loads(summary_path.read_text(encoding="utf-8"))
    best_epoch = int(summary.get("best_epoch", -1))
    with history_path.open("r", encoding="utf-8-sig", newline="") as handle:
        rows = list(csv.DictReader(handle))
    selected = next(
        (row for row in rows if int(float(row.get("epoch", -1))) == best_epoch),
        {},
    )
    return {
        "best_epoch": best_epoch,
        "best_macro_f1": summary.get("best_macro_f1"),
        "best_class1_f1": selected.get("val_class_1_f1"),
        "best_class1_precision": selected.get("val_class_1_precision"),
        "best_class1_recall": selected.get("val_class_1_recall"),
        "stop_reason": summary.get("stop_reason"),
    }


def parse_args(argv: Sequence[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Run the clean DINOv3 direct baseline on canonical class_f."
    )
    parser.add_argument(
        "--mode",
        choices=("preflight", "smoke", "probe", "full"),
        default="preflight",
    )
    parser.add_argument(
        "--experiment",
        choices=tuple(EXPERIMENTS),
        default="b0",
        help=(
            "b0 preserves strict balanced sampling; b1-natural changes only "
            "the train sampler; b1-margin0 changes only LDAM max margin; "
            "b2-tempered-p05 uses q_c proportional n_c**0.5; "
            "b9-b2-reference-completion repeats B2 semantics only as an "
            "exploratory canonical reference; b10-group-tempered-p05 changes "
            "only train sampling within each class to leakage-group-uniform; "
            "hybrid-v2-generic/hybrid-v2-local-surface are the matched "
            "capacity-control/surface probe pair on B10 sampling; the "
            "hybrid-v2-local-surface-randominit arm isolates initialization; "
            "prmr-r1-control/prmr-r1 are a "
            "matched probe pair and never authorize full training by themselves."
        ),
    )
    parser.add_argument("--python", type=Path, default=Path(sys.executable))
    parser.add_argument("--data", type=Path, required=True)
    parser.add_argument(
        "--canonical-attestation",
        type=Path,
        default=None,
        help=(
            "Optional JSON produced by validate_canonical_classf earlier in the "
            "same immutable runtime; avoids rehashing the 1 GiB image tree."
        ),
    )
    parser.add_argument(
        "--train-data",
        type=Path,
        default=None,
        help=(
            "Optional test-locked YAML for train/val. --data remains the "
            "fingerprinted canonical contract."
        ),
    )
    parser.add_argument(
        "--dino-checkpoint",
        type=Path,
        default=None,
        help=(
            "Locked DINOv3 weights; required only when the selected arm uses "
            "external initialization."
        ),
    )
    parser.add_argument("--output-dir", type=Path, default=Path("runs"))
    parser.add_argument("--run-tag", type=str, default="20260731")
    parser.add_argument("--batch-size", type=int, choices=sorted(ACCUMULATION_BY_BATCH), default=24)
    parser.add_argument("--num-workers", type=int, default=4)
    parser.add_argument("--eval-num-workers", type=int, default=2)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument(
        "--amp-init-scale",
        type=float,
        default=None,
        help=(
            "Optional fp16 GradScaler initial scale. Omit to preserve the "
            "historical recipe contract and training default (65536)."
        ),
    )
    parser.add_argument(
        "--amp-dtype",
        choices=("auto", "bf16", "fp16"),
        default="auto",
    )
    parser.add_argument("--auto-resume", action="store_true", default=False)
    parser.add_argument("--confirm-full", action="store_true", default=False)
    parser.add_argument(
        "--source-commit",
        default="",
        help="Required 40-hex source commit for probe/full archive execution.",
    )
    parser.add_argument(
        "--source-tree-sha256",
        default="",
        help="Required source-tree digest for probe/full archive execution.",
    )
    parser.add_argument(
        "--skip-focused-tests",
        action="store_true",
        default=False,
        help="Engineering-only; preflight manifest records the skipped tests.",
    )
    return parser.parse_args(argv)


def main(argv: Sequence[str] | None = None) -> None:
    args = parse_args(argv)
    experiment_config = experiment_spec(args.experiment)
    if args.mode == "full" and not experiment_config.full_train_authorized:
        raise RuntimeError(
            f"{experiment_config.key} full training is closed by its protocol; "
            "use only an explicitly authorized control/reference experiment."
        )
    if (
        experiment_config.key == "b9-b2-reference-completion"
        and args.mode in {"smoke", "probe"}
    ):
        raise RuntimeError(
            "B9 is a one-run 30-epoch-cap reference completion; only "
            "preflight or full mode is allowed."
        )
    if args.mode in {"probe", "full"} and args.amp_dtype == "auto":
        raise RuntimeError(
            f"{args.mode} requires explicit --amp-dtype bf16 or fp16 so "
            "numeric precision cannot drift across GPU types."
        )
    experiment_slug = experiment_config.key.replace("-", "_")
    repo_root = Path(__file__).resolve().parents[2]
    data_yaml = args.data.resolve()
    training_data_yaml = (
        args.train_data.resolve() if args.train_data is not None else data_yaml
    )
    dino_checkpoint: Path | None = None
    dino_sha = ""
    if experiment_config.external_initialization_used:
        if args.dino_checkpoint is None:
            raise ValueError(
                f"{experiment_config.key} requires --dino-checkpoint."
            )
        dino_checkpoint = _absolute_path_without_resolving_symlink(
            args.dino_checkpoint
        )
        if not dino_checkpoint.is_file():
            raise FileNotFoundError(
                f"DINOv3 checkpoint not found: {dino_checkpoint}"
            )
        dino_sha = _sha256(dino_checkpoint)
        if dino_sha != DINO_SHA256:
            raise ValueError(
                "DINOv3 SHA-256 mismatch: "
                f"observed={dino_sha}, expected={DINO_SHA256}."
            )
    output_dir = args.output_dir.resolve()
    output_dir.mkdir(parents=True, exist_ok=True)
    dataset_contract = (
        load_canonical_attestation(
            args.canonical_attestation,
            data_yaml=data_yaml,
        )
        if args.canonical_attestation is not None
        else validate_canonical_classf(data_yaml)
    )
    development_contract = None
    if training_data_yaml != data_yaml:
        development_contract = validate_development_data_yaml(
            training_data_yaml,
            canonical_contract=dataset_contract,
        )
    git = _git_snapshot(repo_root)
    observed_source_tree_sha256 = source_tree_sha256(repo_root)
    archive_source: Dict[str, object] = {
        "commit": str(args.source_commit).strip().lower(),
        "tree_sha256": observed_source_tree_sha256,
        "expected_tree_sha256": str(args.source_tree_sha256).strip().lower(),
    }
    if git["available"]:
        branch = str(git["branch"])
        if not branch.startswith("research/pretrained-"):
            raise RuntimeError(
                f"B0 requires research/pretrained-* branch, observed={branch!r}."
            )
        if args.mode in {"probe", "full"} and not bool(git["clean"]):
            raise RuntimeError(f"{args.mode} requires a clean worktree.")
    elif args.mode in {"probe", "full"}:
        source_commit = str(args.source_commit).strip().lower()
        expected_tree_sha256 = str(args.source_tree_sha256).strip().lower()
        if re.fullmatch(r"[0-9a-f]{40}", source_commit) is None:
            raise RuntimeError(
                f"{args.mode} archive execution requires --source-commit as 40 hex."
            )
        if re.fullmatch(r"[0-9a-f]{64}", expected_tree_sha256) is None:
            raise RuntimeError(
                f"{args.mode} archive execution requires --source-tree-sha256 as 64 hex."
            )
        if expected_tree_sha256 != observed_source_tree_sha256:
            raise RuntimeError(
                "Archive source-tree SHA-256 mismatch: "
                f"observed={observed_source_tree_sha256}, "
                f"expected={expected_tree_sha256}."
            )
    if args.mode == "full" and not bool(args.confirm_full):
        raise RuntimeError("Full requires --confirm-full after reviewing the probe.")

    stage = "smoke" if args.mode == "preflight" else args.mode
    train_args = build_train_args(
        data_yaml=training_data_yaml,
        dino_checkpoint=dino_checkpoint,
        output_dir=output_dir,
        stage=stage,
        run_tag=args.run_tag,
        batch_size=args.batch_size,
        num_workers=args.num_workers,
        eval_num_workers=args.eval_num_workers,
        seed=args.seed,
        amp_init_scale=args.amp_init_scale,
        amp_dtype=args.amp_dtype,
        auto_resume=bool(args.auto_resume),
        source_commit=(
            str(git.get("commit", "")).strip().lower()
            if bool(git.get("available"))
            else str(args.source_commit).strip().lower()
        ),
        source_tree_sha256=observed_source_tree_sha256,
        dataset_image_tree_sha256=str(
            dataset_contract["image_tree_sha256"]
        ).lower(),
        experiment=experiment_config.key,
        tempered_leakage_group_manifest=(data_yaml.parent / "manifest.csv"),
    )

    focused_tests = [
        repo_root / "tests" / "test_pretrained_semantic_branch.py",
        repo_root / "tests" / "test_timm_classifier_model.py",
        repo_root / "tests" / "test_canonical_classf_defaults.py",
        repo_root / "tests" / "test_deploy_classification_folder.py",
        repo_root / "tests" / "test_pretrained_classf_recipe.py",
        repo_root / "tests" / "test_tempered_leakage_group_sampler.py",
        repo_root / "tests" / "test_dinov3_surface_patch_hybrid_v2.py",
        repo_root / "tests" / "test_assess_pretrained_classf_probe.py",
        repo_root / "tests" / "test_illumination_consistency_loss.py",
        repo_root / "tests" / "test_resume_weight_and_distillation_source.py",
        repo_root / "tests" / "test_attention_viz_headless.py",
        repo_root / "tests" / "test_mobile_onnx_quantization.py",
    ]
    test_result: Dict[str, object] = {
        "skipped": bool(args.skip_focused_tests),
        "paths": [str(path) for path in focused_tests],
    }
    if not args.skip_focused_tests:
        completed = subprocess.run(
            [str(args.python.resolve()), "-m", "pytest", "-q", *map(str, focused_tests)],
            cwd=repo_root,
            check=False,
        )
        test_result["returncode"] = int(completed.returncode)
        if completed.returncode != 0:
            raise RuntimeError(
                f"Focused {experiment_config.key} tests failed."
            )

    model_preflight = _preflight_model(
        train_args,
        expected_parameter_count=experiment_config.expected_parameter_count,
        expected_external_initialization=(
            experiment_config.external_initialization_used
        ),
    )
    preflight_dir = output_dir / (
        f"preflight_classf_{experiment_slug}_{args.run_tag}"
    )
    preflight_dir.mkdir(parents=True, exist_ok=True)
    manifest = {
        "schema_version": 1,
        "protocol": experiment_config.protocol_id,
        "experiment": {
            "key": experiment_config.key,
            "balanced_epoch_sampling": (
                experiment_config.balanced_epoch_sampling
            ),
            "tempered_class_sampling_power": (
                experiment_config.tempered_class_sampling_power
            ),
            "tempered_leakage_group_sampling": (
                experiment_config.tempered_leakage_group_sampling
            ),
            "single_semantic_delta_from_b0": (
                experiment_config.single_semantic_delta_from_b0
            ),
            "model_type": experiment_config.model_type,
            "dinov3_surface_hybrid_mode": (
                experiment_config.dinov3_surface_hybrid_mode
            ),
            "expected_parameter_count": (
                experiment_config.expected_parameter_count
            ),
            "external_initialization_used": (
                experiment_config.external_initialization_used
            ),
            "research_role": experiment_config.research_role,
            "promotion_eligible": experiment_config.promotion_eligible,
            "reference_experiment": experiment_config.reference_experiment,
            "known_probe_class1_f1": experiment_config.known_probe_class1_f1,
            "required_probe_class1_f1": (
                experiment_config.required_probe_class1_f1
            ),
            "full_train_authorized": (
                experiment_config.full_train_authorized
            ),
        },
        "created_utc": datetime.now(timezone.utc).isoformat(),
        "mode": args.mode,
        "test_locked": True,
        "old_dataset_trained_weights_used": False,
        "old_teacher_cache_used": False,
        "dataset": dataset_contract,
        "development_data": development_contract,
        "dino": {
            "model_name": DINO_MODEL_NAME,
            "checkpoint": str(dino_checkpoint) if dino_checkpoint is not None else None,
            "sha256": dino_sha,
            "source_url": (
                DINO_SOURCE_URL if experiment_config.external_initialization_used else None
            ),
            "source_revision": (
                DINO_SOURCE_REVISION
                if experiment_config.external_initialization_used
                else None
            ),
            "source_license": (
                DINO_SOURCE_LICENSE
                if experiment_config.external_initialization_used
                else None
            ),
            "consumed_by_model": experiment_config.external_initialization_used,
        },
        "git": git,
        "archive_source": archive_source,
        "runtime": {
            "python": str(args.python.resolve()),
            "torch": torch.__version__,
            "timm": importlib.metadata.version("timm"),
            "cuda_available": bool(torch.cuda.is_available()),
            "cuda_device": torch.cuda.get_device_name(0) if torch.cuda.is_available() else None,
            "amp_dtype_request": args.amp_dtype,
            "batch_size": int(args.batch_size),
            "grad_accum_steps": int(ACCUMULATION_BY_BATCH[args.batch_size]),
            "effective_batch_size": 48,
        },
        "focused_tests": test_result,
        "model_preflight": model_preflight,
        "train_args": train_args,
    }
    manifest_path = preflight_dir / f"{args.mode}_manifest.json"
    manifest_path.write_text(
        json.dumps(manifest, ensure_ascii=False, indent=2) + "\n",
        encoding="utf-8",
    )
    print(
        f"{experiment_config.key} preflight passed: {manifest_path}",
        flush=True,
    )
    if args.mode == "preflight":
        return
    if not torch.cuda.is_available():
        raise RuntimeError("CUDA is required for smoke/probe/full training.")

    environment = os.environ.copy()
    environment["PYTHONPATH"] = str(repo_root)
    environment["TRKH_AMP_DTYPE"] = str(args.amp_dtype)
    environment.setdefault("OMP_NUM_THREADS", "4")
    command = [str(args.python.resolve()), "-m", "trkh.training.train", *train_args]
    run_dir = output_dir / run_name(
        stage,
        args.run_tag,
        experiment=experiment_config.key,
    )
    if run_dir.exists() and any(run_dir.iterdir()) and not bool(args.auto_resume):
        raise RuntimeError(
            f"Run directory already exists: {run_dir}. "
            "Use a new --run-tag or explicitly pass --auto-resume."
        )
    if bool(args.auto_resume) and not (run_dir / "checkpoints" / "last.pt").is_file():
        raise RuntimeError(
            f"--auto-resume requires {run_dir / 'checkpoints' / 'last.pt'}."
        )
    resume_contract = None
    if bool(args.auto_resume):
        resume_contract = validate_auto_resume_checkpoint(
            run_dir / "checkpoints" / "last.pt",
            training_data_yaml=training_data_yaml,
            expected_train_args=train_args,
        )
        manifest["auto_resume_checkpoint"] = resume_contract
        manifest_path.write_text(
            json.dumps(manifest, ensure_ascii=False, indent=2) + "\n",
            encoding="utf-8",
        )
    completed = subprocess.run(
        command,
        cwd=repo_root,
        env=environment,
        check=False,
    )
    marker = {
        "schema_version": 1,
        "protocol": experiment_config.protocol_id,
        "experiment": experiment_config.key,
        "research_role": experiment_config.research_role,
        "promotion_eligible": experiment_config.promotion_eligible,
        "reference_experiment": experiment_config.reference_experiment,
        "full_train_authorized": experiment_config.full_train_authorized,
        "stage": stage,
        "completed_utc": datetime.now(timezone.utc).isoformat(),
        "returncode": int(completed.returncode),
        "preflight_manifest": str(manifest_path),
        "test_locked": True,
        "git_commit": git.get("commit"),
        "metrics": _best_history_row(run_dir),
    }
    run_dir.mkdir(parents=True, exist_ok=True)
    marker_path = run_dir / f"{experiment_slug}_stage_complete.json"
    marker_path.write_text(
        json.dumps(marker, ensure_ascii=False, indent=2) + "\n",
        encoding="utf-8",
    )
    if completed.returncode != 0:
        raise RuntimeError(
            f"{experiment_config.key} {stage} failed with code "
            f"{completed.returncode}; evidence: {run_dir}."
        )
    print(
        f"{experiment_config.key} {stage} completed: {marker_path}",
        flush=True,
    )


if __name__ == "__main__":
    main()
