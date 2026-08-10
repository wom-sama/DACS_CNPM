from __future__ import annotations

import argparse
import csv
import hashlib
import importlib.metadata
import json
from pathlib import Path, PureWindowsPath
import subprocess
import sys
from typing import Dict, Iterable, List, Mapping, Sequence

import numpy as np
import torch


REPOSITORY_ROOT = Path(__file__).resolve().parents[1]
LOCK_PARENT_COMMIT = "dc734a7eed4cf548930a748f05a5fcd7c63d146b"
PROTOCOL_PATH = (
    REPOSITORY_ROOT
    / "docs"
    / "TRKH_5CLASS_BALANCED_BCE_FROZEN_EMBEDDING_A0_PROTOCOL_20260724.md"
)
LOCK_PATH = (
    REPOSITORY_ROOT
    / "docs"
    / "TRKH_5CLASS_BALANCED_BCE_FROZEN_EMBEDDING_A0_LOCK_20260724.json"
)
LOCK_SHA_PATH = LOCK_PATH.with_suffix(".sha256")

EMBEDDING_CACHE = (
    REPOSITORY_ROOT
    / "runs"
    / "diagnostic_reslt_embedding_cache_keeper_yolof_20260712"
    / "train_embeddings.npz"
)
CIDT_PREDICTIONS = (
    REPOSITORY_ROOT
    / "runs"
    / "audit_cidt_readiness_full_train_20260714"
    / "predictions_all_conditions.csv"
)
KEEPER_CHECKPOINT = (
    REPOSITORY_ROOT
    / "runs"
    / "probe_v8_yolof_pairroute_teacherfocusbinary015_boundarydrop_bboxprior_120b_2e_20260701"
    / "checkpoints"
    / "best.pt"
)
KEEPER_CONFIG = KEEPER_CHECKPOINT.parents[1] / "resolved_config.json"
CURRENT_BEST_COMMANDS = (
    REPOSITORY_ROOT
    / "docs"
    / "TRKH_CURRENT_BEST_FULL_TRAIN_COMMANDS_20260706.txt"
)
CURRENT_BEST_HISTORY = (
    REPOSITORY_ROOT
    / "docs"
    / "TRKH_CURRENT_BEST_COMMAND_UPDATE_HISTORY.txt"
)
YOLO_DATA_YAML = Path(r"D:\DataAI\AIEx\newdataset\yolo_f\data.yaml")
LIVT_REPOSITORY = Path(r"D:\DataAI\external_sources\official\LiVT")
LIVT_PAPER = Path(
    r"D:\DataAI\external_sources\papers"
    r"\Xu_Learning_Imbalanced_Data_With_Vision_Transformers_CVPR_2023.pdf"
)

EXPECTED_LIVT_COMMIT = "68546ef189c486caa271066d8bfa25ec214192df"
EXPECTED_LIVT_TREE = "ec1fb1784566fd5f73f0f9710b2e9de06cb0e50c"
EXPECTED_LIVT_PAPER_SHA256 = (
    "79d1afec19528d639209661f4694c8b6506aed1e4e9d6a3ee54ce5be89c505ba"
)
EXPECTED_LIVT_LICENSE_SHA256 = (
    "ebed29105302a78fecb80e3f2c174d03ffba136b664596ce3e41d5fc20002e32"
)
EXPECTED_LIVT_LOSS_SHA256 = (
    "d73c21ba3a9ca8afbc5b230b48df97555dd57a4d91551fae89167d6f5baf8246"
)

ROWS = 9215
EMBEDDING_DIM = 256
CLASSES = 5
OUTER_FOLDS = 5
EPOCHS = 30
BATCH_SIZE = 64
PRIMARY_SEED = 20260724
REPEAT_SEED_OFFSET = 100000
VISUAL_SAMPLE_INDICES = np.asarray(
    [
        4002,
        3195,
        3284,
        4613,
        3217,
        2127,
        2707,
        5213,
        4616,
        4770,
        3955,
        4012,
        4624,
        3248,
        2718,
        5732,
        5327,
        6113,
        4916,
        3624,
    ],
    dtype=np.int64,
)


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def _numeric_array_sha256(value: np.ndarray) -> str:
    array = np.ascontiguousarray(value)
    digest = hashlib.sha256()
    digest.update(str(array.dtype).encode("ascii"))
    digest.update(np.asarray(array.shape, dtype=np.int64).tobytes())
    digest.update(array.tobytes())
    return digest.hexdigest()


def _string_sequence_sha256(values: Iterable[object]) -> str:
    encoded = [str(value).encode("utf-8") for value in values]
    digest = hashlib.sha256()
    digest.update(np.asarray([len(encoded)], dtype=np.int64).tobytes())
    for value in encoded:
        digest.update(np.asarray([len(value)], dtype=np.int64).tobytes())
        digest.update(value)
    return digest.hexdigest()


def _json_sha256(value: object) -> str:
    payload = json.dumps(
        value,
        ensure_ascii=False,
        separators=(",", ":"),
        sort_keys=True,
    ).encode("utf-8")
    return hashlib.sha256(payload).hexdigest()


def _git(repository: Path, *arguments: str) -> str:
    result = subprocess.run(
        ["git", "-C", str(repository), *arguments],
        check=True,
        capture_output=True,
        text=True,
    )
    return result.stdout.strip()


def _package_version(name: str) -> str:
    return importlib.metadata.version(name)


def _assert_equal(actual: object, expected: object, label: str) -> None:
    if actual != expected:
        raise ValueError(f"{label}: expected {expected!r}, observed {actual!r}")


def _load_clean_cidt_rows() -> Dict[int, Dict[str, object]]:
    rows: Dict[int, Dict[str, object]] = {}
    with CIDT_PREDICTIONS.open("r", encoding="utf-8", newline="") as handle:
        for raw in csv.DictReader(handle):
            if raw["condition"] != "clean":
                continue
            sample_index = int(raw["sample_index"])
            if sample_index in rows:
                raise ValueError(f"Duplicate clean sample index {sample_index}")
            rows[sample_index] = {
                "sample_index": sample_index,
                "source_stem": raw["source_stem"],
                "image_path": raw["image_path"],
                "fold": int(raw["fold"]),
                "target_index": int(raw["target_index"]),
            }
    _assert_equal(len(rows), ROWS, "clean CIDT row count")
    _assert_equal(sorted(rows), list(range(ROWS)), "clean CIDT sample indices")
    return rows


def _validate_train_path(path: object) -> str:
    value = str(path)
    parts = [part.lower() for part in PureWindowsPath(value).parts]
    forbidden = {"val", "valid", "validation", "test"}
    if forbidden.intersection(parts):
        raise ValueError(f"Forbidden split component in cache path: {value}")
    required_suffix = ("yolo_f", "images", "train")
    lowered = tuple(parts)
    if not any(
        lowered[index : index + len(required_suffix)] == required_suffix
        for index in range(len(lowered) - len(required_suffix) + 1)
    ):
        raise ValueError(f"Cache path is not under yolo_f/images/train: {value}")
    return value


def _load_inputs() -> Dict[str, np.ndarray]:
    with np.load(EMBEDDING_CACHE, allow_pickle=True) as archive:
        arrays = {key: np.asarray(archive[key]) for key in archive.files}
    expected = {
        "embeddings": ((ROWS, EMBEDDING_DIM), np.dtype(np.float32)),
        "probabilities": ((ROWS, CLASSES), np.dtype(np.float32)),
        "labels": ((ROWS,), np.dtype(np.int64)),
        "base_predictions": ((ROWS,), np.dtype(np.int64)),
        "paths": ((ROWS,), np.dtype(object)),
        "sample_index": ((ROWS,), np.dtype(np.int64)),
    }
    _assert_equal(set(arrays), set(expected), "embedding cache keys")
    for key, (shape, dtype) in expected.items():
        _assert_equal(arrays[key].shape, shape, f"{key} shape")
        _assert_equal(arrays[key].dtype, dtype, f"{key} dtype")
    _assert_equal(
        arrays["sample_index"].tolist(),
        list(range(ROWS)),
        "cache sample index order",
    )
    if not np.isfinite(arrays["embeddings"]).all():
        raise ValueError("Embedding cache contains non-finite values")
    if not np.isfinite(arrays["probabilities"]).all():
        raise ValueError("Probability cache contains non-finite values")
    for path in arrays["paths"]:
        _validate_train_path(path)
    return arrays


def _head_state(seed: int) -> Dict[str, object]:
    torch.manual_seed(seed)
    head = torch.nn.Linear(EMBEDDING_DIM, CLASSES, bias=True)
    weight = head.weight.detach().cpu().numpy().astype(np.float32, copy=True)
    bias = head.bias.detach().cpu().numpy().astype(np.float32, copy=True)
    state = {
        "seed": seed,
        "weight_sha256": _numeric_array_sha256(weight),
        "bias_sha256": _numeric_array_sha256(bias),
    }
    state["combined_sha256"] = _json_sha256(state)
    return state


def _epoch_orders(indices: np.ndarray, seed: int) -> Dict[str, object]:
    rng = np.random.default_rng(seed)
    ordered: List[np.ndarray] = []
    per_epoch: List[str] = []
    for _ in range(EPOCHS):
        epoch = indices[rng.permutation(indices.size)].astype(
            np.int64, copy=False
        )
        ordered.append(epoch)
        per_epoch.append(_numeric_array_sha256(epoch))
    concatenated = np.concatenate(ordered)
    return {
        "seed": seed,
        "per_epoch_sha256": per_epoch,
        "all_epochs_sha256": _numeric_array_sha256(concatenated),
        "rows_per_epoch": int(indices.size),
        "optimizer_steps": int(
            EPOCHS * ((indices.size + BATCH_SIZE - 1) // BATCH_SIZE)
        ),
    }


def _fold_payload(
    fold: int,
    folds: np.ndarray,
    labels: np.ndarray,
    sources: Sequence[str],
) -> Dict[str, object]:
    held = np.flatnonzero(folds == fold).astype(np.int64)
    fit = np.flatnonzero(folds != fold).astype(np.int64)
    fit_counts = np.bincount(labels[fit], minlength=CLASSES).astype(np.int64)
    held_counts = np.bincount(labels[held], minlength=CLASSES).astype(np.int64)
    prior = fit_counts.astype(np.float64) / float(fit_counts.sum())
    balanced_bce_bias = np.log(prior) - np.log1p(-prior)
    balanced_softmax_bias = 0.25 * np.log(prior)
    reversed_prior_bias = -balanced_bce_bias
    if not np.isfinite(balanced_bce_bias).all():
        raise ValueError(f"Fold {fold} has a non-finite Bal-BCE bias")

    fit_sources = {sources[index] for index in fit}
    held_sources = {sources[index] for index in held}
    overlap = sorted(fit_sources.intersection(held_sources))
    if overlap:
        raise ValueError(f"Fold {fold} source overlap: {overlap[:3]}")

    primary_seed = PRIMARY_SEED + fold
    repeat_seed = PRIMARY_SEED + REPEAT_SEED_OFFSET + fold
    return {
        "outer_fold": fold,
        "fit_rows": int(fit.size),
        "held_rows": int(held.size),
        "fit_class_counts": fit_counts.tolist(),
        "held_class_counts": held_counts.tolist(),
        "fit_prior_float64": prior.tolist(),
        "balanced_bce_bias_float64": balanced_bce_bias.tolist(),
        "balanced_bce_bias_float32_sha256": _numeric_array_sha256(
            balanced_bce_bias.astype(np.float32)
        ),
        "balanced_softmax_tau025_bias_float64": (
            balanced_softmax_bias.tolist()
        ),
        "balanced_softmax_tau025_bias_float32_sha256": (
            _numeric_array_sha256(balanced_softmax_bias.astype(np.float32))
        ),
        "reversed_prior_bias_float64": reversed_prior_bias.tolist(),
        "reversed_prior_bias_float32_sha256": _numeric_array_sha256(
            reversed_prior_bias.astype(np.float32)
        ),
        "fit_indices_sha256": _numeric_array_sha256(fit),
        "held_indices_sha256": _numeric_array_sha256(held),
        "fit_labels_sha256": _numeric_array_sha256(labels[fit]),
        "held_labels_sha256": _numeric_array_sha256(labels[held]),
        "fit_sources_sha256": _string_sequence_sha256(
            sources[index] for index in fit
        ),
        "held_sources_sha256": _string_sequence_sha256(
            sources[index] for index in held
        ),
        "source_overlap": 0,
        "primary_initial_head": _head_state(primary_seed),
        "repeat_initial_head": _head_state(repeat_seed),
        "primary_orders": _epoch_orders(fit, primary_seed),
        "repeat_orders": _epoch_orders(fit, repeat_seed),
    }


def _immutable(path: Path) -> Dict[str, object]:
    return {
        "path": str(path),
        "bytes": path.stat().st_size,
        "sha256": _sha256(path),
    }


def build_lock() -> Dict[str, object]:
    required = [
        PROTOCOL_PATH,
        EMBEDDING_CACHE,
        CIDT_PREDICTIONS,
        KEEPER_CHECKPOINT,
        KEEPER_CONFIG,
        CURRENT_BEST_COMMANDS,
        CURRENT_BEST_HISTORY,
        YOLO_DATA_YAML,
        LIVT_PAPER,
        LIVT_REPOSITORY / "LICENSE",
        LIVT_REPOSITORY / "util" / "loss.py",
    ]
    missing = [str(path) for path in required if not path.exists()]
    if missing:
        raise FileNotFoundError(f"Missing lock inputs: {missing}")

    _assert_equal(
        _git(LIVT_REPOSITORY, "rev-parse", "HEAD"),
        EXPECTED_LIVT_COMMIT,
        "LiVT commit",
    )
    _assert_equal(
        _git(LIVT_REPOSITORY, "rev-parse", "HEAD^{tree}"),
        EXPECTED_LIVT_TREE,
        "LiVT tree",
    )
    _assert_equal(
        _sha256(LIVT_PAPER), EXPECTED_LIVT_PAPER_SHA256, "LiVT paper SHA"
    )
    _assert_equal(
        _sha256(LIVT_REPOSITORY / "LICENSE"),
        EXPECTED_LIVT_LICENSE_SHA256,
        "LiVT license SHA",
    )
    _assert_equal(
        _sha256(LIVT_REPOSITORY / "util" / "loss.py"),
        EXPECTED_LIVT_LOSS_SHA256,
        "LiVT loss source SHA",
    )

    arrays = _load_inputs()
    cidt = _load_clean_cidt_rows()
    labels = arrays["labels"].astype(np.int64, copy=False)
    folds = np.empty(ROWS, dtype=np.int64)
    sources: List[str] = []
    for index in range(ROWS):
        row = cidt[index]
        _assert_equal(
            int(labels[index]), row["target_index"], f"label row {index}"
        )
        _assert_equal(
            str(arrays["paths"][index]),
            row["image_path"],
            f"image path row {index}",
        )
        folds[index] = int(row["fold"])
        sources.append(str(row["source_stem"]))

    _assert_equal(
        sorted(np.unique(folds).tolist()),
        list(range(OUTER_FOLDS)),
        "outer fold values",
    )
    source_to_fold: Dict[str, int] = {}
    for source, fold in zip(sources, folds.tolist()):
        previous = source_to_fold.setdefault(source, fold)
        _assert_equal(previous, fold, f"source fold for {source}")

    class_counts = np.bincount(labels, minlength=CLASSES).astype(np.int64)
    folds_payload = [
        _fold_payload(fold, folds, labels, sources)
        for fold in range(OUTER_FOLDS)
    ]
    visual_folds = folds[VISUAL_SAMPLE_INDICES].astype(np.int8)

    runtime = {
        "python": sys.version.split()[0],
        "numpy": np.__version__,
        "torch": torch.__version__,
        "torch_cuda": torch.version.cuda,
        "scikit_learn": _package_version("scikit-learn"),
        "pillow": _package_version("Pillow"),
        "psutil": _package_version("psutil"),
        "device_name": (
            torch.cuda.get_device_name(0)
            if torch.cuda.is_available()
            else "CUDA unavailable during lock build"
        ),
    }

    return {
        "schema_version": 1,
        "protocol_id": "trkh_balanced_bce_frozen_embedding_a0_20260724",
        "created_local_date": "2026-07-24",
        "lock_state": (
            "prospective_before_candidate_implementation_training_or_metrics"
        ),
        "repository_lock_parent_commit": LOCK_PARENT_COMMIT,
        "runtime": runtime,
        "hash_encoding": {
            "numeric_array": (
                "sha256(str(dtype) ASCII || int64 shape bytes || contiguous "
                "array bytes)"
            ),
            "string_sequence": (
                "sha256(int64 count || repeated int64 UTF-8 byte length || "
                "UTF-8 bytes)"
            ),
            "json_value": (
                "sha256(canonical UTF-8 JSON with sorted keys and compact "
                "separators)"
            ),
        },
        "protocol": _immutable(PROTOCOL_PATH),
        "lock_builder": _immutable(Path(__file__).resolve()),
        "claim_boundary": {
            "stage": "train-only source-disjoint frozen-embedding readout A0",
            "does_not_claim": [
                "end-to-end LiVT reproduction",
                "masked generative pretraining",
                "production trainer result",
                "validation or final-test result",
                "per-class F1 greater than 0.98",
            ],
            "candidate_metrics_observed_before_lock": False,
        },
        "external_sources": {
            "livt_cvpr_2023": {
                "paper_url": (
                    "https://openaccess.thecvf.com/content/CVPR2023/html/"
                    "Xu_Learning_Imbalanced_Data_With_Vision_Transformers_"
                    "CVPR_2023_paper.html"
                ),
                "paper": _immutable(LIVT_PAPER),
                "official_repository_url": (
                    "https://github.com/XuZhengzhuo/LiVT"
                ),
                "official_repository": str(LIVT_REPOSITORY),
                "repository_commit": EXPECTED_LIVT_COMMIT,
                "repository_tree": EXPECTED_LIVT_TREE,
                "license": "MIT",
                "license_file": _immutable(LIVT_REPOSITORY / "LICENSE"),
                "loss_source": _immutable(
                    LIVT_REPOSITORY / "util" / "loss.py"
                ),
            },
            "implementation_policy": (
                "Independently implement and oracle-check the pinned equation; "
                "do not import LiVT or install its historical runtime."
            ),
        },
        "immutable_inputs": {
            "train_embedding_cache": _immutable(EMBEDDING_CACHE),
            "cidt_predictions": _immutable(CIDT_PREDICTIONS),
            "keeper_checkpoint": _immutable(KEEPER_CHECKPOINT),
            "keeper_resolved_config": _immutable(KEEPER_CONFIG),
            "yolo_data_yaml": _immutable(YOLO_DATA_YAML),
            "current_best_commands": _immutable(CURRENT_BEST_COMMANDS),
            "current_best_history": _immutable(CURRENT_BEST_HISTORY),
        },
        "protected_untracked": [
            "BaoCao/",
            "deep-research-report (9).md",
            "deep-research-report (10).md",
        ],
        "dataset_declaration": {
            "rows": ROWS,
            "embedding_dim": EMBEDDING_DIM,
            "classes": CLASSES,
            "class_counts": class_counts.tolist(),
            "outer_fold_counts": np.bincount(
                folds, minlength=OUTER_FOLDS
            ).tolist(),
            "outer_assignment_dtype": "int64",
            "outer_assignment_sha256": _numeric_array_sha256(folds),
            "source_sequence_sha256": _string_sequence_sha256(sources),
            "source_group_count": len(source_to_fold),
            "outer_source_overlap": 0,
            "cache_split_allowlist": ["train"],
            "cache_split_denylist": ["val", "valid", "validation", "test"],
            "raw_dataset_mutation_allowed": False,
            "image_augmentation_used": False,
            "synthetic_data_used": False,
        },
        "data_access_audit": {
            "hook": (
                "sys.addaudithook open-event ledger installed before first "
                "candidate input"
            ),
            "installation_probe": (
                "trkh.balanced_bce.data_access_probe must be observed before "
                "candidate input"
            ),
            "allowed_inputs": (
                "exact immutable inputs, pinned LiVT evidence, protocol, lock, "
                "and fixed train-image anchors only"
            ),
            "active_output_excluded_from_input_allowlist": True,
            "all_data_domain_writes_forbidden": True,
            "forbidden_complete_components": [
                "val",
                "valid",
                "validation",
                "test",
            ],
            "block_before_open": True,
            "ordered_ledger_sha256_required": True,
            "blocked_attempt_count_max": 0,
            "validation_open_count_max": 0,
            "test_open_count_max": 0,
        },
        "outer_folds": folds_payload,
        "visual_anchors": {
            "selection_is_metric_independent": True,
            "ordered_sample_indices": VISUAL_SAMPLE_INDICES.tolist(),
            "ordered_indices_sha256": _numeric_array_sha256(
                VISUAL_SAMPLE_INDICES
            ),
            "folds_sha256": _numeric_array_sha256(visual_folds),
            "rows": int(VISUAL_SAMPLE_INDICES.size),
            "pixels_opened_only_after_probabilities_and_actions_frozen": True,
            "split": "train",
        },
        "equation": {
            "class_count": CLASSES,
            "ce_control": "cross_entropy(raw_logits, hard_target)",
            "plain_bce_control": (
                "5 * binary_cross_entropy_with_logits(raw_logits, one_hot, "
                "reduction=mean)"
            ),
            "balanced_bce_candidate": {
                "fit_prior": "fit_class_count / fit_row_count",
                "bias": "log(pi) - log(1-pi)",
                "tau": 1.0,
                "loss": (
                    "5 * binary_cross_entropy_with_logits(raw_logits + bias, "
                    "one_hot, reduction=mean)"
                ),
                "inference": "argmax(raw_logits)",
            },
            "balanced_softmax_tau025_control": {
                "tau": 0.25,
                "loss": (
                    "cross_entropy(raw_logits + 0.25*log(pi), hard_target)"
                ),
                "inference": "argmax(raw_logits)",
            },
            "reversed_prior_bce_control": {
                "applied_bias": "-balanced_bce_bias",
                "loss": (
                    "5 * binary_cross_entropy_with_logits(raw_logits - "
                    "balanced_bce_bias, one_hot, reduction=mean)"
                ),
                "inference": "argmax(raw_logits)",
            },
            "label_smoothing": 0.0,
            "pos_weight": None,
            "class_weights": None,
            "test_prior_term": False,
            "threshold_calibration": False,
        },
        "roles": [
            "ce_control",
            "plain_bce_control",
            "balanced_bce_candidate",
            "balanced_bce_seed_repeat",
            "balanced_softmax_tau025_control",
            "reversed_prior_bce_control",
        ],
        "optimization": {
            "device": "cuda",
            "dtype": "float32",
            "deterministic_algorithms": True,
            "cublas_workspace_config": ":4096:8",
            "cuda_matmul_tf32": False,
            "cudnn_tf32": False,
            "seed": PRIMARY_SEED,
            "repeat_seed_offset": REPEAT_SEED_OFFSET,
            "head": "torch.nn.Linear(256,5,bias=True)",
            "epochs": EPOCHS,
            "batch_size": BATCH_SIZE,
            "drop_last": False,
            "optimizer": {
                "name": "SGD",
                "initial_lr": 0.03,
                "momentum": 0.9,
                "dampening": 0.0,
                "weight_decay": 0.0,
                "nesterov": False,
            },
            "lr_schedule": {
                "warmup_epochs": 5,
                "post_warmup": "cosine",
                "horizon_epochs": 30,
            },
            "order_rng": "numpy.default_rng",
            "order_seed": "20260724 + outer_fold",
            "repeat_order_seed": "20260724 + 100000 + outer_fold",
            "every_fit_row_once_per_epoch": True,
            "score_epoch": 30,
            "early_stopping": False,
            "best_epoch_selection": False,
            "feature_update": False,
            "feature_standardization": False,
            "global_class1_oversampling": False,
        },
        "metric_definition": {
            "prediction": "argmax(raw_logits)",
            "threshold_calibration": False,
            "ece_bins": 15,
            "ece_binning": "equal_width",
            "restricted_fp_true_classes": [0, 2, 4],
            "oof_rows_per_role": ROWS,
            "trace_epochs": [1, 5, 10, 20, 30],
        },
        "gates": {
            "resources": {
                "start_available_physical_ram_gib_min": 4.0,
                "peak_process_rss_gib_max": 3.0,
                "peak_cuda_allocated_gib_max": 1.0,
                "formal_elapsed_minutes_max": 15.0,
                "unrelated_python_or_tensorrt_allowed": False,
            },
            "control_compatibility": {
                "ce_macro_f1_min": 0.935,
                "ce_class1_f1_min": 0.8,
            },
            "candidate_vs_ce": {
                "macro_f1_gain_min": 0.001,
                "class1_precision_gain_min": 0.015,
                "class1_f1_gain_min": 0.005,
                "class1_recall_loss_max": 0.01,
                "class1_tp_net_min": -3,
                "restricted_fp_net_removal_min": 10,
                "corrections_ge_harms": True,
                "maximum_nonfocus_f1_loss": 0.005,
                "nll_increase_max": 0.01,
                "ece_increase_max": 0.01,
            },
            "candidate_vs_plain_bce": {
                "macro_f1_loss_max": 0.001,
                "class1_precision_gain_min": 0.005,
                "class1_f1_gain_min": 0.005,
                "class1_tp_net_min": -2,
                "restricted_fp_net_removal_min": 5,
            },
            "candidate_vs_controls": {
                "macro_f1_gain_vs_balanced_softmax_min": 0.01,
                "class1_f1_gain_vs_balanced_softmax_min": 0.01,
                "macro_f1_gain_vs_reversed_prior_min": 0.005,
                "class1_f1_gain_vs_reversed_prior_min": 0.01,
                "mean_raw_class1_logit_delta_vs_plain_abs_min": 0.1,
                "prediction_array_must_differ_from_reversed": True,
            },
            "fold_stability": {
                "precision_nonworse_fold_min": 4,
                "class1_f1_improved_fold_min": 4,
                "per_fold_recall_loss_max": 0.03,
                "per_fold_class1_tp_net_min": -2,
                "per_fold_macro_f1_loss_max": 0.01,
            },
            "repeat": {
                "must_pass_primary_safety": True,
                "prediction_agreement_min": 0.97,
                "macro_f1_absolute_difference_max": 0.01,
                "class1_f1_absolute_difference_max": 0.01,
                "class1_precision_absolute_difference_max": 0.02,
                "class1_recall_absolute_difference_max": 0.02,
            },
            "data_access": {
                "dynamic_hook_installed": True,
                "dynamic_hook_probe_seen": True,
                "blocked_attempt_count_max": 0,
                "validation_open_count_max": 0,
                "test_open_count_max": 0,
                "replay_contract_exact": True,
            },
            "replay": {
                "logit_probability_state_max_error": 1e-7,
                "nested_metric_max_error": 1e-10,
                "integer_actions_exact": True,
                "visual_arrays_exact": True,
            },
            "manual": {
                "fixed_visual_rows": 20,
                "explicit_accept_required": True,
                "non_train_source_count_max": 0,
            },
            "logic": (
                "All structural, compatibility, performance, fold, repeat, "
                "mechanism, resource, replay, and manual gates are conjunctive."
            ),
        },
        "formal_artifacts": [
            "summary.json",
            "fold_protocol.json",
            "fold_metrics.csv",
            "oof_probabilities.npz",
            "oof_rows.csv",
            "role_states.npz",
            "training_trace.jsonl",
            "data_access_ledger.json",
            "fixed_visual_arrays.npz",
            "fixed_visual_metadata.json",
            "fixed_visual_contact_sheet.png",
            "artifact_manifest.json",
            "replay.json",
            "formal_visual_review.json",
            "final_decision.json",
            "final_manifest.json",
        ],
        "manual_review_scope": {
            "class1_tp_breaks_visually_defensible": True,
            "restricted_fp_removals_preserve_class_semantics": True,
            "new_class1_predictions_not_contradicted_by_visible_state": True,
            "pixel_level_xai_claim": False,
            "production_smoke_requires_full_xai_stack": True,
        },
        "failure_closure": {
            "forbid_sweeps": [
                "tau",
                "prior clipping",
                "label smoothing",
                "pos_weight",
                "class weights",
                "threshold",
                "epoch",
                "seed",
                "batch size",
                "learning rate",
                "nearby loss mixtures",
            ],
            "validation_test_smoke_probe_full_train_forbidden": True,
            "trainer_integration_forbidden": True,
            "current_best_update_forbidden": True,
        },
        "pass_policy": {
            "next_stage": (
                "default-off production Bal-BCE loss plus separately locked "
                "matched CE-versus-Bal-BCE validation smoke with full audits "
                "and XAI"
            ),
            "test_allowed": False,
            "probe_allowed": False,
            "full_train_allowed": False,
            "current_best_update_allowed": False,
        },
    }


def _serialize(lock: Mapping[str, object]) -> str:
    return json.dumps(lock, indent=2, ensure_ascii=False) + "\n"


def write_lock(lock: Mapping[str, object]) -> None:
    LOCK_PATH.write_text(_serialize(lock), encoding="utf-8")
    lock_sha = _sha256(LOCK_PATH)
    LOCK_SHA_PATH.write_text(
        f"{lock_sha}  {LOCK_PATH.name}\n",
        encoding="ascii",
    )
    print(f"Wrote {LOCK_PATH}")
    print(f"SHA256 {lock_sha}")


def check_lock(lock: Mapping[str, object]) -> None:
    expected = _serialize(lock)
    actual = LOCK_PATH.read_text(encoding="utf-8")
    if actual != expected:
        raise ValueError(f"Lock differs from prospective inputs: {LOCK_PATH}")
    expected_sha_line = f"{_sha256(LOCK_PATH)}  {LOCK_PATH.name}\n"
    actual_sha_line = LOCK_SHA_PATH.read_text(encoding="ascii")
    if actual_sha_line != expected_sha_line:
        raise ValueError(f"Lock SHA sidecar differs: {LOCK_SHA_PATH}")
    print(f"Lock check passed: {_sha256(LOCK_PATH)}")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Build or check the prospective TRKH Bal-BCE A0 lock."
    )
    parser.add_argument(
        "--check",
        action="store_true",
        help="Fail if the existing lock and SHA sidecar differ.",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    lock = build_lock()
    if args.check:
        check_lock(lock)
    else:
        write_lock(lock)


if __name__ == "__main__":
    main()
