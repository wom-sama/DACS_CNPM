from __future__ import annotations

import argparse
import csv
import hashlib
import json
import math
import os
from pathlib import Path
import random
import subprocess
import time
from typing import Dict, List, Mapping, Optional, Sequence, Tuple
import warnings

import numpy as np
from PIL import Image, ImageDraw, ImageFont

os.environ.setdefault("CUBLAS_WORKSPACE_CONFIG", ":4096:8")

import sklearn
from sklearn.decomposition import non_negative_factorization
from sklearn.exceptions import ConvergenceWarning
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import roc_auc_score
from sklearn.preprocessing import StandardScaler
import torch
from torch import Tensor, nn
from torch.utils.data import DataLoader

from trkh.core.config import load_data_spec, to_serializable
from trkh.core.utils import build_safe_dataloader_kwargs, set_seed
from trkh.data.dataset import MangoYOLOCropDataset
from trkh.evaluation.evaluate import resolve_crop_to_primary_object
from trkh.evaluation.robustness_eval import IdentityCorruption, LightingShift
from trkh.inference.inference import load_model
from trkh.models.model import classification_logits_from_features
from trkh.tools.audit_counterfactual_illumination_disagreement_readiness import (
    _SelectedConditionDataset,
    _build_eval_transform,
)
from trkh.tools.audit_more_model_rebalancing_readiness import (
    CleanTrainRow,
    _read_clean_train_rows,
)
from trkh.tools.build_precision_ensemble_checkpoint import _eval_semantics


METHOD = "hamburger_nmf_surface_a0"
MODE = "hamburger_nmf_surface_a0_train_information_gate"
SEED = 20260720
BATCH_SIZE = 64
NUM_WORKERS = 2
FOCUS_CLASS = 1
RESTRICTED_NEGATIVE_CLASSES = (0, 2, 4)
FOLDS = (0, 1, 2, 3, 4)
ROLE_NAMES = (
    "base_only",
    "raw_surface_control",
    "svd_rank8_control",
    "nmf_candidate",
    "nmf_seed_repeat",
    "source_deranged",
)
CONTROL_ROLES = ("base_only", "raw_surface_control", "svd_rank8_control")
PROJECTION_INPUT_DIM = 256
PROJECTION_DIM = 32
NMF_RANK = 8
NMF_STEPS = 6
NMF_EPSILON = 1e-6
MIN_SURFACE_TOKENS = 16
MAX_PATCHES = 256
BBOX_PRIOR_THRESHOLD = 0.5
BBOX_PRIOR_MARGIN_RATIO = 0.04
SEED_REPEAT_OFFSET = 7919
BASE_DIM = 6
SURFACE_DESCRIPTOR_DIM = 284
ROLE_DIM = BASE_DIM + SURFACE_DESCRIPTOR_DIM
LOG_PROBABILITY_CLIP = 1e-8
READOUT_C = 0.1
READOUT_MAX_ITER = 4000
READOUT_TOLERANCE = 1e-9
MIN_FIT_TP_RETENTION = 0.97
MAX_CIDT_PROBABILITY_ERROR = 3e-5
MAX_PEAK_CUDA_GIB = 3.5
EXPECTED_COHORT_ROWS = 750
EXPECTED_POSITIVES = 528
EXPECTED_NEGATIVES = 222
EXPECTED_FOLD_COUNTS = {
    0: {"tp": 107, "fp": 36, "target_0": 23, "target_2": 12, "target_4": 1},
    1: {"tp": 112, "fp": 45, "target_0": 36, "target_2": 9, "target_4": 0},
    2: {"tp": 100, "fp": 48, "target_0": 37, "target_2": 8, "target_4": 3},
    3: {"tp": 101, "fp": 52, "target_0": 36, "target_2": 13, "target_4": 3},
    4: {"tp": 108, "fp": 41, "target_0": 26, "target_2": 12, "target_4": 3},
}
EXPECTED_ORDERED_INDEX_SHA256 = (
    "55913ec45265b156a611dc96c779e46b08f28582737af8269105afe6f65694d7"
)
LOCKED_PROJECTION_SHA256 = (
    "b93eafaee2cd389e5483c42a9db048f7734905a4df94f35353fc859ebb11a42a"
)
LOCKED_KEEPER_SHA256 = "1f49d577240c69dc63c30af70db52ec2aa9da65a17aef1c4b1c09ece6c482677"
LOCKED_LAUNCHER_ARGS_SHA256 = "908a05cf66b2a01162cae62e4ff2251eaae1297d31e70510144e4954159b7eff"
LOCKED_RESOLVED_CONFIG_SHA256 = "e9c4f48917e333d2f34f61806bb54041b35f2217ebb23afb3bd0ced969854674"
LOCKED_DATA_SHA256 = "716e33df24c63a9e9920f97b685199707fb84ab4c7154544f5dd9a3e00d884ef"
LOCKED_CIDT_SUMMARY_SHA256 = "d4891edf2963ab12385b7ce5bdc812ec3e19c5c098acd25c66eb557af541d7ad"
LOCKED_CIDT_PREDICTIONS_SHA256 = "2e0993752d58d99ea429bfefe1e2bfe6fa949e45aea1a26cc4bdfee97d4db21c"
LOCKED_PROTOCOL_SHA256 = "0bc33ac9c7934e0b8bb274f79a4e3818a1219538f61d11abc28e71a4d36d7996"
LOCKED_CURRENT_COMMAND_SHA256 = "36b9aa1a21b765829acf4c8321be147bd76297de4ccdb8a40e6dee8e37940faf"
LOCKED_COMMAND_HISTORY_SHA256 = "39bd2879ce66fddf36a953021ea1e40f8d9de6cb4334b9b825011b2b8dc98f53"
LOCKED_PAPER_SHA256 = "4eed8898973d9aa9a1101e32d436c95e57f0f61c5ef459b46f673e33666ae9f1"
LOCKED_OFFICIAL_COMMIT = "d9b51f6f197486df68c6e059e396520680157c08"
LOCKED_OFFICIAL_TREE = "a399506b5556b8d243a80b2cfeee47ba64b02c2e"
LOCKED_OFFICIAL_LICENSE_SHA256 = "230184f60bae2feaf244f10a8bac053c8ff33a183bcc365b4d8b876d2b7f4809"
LOCKED_OFFICIAL_SOURCE_SHA256 = "c6a261aa8fd7f932246b6e57addde4c3e971dbaa1f6a42e9a70bcbebeecc2829"
LOCKED_SKLEARN_VERSION = "1.6.1"
LOCKED_SKLEARN_LICENSE_SHA256 = "1b74e02d0cb8e6502091124787fc91695b9dd92c9cb146d9d34830f9a300ab3c"
LOCKED_SKLEARN_SOURCE_SHA256 = "6f0ed846e1531f773eb82de20dfa4f07e57f53fa472f3f8c576e8ee3d7007369"

CONDITIONS = (
    ("lighting_dim", 0.70, 0.90),
    ("lighting_bright", 1.25, 1.10),
    ("low_contrast", 1.00, 0.65),
)

REPO_ROOT = Path(__file__).resolve().parents[2]
KEEPER_ROOT = (
    REPO_ROOT
    / "runs"
    / "probe_v8_yolof_pairroute_teacherfocusbinary015_boundarydrop_bboxprior_120b_2e_20260701"
)
OFFICIAL_ROOT = Path(r"D:\DataAI\external_sources\official\hamburger-iclr2021")
PAPER_PATH = Path(r"D:\DataAI\external_sources\papers\hamburger_iclr2021.pdf")
SKLEARN_ROOT = Path(sklearn.__file__).resolve().parent
SKLEARN_SOURCE_PATH = SKLEARN_ROOT / "decomposition" / "_nmf.py"
SKLEARN_LICENSE_PATH = (
    SKLEARN_ROOT.parent / f"scikit_learn-{LOCKED_SKLEARN_VERSION}.dist-info" / "COPYING"
)
PROTECTED_UNTRACKED_FILES: Tuple[Tuple[str, str], ...] = (
    (
        "BaoCao/GT.md",
        "c784717657d1dda168de2e70f08256d41397c5d7c0b8a916f4b0367870c7b840",
    ),
    (
        "BaoCao/mango_cls_256_merge01_4cls_architecture.dot",
        "c9aebcb97275a4dbb069e866df71d0022d9c8b6e9f9caa4e55d0f991ce528a9d",
    ),
    (
        "BaoCao/mango_cls_256_merge01_4cls_architecture.png",
        "93d6f9c27b0b77bdf5113cede18d828f9a50949e5f324bba6b990f3cf204832d",
    ),
    (
        "BaoCao/mango_cls_256_merge01_4cls_architecture.svg",
        "2ead05f8e42f41d256bc5b7df21df4f37b181dd2a54d89cdc8098ea074453d05",
    ),
    (
        "BaoCao/mango_cls_256_merge01_4cls_architecture_summary.md",
        "d43b82beb7910f268793f38a689b74625df7764a92b85402d5fa13f8fcf0e493",
    ),
    (
        "deep-research-report (9).md",
        "7e52cc8e2bed2e48dafc68b25dc2536318a895c866f4aa280c4fe2e817546f8a",
    ),
    (
        "deep-research-report (10).md",
        "37f74b5e97b6160e657237844d66cedc13a98808b476b8328a58a52eb7c1522a",
    ),
)


def parse_args(argv: Optional[Sequence[str]] = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Locked train-only Hamburger-NMF surface information gate."
    )
    parser.add_argument("--checkpoint", type=Path, default=KEEPER_ROOT / "checkpoints" / "best.pt")
    parser.add_argument("--launcher-args", type=Path, default=KEEPER_ROOT / "launcher_args.json")
    parser.add_argument("--resolved-config", type=Path, default=KEEPER_ROOT / "resolved_config.json")
    parser.add_argument("--data", type=Path, default=Path(r"D:\DataAI\AIEx\newdataset\yolo_f\data.yaml"))
    parser.add_argument(
        "--cidt-summary",
        type=Path,
        default=REPO_ROOT / "runs" / "audit_cidt_readiness_full_train_20260714" / "summary.json",
    )
    parser.add_argument(
        "--cidt-predictions",
        type=Path,
        default=REPO_ROOT
        / "runs"
        / "audit_cidt_readiness_full_train_20260714"
        / "predictions_all_conditions.csv",
    )
    parser.add_argument(
        "--protocol",
        type=Path,
        default=REPO_ROOT
        / "docs"
        / "TRKH_5CLASS_HAMBURGER_NMF_SURFACE_A0_PROTOCOL_20260720.md",
    )
    parser.add_argument("--official-root", type=Path, default=OFFICIAL_ROOT)
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=REPO_ROOT / "runs" / "audit_hamburger_nmf_surface_a0_20260720",
    )
    parser.add_argument("--batch-size", type=int, default=BATCH_SIZE)
    parser.add_argument("--num-workers", type=int, default=NUM_WORKERS)
    parser.add_argument("--seed", type=int, default=SEED)
    parser.add_argument("--device", choices=("cuda",), default="cuda")
    parser.add_argument("--preflight-only", action="store_true")
    parser.add_argument("--engineering-forward", action="store_true")
    parser.add_argument("--replay-summary", type=Path)
    parser.add_argument("--finalize-visual-review", choices=("pass", "fail"))
    parser.add_argument("--expected-summary-sha256")
    return parser.parse_args(argv)


def _validate_locked_args(args: argparse.Namespace) -> None:
    if str(args.device) != "cuda":
        raise ValueError("Hamburger-NMF A0 is locked to CUDA")
    if int(args.batch_size) != BATCH_SIZE or int(args.num_workers) != NUM_WORKERS:
        raise ValueError(
            f"Hamburger-NMF A0 locks batch-size={BATCH_SIZE}, num-workers={NUM_WORKERS}"
        )
    if int(args.seed) != SEED:
        raise ValueError(f"Hamburger-NMF A0 locks seed={SEED}")


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with Path(path).open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _array_sha256(array: np.ndarray) -> str:
    value = np.ascontiguousarray(array)
    digest = hashlib.sha256()
    digest.update(str(value.dtype).encode("ascii"))
    digest.update(np.asarray(value.shape, dtype=np.int64).tobytes())
    digest.update(value.tobytes())
    return digest.hexdigest()


def _records_sha256(rows: Sequence[Mapping[str, object]]) -> str:
    payload = json.dumps(
        [to_serializable(dict(row)) for row in rows],
        sort_keys=True,
        separators=(",", ":"),
    ).encode("utf-8")
    return hashlib.sha256(payload).hexdigest()


def _cohort_index_sha256(indices: Sequence[int]) -> str:
    payload = ",".join(str(int(index)) for index in indices).encode("ascii")
    return hashlib.sha256(payload).hexdigest()


def _verify_hash(path: Path, expected: str, label: str) -> Dict[str, object]:
    resolved = Path(path).expanduser().resolve()
    if not resolved.is_file():
        raise FileNotFoundError(f"Locked {label} is missing: {resolved}")
    observed = _sha256(resolved)
    if observed != str(expected).strip().casefold():
        raise ValueError(
            f"Locked {label} SHA-256 mismatch: expected={expected}, observed={observed}"
        )
    return {
        "path": str(resolved),
        "bytes": int(resolved.stat().st_size),
        "sha256": observed,
    }


def _git_value(repo: Path, *arguments: str) -> str:
    return subprocess.check_output(
        ["git", "-C", str(Path(repo).resolve()), *arguments],
        text=True,
        encoding="utf-8",
    ).strip()


def _repo_state() -> Dict[str, object]:
    head = _git_value(REPO_ROOT, "rev-parse", "HEAD")
    upstream = _git_value(REPO_ROOT, "rev-parse", "@{upstream}")
    counts = _git_value(
        REPO_ROOT, "rev-list", "--left-right", "--count", "HEAD...@{upstream}"
    ).split()
    tracked = _git_value(REPO_ROOT, "status", "--short", "--untracked-files=no")
    full = _git_value(REPO_ROOT, "status", "--short")
    return {
        "branch": _git_value(REPO_ROOT, "branch", "--show-current"),
        "head": head,
        "upstream": upstream,
        "ahead": int(counts[0]),
        "behind": int(counts[1]),
        "tracked_worktree_clean": tracked == "",
        "head_matches_upstream": head == upstream,
        "full_status": full.splitlines() if full else [],
    }


def _protected_untracked_state() -> Dict[str, object]:
    expected = dict(PROTECTED_UNTRACKED_FILES)
    observed = {
        path.relative_to(REPO_ROOT).as_posix()
        for path in (REPO_ROOT / "BaoCao").rglob("*")
        if path.is_file()
    }
    observed.update(
        relative
        for relative in ("deep-research-report (9).md", "deep-research-report (10).md")
        if (REPO_ROOT / relative).is_file()
    )
    raw = subprocess.check_output(
        [
            "git",
            "-C",
            str(REPO_ROOT),
            "ls-files",
            "--others",
            "--exclude-standard",
            "-z",
        ]
    )
    untracked = {value.decode("utf-8") for value in raw.split(b"\0") if value}
    hashes = {
        relative: _sha256(REPO_ROOT / relative)
        for relative in sorted(expected)
        if (REPO_ROOT / relative).is_file()
    }
    checks = {
        "file_set_exact": observed == set(expected),
        "all_remain_untracked": set(expected).issubset(untracked),
        "hashes_exact": hashes == expected,
    }
    return {
        "checks": checks,
        "passed": all(checks.values()),
        "files": [
            {
                "path": relative,
                "bytes": int((REPO_ROOT / relative).stat().st_size),
                "sha256": hashes[relative],
            }
            for relative in sorted(hashes)
        ],
        "missing_or_extra": sorted(set(expected).symmetric_difference(observed)),
    }


def verify_locked_inputs(args: argparse.Namespace) -> Dict[str, object]:
    files = {
        "keeper": _verify_hash(args.checkpoint, LOCKED_KEEPER_SHA256, "keeper"),
        "keeper_launcher_args": _verify_hash(
            args.launcher_args, LOCKED_LAUNCHER_ARGS_SHA256, "keeper launcher args"
        ),
        "keeper_resolved_config": _verify_hash(
            args.resolved_config, LOCKED_RESOLVED_CONFIG_SHA256, "keeper resolved config"
        ),
        "data_yaml": _verify_hash(args.data, LOCKED_DATA_SHA256, "data YAML"),
        "cidt_summary": _verify_hash(
            args.cidt_summary, LOCKED_CIDT_SUMMARY_SHA256, "CIDT summary"
        ),
        "cidt_predictions": _verify_hash(
            args.cidt_predictions, LOCKED_CIDT_PREDICTIONS_SHA256, "CIDT predictions"
        ),
        "protocol": _verify_hash(args.protocol, LOCKED_PROTOCOL_SHA256, "NMF protocol"),
        "current_best_commands": _verify_hash(
            REPO_ROOT / "docs" / "TRKH_CURRENT_BEST_FULL_TRAIN_COMMANDS_20260706.txt",
            LOCKED_CURRENT_COMMAND_SHA256,
            "current-best commands",
        ),
        "current_best_history": _verify_hash(
            REPO_ROOT / "docs" / "TRKH_CURRENT_BEST_COMMAND_UPDATE_HISTORY.txt",
            LOCKED_COMMAND_HISTORY_SHA256,
            "current-best command history",
        ),
        "paper": _verify_hash(PAPER_PATH, LOCKED_PAPER_SHA256, "Hamburger paper"),
        "official_license": _verify_hash(
            Path(args.official_root) / "LICENSE",
            LOCKED_OFFICIAL_LICENSE_SHA256,
            "official Hamburger license",
        ),
        "official_source": _verify_hash(
            Path(args.official_root) / "seg" / "HamNet" / "hamburger" / "ham.py",
            LOCKED_OFFICIAL_SOURCE_SHA256,
            "official Hamburger NMF source",
        ),
        "sklearn_license": _verify_hash(
            SKLEARN_LICENSE_PATH, LOCKED_SKLEARN_LICENSE_SHA256, "scikit-learn license"
        ),
        "sklearn_source": _verify_hash(
            SKLEARN_SOURCE_PATH, LOCKED_SKLEARN_SOURCE_SHA256, "scikit-learn NMF source"
        ),
    }
    official_root = Path(args.official_root).expanduser().resolve()
    commit = _git_value(official_root, "rev-parse", "HEAD")
    tree = _git_value(official_root, "rev-parse", "HEAD^{tree}")
    status = _git_value(official_root, "status", "--porcelain")
    if commit != LOCKED_OFFICIAL_COMMIT or tree != LOCKED_OFFICIAL_TREE or status:
        raise ValueError(
            "Official Hamburger repository differs from lock: "
            f"commit={commit}, tree={tree}, status={status!r}"
        )
    if sklearn.__version__ != LOCKED_SKLEARN_VERSION:
        raise ValueError(
            f"Installed scikit-learn differs: {sklearn.__version__} != {LOCKED_SKLEARN_VERSION}"
        )
    protected = _protected_untracked_state()
    if not bool(protected["passed"]):
        raise ValueError(f"Protected user payloads differ from lock: {protected}")
    return {
        "files": files,
        "official_repository": {
            "path": str(official_root),
            "remote": _git_value(official_root, "remote", "get-url", "origin"),
            "commit": commit,
            "tree": tree,
            "status": status,
            "worktree_clean": status == "",
        },
        "installed_runtime": {
            "torch": torch.__version__,
            "numpy": np.__version__,
            "sklearn": sklearn.__version__,
        },
        "protected_untracked": protected,
    }


def _model_state_sha256(model: nn.Module) -> str:
    digest = hashlib.sha256()
    for name, tensor in sorted(model.state_dict().items()):
        value = tensor.detach().cpu().contiguous()
        digest.update(name.encode("utf-8"))
        digest.update(str(value.dtype).encode("ascii"))
        digest.update(np.asarray(value.shape, dtype=np.int64).tobytes())
        digest.update(value.reshape(-1).view(torch.uint8).numpy().tobytes())
    return digest.hexdigest()


def _cohort_from_rows(rows: Sequence[CleanTrainRow]) -> List[CleanTrainRow]:
    cohort = [
        row
        for row in rows
        if row.keeper_prediction == FOCUS_CLASS
        and (row.target == FOCUS_CLASS or row.target in RESTRICTED_NEGATIVE_CLASSES)
    ]
    positives = sum(row.target == FOCUS_CLASS for row in cohort)
    negatives = len(cohort) - positives
    if (len(cohort), positives, negatives) != (
        EXPECTED_COHORT_ROWS,
        EXPECTED_POSITIVES,
        EXPECTED_NEGATIVES,
    ):
        raise ValueError(
            "Hamburger-NMF cohort differs: "
            f"rows/TP/FP={(len(cohort), positives, negatives)}"
        )
    observed: Dict[int, Dict[str, int]] = {}
    for fold in FOLDS:
        selected = [row for row in cohort if row.fold == fold]
        observed[fold] = {
            "tp": sum(row.target == FOCUS_CLASS for row in selected),
            "fp": sum(row.target != FOCUS_CLASS for row in selected),
            "target_0": sum(row.target == 0 for row in selected),
            "target_2": sum(row.target == 2 for row in selected),
            "target_4": sum(row.target == 4 for row in selected),
        }
    if observed != EXPECTED_FOLD_COUNTS:
        raise ValueError(f"Hamburger-NMF cohort fold counts differ: {observed}")
    ordered_hash = _cohort_index_sha256([row.sample_index for row in cohort])
    if ordered_hash != EXPECTED_ORDERED_INDEX_SHA256:
        raise ValueError(f"Hamburger-NMF cohort order hash differs: {ordered_hash}")
    source_folds: Dict[str, set[int]] = {}
    for row in cohort:
        source_folds.setdefault(row.source_stem, set()).add(int(row.fold))
    overlap = {
        source: sorted(values) for source, values in source_folds.items() if len(values) > 1
    }
    if overlap:
        raise ValueError(f"Hamburger-NMF cohort has cross-fold source overlap: {overlap}")
    return cohort


def _build_dataset(
    checkpoint: Mapping[str, object],
    rows: Sequence[CleanTrainRow],
    data_path: Path,
) -> Tuple[MangoYOLOCropDataset, object, Dict[str, object]]:
    semantics = _eval_semantics(checkpoint)
    if int(semantics["temporal_frames"]) != 1:
        raise ValueError("Hamburger-NMF A0 supports temporal_frames=1 only")
    data_spec = load_data_spec(data_path, class_name_mode="raw", expected_num_classes=5)
    class_names = [str(value) for value in checkpoint.get("class_names", [])]
    if list(data_spec.class_names) != class_names:
        raise ValueError("Dataset class order differs from keeper checkpoint")
    dataset = MangoYOLOCropDataset.from_data_spec(
        data_spec=data_spec,
        split="train",
        transform=None,
        crop_margin_ratio=float(semantics["crop_margin_ratio"]),
        crop_to_primary_object=resolve_crop_to_primary_object(checkpoint),
        classification_target=True,
        classification_object_crops=True,
        classification_bbox_metadata=True,
    )
    paths = [Path(value).resolve() for value in dataset.sample_paths()]
    paths_exact = len(paths) == len(rows) and all(
        expected.image_path == observed for expected, observed in zip(rows, paths)
    )
    train_only = all(
        "train" in {part.casefold() for part in path.parts}
        and "val" not in {part.casefold() for part in path.parts}
        and "test" not in {part.casefold() for part in path.parts}
        for path in paths
    )
    if not paths_exact or not train_only:
        raise ValueError("Dataset mapping violates the locked train-only declaration")
    return dataset, _build_eval_transform(semantics), {
        "rows": len(paths),
        "paths_exact": paths_exact,
        "train_paths_only": train_only,
        "semantics": dict(semantics),
    }


def _make_condition_loader(
    *,
    base_dataset: MangoYOLOCropDataset,
    transform,
    indices: Sequence[int],
    brightness: float,
    contrast: float,
    batch_size: int,
    num_workers: int,
    context: str,
) -> Tuple[DataLoader, Dict[str, object]]:
    corruption = (
        IdentityCorruption()
        if math.isclose(brightness, 1.0) and math.isclose(contrast, 1.0)
        else LightingShift(brightness=brightness, contrast=contrast)
    )
    dataset = _SelectedConditionDataset(
        base_dataset, indices, corruption=corruption, transform=transform
    )
    kwargs, summary = build_safe_dataloader_kwargs(
        requested_num_workers=int(num_workers),
        requested_pin_memory=True,
        context=context,
        prefetch_factor=2,
        persistent_workers=True,
    )
    generator = torch.Generator().manual_seed(SEED)
    return DataLoader(
        dataset,
        batch_size=int(batch_size),
        shuffle=False,
        drop_last=False,
        generator=generator,
        **kwargs,
    ), summary


def fixed_projection() -> np.ndarray:
    generator = np.random.default_rng(SEED)
    matrix = generator.standard_normal(
        (PROJECTION_INPUT_DIM, PROJECTION_DIM), dtype=np.float64
    )
    projection, upper = np.linalg.qr(matrix, mode="reduced")
    signs = np.where(np.diag(upper) < 0.0, -1.0, 1.0)
    projection = (projection * signs[None, :]).astype(np.float32)
    if _array_sha256(projection) != LOCKED_PROJECTION_SHA256:
        raise RuntimeError("Fixed projection differs from prospective lock")
    return projection


def projection_diagnostics(projection: np.ndarray) -> Dict[str, object]:
    value = np.asarray(projection, dtype=np.float32)
    error = float(
        np.max(
            np.abs(
                value.T @ value - np.eye(PROJECTION_DIM, dtype=np.float32)
            )
        )
    )
    return {
        "shape": [int(item) for item in value.shape],
        "sha256": _array_sha256(value),
        "maximum_orthogonality_error": error,
        "passed": value.shape == (PROJECTION_INPUT_DIM, PROJECTION_DIM)
        and _array_sha256(value) == LOCKED_PROJECTION_SHA256
        and error <= 1e-6,
    }


def project_surface_tokens(tokens: np.ndarray, projection: np.ndarray) -> np.ndarray:
    value = np.asarray(tokens, dtype=np.float32)
    matrix = np.asarray(projection, dtype=np.float32)
    if value.ndim != 2 or value.shape[1] != PROJECTION_INPUT_DIM:
        raise ValueError("Surface tokens must have shape [N,256]")
    if value.shape[0] < MIN_SURFACE_TOKENS:
        raise ValueError("Surface token count is below the locked minimum")
    projected = np.maximum(
        value.astype(np.float64) @ matrix.astype(np.float64), 0.0
    )
    rms = np.sqrt(np.mean(np.square(projected), axis=0, keepdims=True))
    projected = projected / (rms + NMF_EPSILON)
    result = projected.astype(np.float32)
    if result.shape != (value.shape[0], PROJECTION_DIM):
        raise RuntimeError("Projected surface-token shape differs")
    if not np.isfinite(result).all() or float(result.min()) < 0.0:
        raise ValueError("Projected surface tokens must be finite and nonnegative")
    return result


def _local_basis(sample_index: int, *, seed_offset: int = 0) -> np.ndarray:
    local_seed = int(SEED + 1_000_003 * int(sample_index) + 17 + int(seed_offset))
    generator = np.random.default_rng(local_seed)
    basis = generator.random((PROJECTION_DIM, NMF_RANK), dtype=np.float64)
    norms = np.linalg.norm(basis, axis=0, keepdims=True)
    return basis / np.maximum(norms, 1e-12)


def _softmax_rows_numpy(value: np.ndarray) -> np.ndarray:
    shifted = value - np.max(value, axis=1, keepdims=True)
    exponential = np.exp(shifted)
    return exponential / exponential.sum(axis=1, keepdims=True)


def _nmf_objective_numpy(v: np.ndarray, basis: np.ndarray, coef: np.ndarray) -> float:
    residual = v - basis @ coef.T
    return float(0.5 * np.square(residual).sum())


def hamburger_nmf_numpy(
    x: np.ndarray,
    *,
    sample_index: int,
    seed_offset: int = 0,
) -> Dict[str, np.ndarray]:
    value = np.asarray(x, dtype=np.float64)
    if value.ndim != 2 or value.shape[1] != PROJECTION_DIM:
        raise ValueError("NMF input must be [N,32]")
    if value.shape[0] < MIN_SURFACE_TOKENS or float(value.min()) < 0.0:
        raise ValueError("NMF input violates token-count/nonnegativity lock")
    v = value.T.copy()
    basis = _local_basis(sample_index, seed_offset=seed_offset)
    coef = _softmax_rows_numpy(v.T @ basis)
    objectives = [_nmf_objective_numpy(v, basis, coef)]
    for _ in range(NMF_STEPS):
        coef = coef * (v.T @ basis) / (coef @ (basis.T @ basis) + NMF_EPSILON)
        basis = basis * (v @ coef) / (basis @ (coef.T @ coef) + NMF_EPSILON)
        objectives.append(_nmf_objective_numpy(v, basis, coef))
    coef = coef * (v.T @ basis) / (coef @ (basis.T @ basis) + NMF_EPSILON)
    objectives.append(_nmf_objective_numpy(v, basis, coef))
    reconstruction = (basis @ coef.T).T
    result = {
        "reconstruction": reconstruction,
        "basis": basis,
        "coef": coef,
        "objectives": np.asarray(objectives, dtype=np.float64),
    }
    if not all(np.isfinite(item).all() for item in result.values()):
        raise ValueError("NumPy NMF emitted non-finite values")
    return result


def hamburger_nmf_torch(
    x: np.ndarray,
    *,
    sample_index: int,
    seed_offset: int = 0,
) -> Dict[str, np.ndarray]:
    value = np.asarray(x, dtype=np.float64)
    if value.ndim != 2 or value.shape[1] != PROJECTION_DIM:
        raise ValueError("NMF input must be [N,32]")
    if value.shape[0] < MIN_SURFACE_TOKENS or float(value.min()) < 0.0:
        raise ValueError("NMF input violates token-count/nonnegativity lock")
    v = torch.from_numpy(value.T.copy()).to(dtype=torch.float64)
    basis = torch.from_numpy(
        _local_basis(sample_index, seed_offset=seed_offset).copy()
    ).to(dtype=torch.float64)
    coef = torch.softmax(v.T @ basis, dim=1)

    def objective() -> Tensor:
        return 0.5 * torch.square(v - basis @ coef.T).sum()

    objectives = [float(objective().item())]
    for _ in range(NMF_STEPS):
        coef = coef * (v.T @ basis) / (coef @ (basis.T @ basis) + NMF_EPSILON)
        basis = basis * (v @ coef) / (basis @ (coef.T @ coef) + NMF_EPSILON)
        objectives.append(float(objective().item()))
    coef = coef * (v.T @ basis) / (coef @ (basis.T @ basis) + NMF_EPSILON)
    objectives.append(float(objective().item()))
    result = {
        "reconstruction": (basis @ coef.T).T.numpy(),
        "basis": basis.numpy(),
        "coef": coef.numpy(),
        "objectives": np.asarray(objectives, dtype=np.float64),
    }
    if not all(np.isfinite(item).all() for item in result.values()):
        raise ValueError("Torch NMF emitted non-finite values")
    return result


def svd_rank_reconstruction(x: np.ndarray) -> np.ndarray:
    value = np.asarray(x, dtype=np.float64)
    if value.ndim != 2 or min(value.shape) < NMF_RANK:
        raise ValueError("SVD input cannot support the locked rank")
    left, singular, right = np.linalg.svd(value, full_matrices=False)
    return (left[:, :NMF_RANK] * singular[:NMF_RANK]) @ right[:NMF_RANK]


def _distribution_summary(values: np.ndarray) -> np.ndarray:
    vector = np.asarray(values, dtype=np.float64).reshape(-1)
    return np.asarray(
        [
            vector.mean(),
            vector.std(ddof=0),
            vector.min(),
            vector.max(),
            np.quantile(vector, 0.25, method="linear"),
            np.quantile(vector, 0.50, method="linear"),
            np.quantile(vector, 0.75, method="linear"),
            np.quantile(vector, 0.90, method="linear"),
        ],
        dtype=np.float64,
    )


def _effective_rank(value: np.ndarray) -> float:
    singular = np.linalg.svd(
        np.asarray(value, dtype=np.float64), full_matrices=False, compute_uv=False
    )
    energy = np.square(singular)
    total = float(energy.sum())
    if total <= 1e-30:
        return 0.0
    probability = energy / total
    positive = probability > 0.0
    return float(np.exp(-np.sum(probability[positive] * np.log(probability[positive]))))


def surface_descriptor(x: np.ndarray, reconstruction: np.ndarray) -> np.ndarray:
    value = np.asarray(x, dtype=np.float64)
    restored = np.asarray(reconstruction, dtype=np.float64)
    if value.shape != restored.shape or value.ndim != 2 or value.shape[1] != PROJECTION_DIM:
        raise ValueError("Surface descriptor inputs must be aligned [N,32]")
    absolute_residual = np.abs(value - restored)
    channel_columns = []
    for matrix in (restored, absolute_residual):
        channel_columns.extend(
            (
                matrix.mean(axis=0),
                matrix.std(axis=0, ddof=0),
                matrix.max(axis=0),
                np.quantile(matrix, 0.75, axis=0, method="linear"),
            )
        )
    input_norm = np.linalg.norm(value, axis=1)
    reconstruction_norm = np.linalg.norm(restored, axis=1)
    residual_norm = np.linalg.norm(value - restored, axis=1)
    cosine = np.sum(value * restored, axis=1) / (
        input_norm * reconstruction_norm + 1e-12
    )
    token_columns = np.concatenate(
        (
            _distribution_summary(reconstruction_norm),
            _distribution_summary(residual_norm),
            _distribution_summary(cosine),
        )
    )
    input_frobenius = float(np.linalg.norm(value))
    residual_frobenius = float(np.linalg.norm(value - restored))
    explained_energy = 1.0 - residual_frobenius**2 / (input_frobenius**2 + 1e-12)
    scalars = np.asarray(
        [
            residual_frobenius / (input_frobenius + 1e-12),
            explained_energy,
            _effective_rank(restored) / float(min(value.shape)),
            float(np.abs(restored).sum()) / (float(np.abs(value).sum()) + 1e-12),
        ],
        dtype=np.float64,
    )
    result = np.concatenate((*channel_columns, token_columns, scalars))
    if result.shape != (SURFACE_DESCRIPTOR_DIM,):
        raise RuntimeError(f"Surface descriptor shape differs: {result.shape}")
    if not np.isfinite(result).all():
        raise ValueError("Surface descriptor contains non-finite values")
    return result


def _objectives_monotonic(objectives: np.ndarray) -> bool:
    value = np.asarray(objectives, dtype=np.float64)
    previous = value[..., :-1]
    current = value[..., 1:]
    tolerance = 1e-10 * np.maximum(1.0, np.abs(previous))
    return bool(np.all(current <= previous + tolerance))


def engineering_checks() -> Dict[str, object]:
    projection = fixed_projection()
    projection_checks = projection_diagnostics(projection)
    local = np.random.default_rng(1771)
    x = local.random((23, PROJECTION_DIM), dtype=np.float64)
    torch_result = hamburger_nmf_torch(x, sample_index=19)
    numpy_result = hamburger_nmf_numpy(x, sample_index=19)
    differences = {
        key: float(
            np.max(
                np.abs(
                    np.asarray(torch_result[key], dtype=np.float64)
                    - np.asarray(numpy_result[key], dtype=np.float64)
                )
            )
        )
        for key in ("reconstruction", "basis", "coef", "objectives")
    }
    raw_descriptor = surface_descriptor(x, x)
    svd = svd_rank_reconstruction(x)
    svd_descriptor = surface_descriptor(x, svd)
    nmf_descriptor = surface_descriptor(x, torch_result["reconstruction"])
    sklearn_w, sklearn_h, sklearn_iterations = non_negative_factorization(
        x,
        n_components=NMF_RANK,
        init="random",
        update_H=True,
        solver="mu",
        beta_loss="frobenius",
        tol=0.0,
        max_iter=50,
        random_state=SEED,
    )
    sklearn_objective = float(0.5 * np.square(x - sklearn_w @ sklearn_h).sum())
    zero_objective = float(0.5 * np.square(x).sum())
    descriptor_distinct = bool(
        not np.array_equal(raw_descriptor, svd_descriptor)
        and not np.array_equal(raw_descriptor, nmf_descriptor)
        and not np.array_equal(svd_descriptor, nmf_descriptor)
    )
    checks = {
        "projection_exact": bool(projection_checks["passed"]),
        "torch_numpy_oracle_le_1e_10": max(differences.values()) <= 1e-10,
        "candidate_objective_monotonic": _objectives_monotonic(
            torch_result["objectives"]
        ),
        "candidate_nonnegative": float(torch_result["basis"].min()) >= -1e-12
        and float(torch_result["coef"].min()) >= -1e-12
        and float(torch_result["reconstruction"].min()) >= -1e-12,
        "descriptor_dimensions_exact": raw_descriptor.shape
        == svd_descriptor.shape
        == nmf_descriptor.shape
        == (SURFACE_DESCRIPTOR_DIM,),
        "descriptor_roles_distinct": descriptor_distinct,
        "svd_rank_le_8": int(np.linalg.matrix_rank(svd, tol=1e-10)) <= NMF_RANK,
        "sklearn_reference_finite_nonnegative": bool(
            np.isfinite(sklearn_w).all()
            and np.isfinite(sklearn_h).all()
            and float(sklearn_w.min()) >= 0.0
            and float(sklearn_h.min()) >= 0.0
        ),
        "sklearn_reference_reduces_zero_objective": sklearn_objective < zero_objective,
    }
    return {
        "passed": all(checks.values()),
        "checks": checks,
        "projection": projection_checks,
        "oracle_maximum_abs_differences": differences,
        "candidate_objectives": torch_result["objectives"].tolist(),
        "sklearn_reference": {
            "iterations": int(sklearn_iterations),
            "objective": sklearn_objective,
            "zero_reconstruction_objective": zero_objective,
        },
    }


def _global_rng_snapshot() -> Dict[str, object]:
    return {
        "python": random.getstate(),
        "numpy": np.random.get_state(),
        "torch": torch.random.get_rng_state().clone(),
        "cuda": [state.clone() for state in torch.cuda.get_rng_state_all()]
        if torch.cuda.is_available()
        else [],
    }


def _global_rng_equal(left: Mapping[str, object], right: Mapping[str, object]) -> bool:
    numpy_left = left["numpy"]
    numpy_right = right["numpy"]
    return bool(
        left["python"] == right["python"]
        and numpy_left[0] == numpy_right[0]
        and np.array_equal(numpy_left[1], numpy_right[1])
        and numpy_left[2:] == numpy_right[2:]
        and torch.equal(left["torch"], right["torch"])
        and len(left["cuda"]) == len(right["cuda"])
        and all(
            torch.equal(a, b) for a, b in zip(left["cuda"], right["cuda"])
        )
    )


def _normalization_tensors(
    semantics: Mapping[str, object], *, dtype: torch.dtype
) -> Tuple[Tensor, Tensor]:
    mean = torch.tensor(
        semantics["input_mean"], dtype=dtype
    ).view(3, 1, 1)
    std = torch.tensor(
        semantics["input_std"], dtype=dtype
    ).view(3, 1, 1)
    return mean, std


def _to_rgb(images: Tensor, semantics: Mapping[str, object]) -> Tensor:
    mean, std = _normalization_tensors(semantics, dtype=torch.float32)
    value = images.detach().cpu().float()
    return (value * std + mean).clamp(0.0, 1.0)


def _metadata_tensor(
    metadata: Mapping[str, object],
    key: str,
    *,
    device: torch.device,
    dtype: torch.dtype,
) -> Optional[Tensor]:
    value = metadata.get(key)
    if not torch.is_tensor(value):
        return None
    return value.to(device=device, dtype=dtype, non_blocking=True)


def _review_indices(cohort: Sequence[CleanTrainRow]) -> List[int]:
    categories = (FOCUS_CLASS, 0, 2, 4)
    selected: List[int] = []
    for target in categories:
        values = sorted(row.sample_index for row in cohort if row.target == target)[:3]
        if len(values) != 3:
            raise ValueError("Fixed NMF review cohort lacks three category rows")
        selected.extend(values)
    return selected


def build_transformed_crop_bbox_prior(
    crop_bbox: Tensor,
    patch_indices: Tensor,
    *,
    grid_size: Tuple[int, int],
) -> Tensor:
    if crop_bbox.ndim != 2 or int(crop_bbox.shape[1]) < 4:
        raise ValueError("Transformed crop bbox must be [B,4]")
    if patch_indices.ndim != 2 or int(patch_indices.shape[0]) != int(crop_bbox.shape[0]):
        raise ValueError("Patch indices must align with transformed crop bbox")
    grid_height, grid_width = (int(grid_size[0]), int(grid_size[1]))
    if (grid_height, grid_width) != (16, 16):
        raise ValueError("Hamburger-NMF audit locks the original grid to 16x16")
    bbox = crop_bbox[:, :4].to(dtype=torch.float32).clamp(0.0, 1.0)
    cx, cy, width, height = bbox.unbind(dim=1)
    margin = BBOX_PRIOR_MARGIN_RATIO
    x1 = (cx - 0.5 * width - width * margin).clamp(0.0, 1.0)
    y1 = (cy - 0.5 * height - height * margin).clamp(0.0, 1.0)
    x2 = (cx + 0.5 * width + width * margin).clamp(0.0, 1.0)
    y2 = (cy + 0.5 * height + height * margin).clamp(0.0, 1.0)
    x_edges = torch.linspace(
        0.0, 1.0, grid_width + 1, device=bbox.device, dtype=torch.float32
    )
    y_edges = torch.linspace(
        0.0, 1.0, grid_height + 1, device=bbox.device, dtype=torch.float32
    )
    patch_x1 = x_edges[:-1].view(1, 1, grid_width)
    patch_x2 = x_edges[1:].view(1, 1, grid_width)
    patch_y1 = y_edges[:-1].view(1, grid_height, 1)
    patch_y2 = y_edges[1:].view(1, grid_height, 1)
    intersection_width = (
        torch.minimum(patch_x2, x2.view(-1, 1, 1))
        - torch.maximum(patch_x1, x1.view(-1, 1, 1))
    ).clamp(min=0.0)
    intersection_height = (
        torch.minimum(patch_y2, y2.view(-1, 1, 1))
        - torch.maximum(patch_y1, y1.view(-1, 1, 1))
    ).clamp(min=0.0)
    patch_area = (1.0 / float(grid_width)) * (1.0 / float(grid_height))
    full_prior = (intersection_width * intersection_height / patch_area).flatten(1)
    minimum = full_prior.amin(dim=1, keepdim=True)
    maximum = full_prior.amax(dim=1, keepdim=True)
    normalized = (full_prior - minimum) / (maximum - minimum).clamp(min=1e-6)
    return normalized.gather(1, patch_indices.to(dtype=torch.long))


def _extract_surface_row(
    *,
    tokens: Tensor,
    patch_indices: Tensor,
    bbox_prior: Tensor,
    key_padding_mask: Optional[Tensor],
    projection: np.ndarray,
) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    if tokens.ndim != 2 or int(tokens.shape[1]) != PROJECTION_INPUT_DIM:
        raise ValueError("Keeper final patch tokens must be [N,256]")
    if patch_indices.shape != tokens.shape[:1] or bbox_prior.shape != tokens.shape[:1]:
        raise ValueError("Keeper patch index/prior lineage does not align")
    valid = (patch_indices >= 0) & (patch_indices < MAX_PATCHES)
    if key_padding_mask is not None:
        if key_padding_mask.shape != patch_indices.shape:
            raise ValueError("Keeper key-padding mask does not align")
        valid = valid & ~key_padding_mask.to(dtype=torch.bool)
    selected = valid & (bbox_prior >= BBOX_PRIOR_THRESHOLD)
    positions = torch.nonzero(selected, as_tuple=False).flatten()
    if int(positions.numel()) < MIN_SURFACE_TOKENS:
        raise ValueError(
            f"Locked object-surface support has only {int(positions.numel())} tokens"
        )
    selected_indices = patch_indices[positions].detach().cpu().long().numpy()
    order = np.argsort(selected_indices, kind="stable")
    selected_positions = positions.detach().cpu().numpy()[order]
    token_value = tokens[selected_positions].detach().float().cpu().numpy()
    prior_value = bbox_prior[selected_positions].detach().float().cpu().numpy()
    index_value = selected_indices[order].astype(np.int16, copy=False)
    if np.unique(index_value).size != index_value.size:
        raise ValueError("Selected surface patch indices are not unique")
    return (
        project_surface_tokens(token_value, projection),
        index_value,
        prior_value.astype(np.float32, copy=False),
        selected_positions.astype(np.int64, copy=False),
    )


def extract_condition(
    *,
    model: nn.Module,
    loader: DataLoader,
    loader_summary: Mapping[str, object],
    cohort: Sequence[CleanTrainRow],
    device: torch.device,
    semantics: Mapping[str, object],
    condition: str,
    review_indices: Sequence[int],
) -> Dict[str, object]:
    rows = len(cohort)
    projection = fixed_projection()
    projected = np.zeros(
        (rows, MAX_PATCHES, PROJECTION_DIM), dtype=np.float32
    )
    patch_indices_bank = np.full((rows, MAX_PATCHES), -1, dtype=np.int16)
    bbox_prior_bank = np.zeros((rows, MAX_PATCHES), dtype=np.float32)
    keeper_bbox_prior_bank = np.zeros((rows, MAX_PATCHES), dtype=np.float32)
    counts = np.zeros(rows, dtype=np.int16)
    logits_bank = np.empty((rows, 5), dtype=np.float32)
    probability_bank = np.empty((rows, 5), dtype=np.float32)
    target_bank = np.empty(rows, dtype=np.int64)
    sample_index_bank = np.empty(rows, dtype=np.int64)
    review_set = {int(value) for value in review_indices}
    review_rgb: Dict[int, np.ndarray] = {}
    model.eval()
    state_before = _model_state_sha256(model)
    if device.type == "cuda":
        torch.cuda.empty_cache()
        torch.cuda.reset_peak_memory_stats(device)
        torch.cuda.synchronize(device)
    started = time.perf_counter()
    position = 0
    with torch.inference_mode():
        for images, targets, metadata in loader:
            if not isinstance(metadata, Mapping):
                raise ValueError("Hamburger-NMF extraction requires tensor metadata")
            batch = int(targets.numel())
            stop = position + batch
            sample_indices = metadata["sample_index"].detach().cpu().long()
            expected = [row.sample_index for row in cohort[position:stop]]
            if sample_indices.tolist() != expected:
                raise ValueError("Hamburger-NMF loader changed cohort order")
            image_mask = _metadata_tensor(
                metadata, "image_mask", device=device, dtype=torch.bool
            )
            if image_mask is not None and image_mask.ndim == 3:
                image_mask = image_mask.unsqueeze(1)
            bbox = _metadata_tensor(metadata, "bbox", device=device, dtype=torch.float32)
            if bbox is None or bbox.ndim != 2 or int(bbox.shape[1]) < 4:
                raise ValueError("Hamburger-NMF extraction requires object bbox metadata")
            crop_bbox = _metadata_tensor(
                metadata, "crop_bbox", device=device, dtype=torch.float32
            )
            if crop_bbox is None or crop_bbox.ndim != 2 or int(crop_bbox.shape[1]) < 4:
                raise ValueError("Hamburger-NMF extraction requires transformed crop_bbox")
            images_device = images.to(
                device=device, dtype=torch.float32, non_blocking=True
            )
            features = model.forward_features(  # type: ignore[attr-defined]
                images_device,
                image_valid_mask=image_mask,
                bbox_token_prior=bbox,
            )
            features["bbox"] = bbox[:, :4]
            logits = classification_logits_from_features(model, features).float()
            probabilities = logits.softmax(dim=1)
            tokens = features.get("patches")
            patch_indices = features.get("patch_indices")
            bbox_prior = features.get("patch_bbox_prior")
            key_padding = features.get("memory_key_padding_mask")
            if not torch.is_tensor(tokens) or tokens.ndim != 3:
                raise ValueError("Keeper did not expose final patch tokens")
            if not torch.is_tensor(patch_indices) or patch_indices.shape != tokens.shape[:2]:
                raise ValueError("Keeper did not expose aligned patch indices")
            if not torch.is_tensor(bbox_prior) or bbox_prior.shape != tokens.shape[:2]:
                raise ValueError("Keeper did not expose aligned patch bbox prior")
            if torch.is_tensor(key_padding) and key_padding.shape != tokens.shape[:2]:
                raise ValueError("Keeper key-padding mask shape differs")
            grid_size = tuple(int(value) for value in features.get("grid_size", (0, 0)))
            audit_bbox_prior = build_transformed_crop_bbox_prior(
                crop_bbox,
                patch_indices,
                grid_size=grid_size,
            )
            if logits.shape != (batch, 5) or not bool(
                torch.isfinite(logits).all() and torch.isfinite(tokens).all()
            ):
                raise ValueError("Keeper emitted invalid Hamburger-NMF features")
            for local_index in range(batch):
                (
                    surface,
                    selected_indices,
                    selected_priors,
                    selected_positions,
                ) = _extract_surface_row(
                    tokens=tokens[local_index],
                    patch_indices=patch_indices[local_index],
                    bbox_prior=audit_bbox_prior[local_index],
                    key_padding_mask=(
                        key_padding[local_index] if torch.is_tensor(key_padding) else None
                    ),
                    projection=projection,
                )
                count = int(surface.shape[0])
                projected[position + local_index, :count] = surface
                patch_indices_bank[position + local_index, :count] = selected_indices
                bbox_prior_bank[position + local_index, :count] = selected_priors
                selected_position_tensor = torch.as_tensor(
                    selected_positions,
                    device=bbox_prior.device,
                    dtype=torch.long,
                )
                keeper_bbox_prior_bank[position + local_index, :count] = (
                    bbox_prior[local_index]
                    .index_select(0, selected_position_tensor)
                    .detach()
                    .float()
                    .cpu()
                    .numpy()
                )
                counts[position + local_index] = count
            logits_bank[position:stop] = logits.detach().cpu().numpy()
            probability_bank[position:stop] = probabilities.detach().cpu().numpy()
            target_bank[position:stop] = targets.detach().cpu().numpy()
            sample_index_bank[position:stop] = sample_indices.numpy()
            rgb = _to_rgb(images, semantics)
            for local_index, sample_index in enumerate(sample_indices.tolist()):
                if int(sample_index) in review_set:
                    rgb_array = (
                        rgb[local_index]
                        .permute(1, 2, 0)
                        .numpy()
                        .clip(0.0, 1.0)
                    )
                    review_rgb[int(sample_index)] = (
                        (rgb_array * 255.0).round().astype(np.uint8)
                    )
            position = stop
            if position % 256 < batch or position == rows:
                print(
                    json.dumps(
                        {
                            "stage": "hamburger_nmf_token_extraction",
                            "condition": condition,
                            "processed": position,
                            "rows": rows,
                            "elapsed_seconds": time.perf_counter() - started,
                        }
                    ),
                    flush=True,
                )
    if position != rows or len(review_rgb) != len(review_set):
        raise RuntimeError("Hamburger-NMF extraction did not cover its locked rows")
    if device.type == "cuda":
        torch.cuda.synchronize(device)
    state_after = _model_state_sha256(model)
    return {
        "condition": condition,
        "projected_tokens": projected,
        "patch_indices": patch_indices_bank,
        "bbox_priors": bbox_prior_bank,
        "keeper_bbox_priors": keeper_bbox_prior_bank,
        "counts": counts,
        "logits": logits_bank,
        "probabilities": probability_bank,
        "targets": target_bank,
        "sample_indices": sample_index_bank,
        "review_rgb": review_rgb,
        "runtime": {
            "loader": dict(loader_summary),
            "elapsed_seconds": float(time.perf_counter() - started),
            "source_images_per_second": float(rows / max(time.perf_counter() - started, 1e-12)),
            "peak_cuda_bytes": int(
                torch.cuda.max_memory_allocated(device) if device.type == "cuda" else 0
            ),
            "model_state_before_sha256": state_before,
            "model_state_after_sha256": state_after,
        },
        "token_statistics": {
            "minimum": int(counts.min()),
            "maximum": int(counts.max()),
            "mean": float(counts.mean()),
            "all_at_least_16": bool(np.all(counts >= MIN_SURFACE_TOKENS)),
            "all_at_most_256": bool(np.all(counts <= MAX_PATCHES)),
            "selected_prior_minimum": float(
                min(
                    bbox_prior_bank[index, : int(count)].min()
                    for index, count in enumerate(counts)
                )
            ),
            "selected_keeper_prior_minimum": float(
                min(
                    keeper_bbox_prior_bank[index, : int(count)].min()
                    for index, count in enumerate(counts)
                )
            ),
            "selected_keeper_prior_maximum": float(
                max(
                    keeper_bbox_prior_bank[index, : int(count)].max()
                    for index, count in enumerate(counts)
                )
            ),
            "indices_sorted_unique": all(
                np.array_equal(
                    patch_indices_bank[index, : int(count)],
                    np.unique(patch_indices_bank[index, : int(count)]),
                )
                for index, count in enumerate(counts)
            ),
        },
    }


def build_decomposition_bank(
    projected_tokens: np.ndarray,
    counts: np.ndarray,
    sample_indices: np.ndarray,
    *,
    oracle: bool,
    progress_label: Optional[str] = None,
) -> Dict[str, object]:
    tokens = np.asarray(projected_tokens, dtype=np.float32)
    token_counts = np.asarray(counts, dtype=np.int64).reshape(-1)
    indices = np.asarray(sample_indices, dtype=np.int64).reshape(-1)
    rows = int(token_counts.size)
    if tokens.shape != (rows, MAX_PATCHES, PROJECTION_DIM) or indices.size != rows:
        raise ValueError("Projected-token bank shape differs from lock")
    descriptors = {
        role: np.empty((rows, SURFACE_DESCRIPTOR_DIM), dtype=np.float64)
        for role in (
            "raw_surface_control",
            "svd_rank8_control",
            "nmf_candidate",
            "nmf_seed_repeat",
        )
    }
    objective_count = NMF_STEPS + 2
    candidate_objectives = np.empty((rows, objective_count), dtype=np.float64)
    repeat_objectives = np.empty_like(candidate_objectives)
    factor_minima = np.empty((rows, 6), dtype=np.float64)
    solver = hamburger_nmf_numpy if oracle else hamburger_nmf_torch
    before_rng = _global_rng_snapshot()
    started = time.perf_counter()
    for row_index in range(rows):
        count = int(token_counts[row_index])
        if count < MIN_SURFACE_TOKENS or count > MAX_PATCHES:
            raise ValueError("Projected-token count differs from lock")
        x = tokens[row_index, :count].astype(np.float64)
        svd = svd_rank_reconstruction(x)
        candidate = solver(x, sample_index=int(indices[row_index]))
        repeat = solver(
            x,
            sample_index=int(indices[row_index]),
            seed_offset=SEED_REPEAT_OFFSET,
        )
        descriptors["raw_surface_control"][row_index] = surface_descriptor(x, x)
        descriptors["svd_rank8_control"][row_index] = surface_descriptor(x, svd)
        descriptors["nmf_candidate"][row_index] = surface_descriptor(
            x, candidate["reconstruction"]
        )
        descriptors["nmf_seed_repeat"][row_index] = surface_descriptor(
            x, repeat["reconstruction"]
        )
        candidate_objectives[row_index] = candidate["objectives"]
        repeat_objectives[row_index] = repeat["objectives"]
        factor_minima[row_index] = (
            float(candidate["basis"].min()),
            float(candidate["coef"].min()),
            float(candidate["reconstruction"].min()),
            float(repeat["basis"].min()),
            float(repeat["coef"].min()),
            float(repeat["reconstruction"].min()),
        )
        if progress_label and ((row_index + 1) % 128 == 0 or row_index + 1 == rows):
            print(
                json.dumps(
                    {
                        "stage": progress_label,
                        "processed": row_index + 1,
                        "rows": rows,
                        "elapsed_seconds": time.perf_counter() - started,
                    }
                ),
                flush=True,
            )
    after_rng = _global_rng_snapshot()
    candidate_descriptor = descriptors["nmf_candidate"]
    repeat_descriptor = descriptors["nmf_seed_repeat"]
    cosine = np.sum(candidate_descriptor * repeat_descriptor, axis=1) / (
        np.linalg.norm(candidate_descriptor, axis=1)
        * np.linalg.norm(repeat_descriptor, axis=1)
        + 1e-12
    )
    return {
        "descriptors": descriptors,
        "candidate_objectives": candidate_objectives,
        "repeat_objectives": repeat_objectives,
        "factor_minima": factor_minima,
        "diagnostics": {
            "solver": "numpy_oracle" if oracle else "torch_production",
            "candidate_objectives_monotonic_all_rows": _objectives_monotonic(
                candidate_objectives
            ),
            "repeat_objectives_monotonic_all_rows": _objectives_monotonic(
                repeat_objectives
            ),
            "minimum_factor_or_reconstruction": float(factor_minima.min()),
            "all_nonnegative_within_1e_12": float(factor_minima.min()) >= -1e-12,
            "median_seed_repeat_descriptor_cosine": float(np.median(cosine)),
            "minimum_seed_repeat_descriptor_cosine": float(cosine.min()),
            "local_rng_only": _global_rng_equal(before_rng, after_rng),
            "elapsed_seconds": float(time.perf_counter() - started),
        },
    }


def _descriptor_statistics(descriptor: np.ndarray) -> Dict[str, object]:
    values = np.asarray(descriptor, dtype=np.float64)
    if values.ndim != 2 or int(values.shape[1]) != SURFACE_DESCRIPTOR_DIM:
        raise ValueError("Surface descriptor bank shape differs")
    centered = values - values.mean(axis=0, keepdims=True)
    singular = np.linalg.svd(centered, full_matrices=False, compute_uv=False)
    energy = np.square(singular)
    distribution = energy / max(float(energy.sum()), 1e-30)
    positive = distribution > 0.0
    effective_rank = float(
        np.exp(-np.sum(distribution[positive] * np.log(distribution[positive])))
    )
    standard_deviation = values.std(axis=0, ddof=0)
    return {
        "shape": [int(item) for item in values.shape],
        "sha256": _array_sha256(values),
        "effective_rank": effective_rank,
        "active_dimensions": int((standard_deviation > 1e-8).sum()),
        "minimum_std": float(standard_deviation.min()),
        "maximum_std": float(standard_deviation.max()),
        "all_values_finite": bool(np.isfinite(values).all()),
    }


def build_source_derangement(
    descriptors: np.ndarray,
    *,
    folds: np.ndarray,
    clean_margins: np.ndarray,
    source_stems: Sequence[str],
    sample_indices: np.ndarray,
) -> Tuple[np.ndarray, np.ndarray, List[Dict[str, object]]]:
    values = np.asarray(descriptors, dtype=np.float64)
    fold_values = np.asarray(folds, dtype=np.int64).reshape(-1)
    margins = np.asarray(clean_margins, dtype=np.float64).reshape(-1)
    indices = np.asarray(sample_indices, dtype=np.int64).reshape(-1)
    sources = np.asarray([str(value) for value in source_stems], dtype=str)
    rows = int(values.shape[0])
    if not (fold_values.size == margins.size == indices.size == sources.size == rows):
        raise ValueError("Source-derangement inputs do not align")
    assigned = np.full(rows, -1, dtype=np.int64)
    mapping: List[Dict[str, object]] = []
    for fold in FOLDS:
        fold_positions = np.flatnonzero(fold_values == int(fold))
        ranked = fold_positions[
            np.lexsort((indices[fold_positions], margins[fold_positions]))
        ]
        quartiles = np.minimum(
            3, (4 * np.arange(ranked.size, dtype=np.int64)) // ranked.size
        )
        for quartile in range(4):
            partition = ranked[quartiles == quartile]
            partition = partition[np.argsort(indices[partition], kind="stable")]
            if partition.size < 2:
                raise ValueError("Source-derangement partition is too small")
            chosen_shift = None
            for shift in range(1, int(partition.size)):
                candidates = np.roll(partition, -shift)
                if np.all(sources[partition] != sources[candidates]):
                    chosen_shift = int(shift)
                    assigned[partition] = candidates
                    break
            if chosen_shift is None:
                raise RuntimeError(
                    f"No source-safe derangement for fold={fold}, quartile={quartile}"
                )
            for destination, source in zip(partition.tolist(), assigned[partition].tolist()):
                mapping.append(
                    {
                        "destination_position": int(destination),
                        "source_position": int(source),
                        "destination_sample_index": int(indices[destination]),
                        "source_sample_index": int(indices[source]),
                        "fold": int(fold),
                        "quartile": int(quartile),
                        "cyclic_shift": int(chosen_shift),
                        "destination_source_stem": str(sources[destination]),
                        "descriptor_source_stem": str(sources[source]),
                    }
                )
    if bool((assigned < 0).any()):
        raise RuntimeError("Source derangement is incomplete")
    if not np.array_equal(fold_values, fold_values[assigned]):
        raise RuntimeError("Source derangement crossed a held fold")
    if np.any(sources == sources[assigned]):
        raise RuntimeError("Source derangement retained an identical source")
    mapping.sort(key=lambda row: int(row["destination_sample_index"]))
    return values[assigned], assigned, mapping


def build_readout_features(
    *,
    logits: np.ndarray,
    probabilities: np.ndarray,
    descriptors: Mapping[str, np.ndarray],
    folds: np.ndarray,
    source_stems: Sequence[str],
    sample_indices: np.ndarray,
    assigned: Optional[np.ndarray] = None,
) -> Tuple[Dict[str, np.ndarray], np.ndarray, List[Dict[str, object]]]:
    logit_values = np.asarray(logits, dtype=np.float64)
    probability_values = np.asarray(probabilities, dtype=np.float64)
    if logit_values.shape != probability_values.shape or logit_values.shape[1] != 5:
        raise ValueError("Keeper logits/probabilities must be aligned [N,5]")
    margin = logit_values[:, FOCUS_CLASS] - logit_values[
        :, RESTRICTED_NEGATIVE_CLASSES
    ].max(axis=1)
    base = np.concatenate(
        (
            np.log(np.clip(probability_values, LOG_PROBABILITY_CLIP, 1.0)),
            margin[:, None],
        ),
        axis=1,
    )
    candidate = np.asarray(descriptors["nmf_candidate"], dtype=np.float64)
    mapping: List[Dict[str, object]] = []
    if assigned is None:
        deranged, assigned_values, mapping = build_source_derangement(
            candidate,
            folds=folds,
            clean_margins=margin,
            source_stems=source_stems,
            sample_indices=sample_indices,
        )
    else:
        assigned_values = np.asarray(assigned, dtype=np.int64)
        if assigned_values.shape != (candidate.shape[0],):
            raise ValueError("Persisted derangement assignment shape differs")
        deranged = candidate[assigned_values]
    features = {
        "base_only": base,
        "raw_surface_control": np.concatenate(
            (base, np.asarray(descriptors["raw_surface_control"], dtype=np.float64)),
            axis=1,
        ),
        "svd_rank8_control": np.concatenate(
            (base, np.asarray(descriptors["svd_rank8_control"], dtype=np.float64)),
            axis=1,
        ),
        "nmf_candidate": np.concatenate((base, candidate), axis=1),
        "nmf_seed_repeat": np.concatenate(
            (base, np.asarray(descriptors["nmf_seed_repeat"], dtype=np.float64)),
            axis=1,
        ),
        "source_deranged": np.concatenate((base, deranged), axis=1),
    }
    for role, values in features.items():
        expected_dim = BASE_DIM if role == "base_only" else ROLE_DIM
        if values.shape != (probability_values.shape[0], expected_dim):
            raise ValueError(f"Readout feature shape differs for {role}: {values.shape}")
        if not np.isfinite(values).all():
            raise ValueError(f"Readout feature contains non-finite values: {role}")
    return features, assigned_values, mapping


def _save_token_cache(path: Path, extraction: Mapping[str, object]) -> None:
    review_indices = np.asarray(sorted(extraction["review_rgb"]), dtype=np.int64)
    review_rgb = np.stack(
        [extraction["review_rgb"][int(index)] for index in review_indices], axis=0
    )
    np.savez_compressed(
        path,
        projected_tokens=np.asarray(extraction["projected_tokens"], dtype=np.float32),
        patch_indices=np.asarray(extraction["patch_indices"], dtype=np.int16),
        bbox_priors=np.asarray(extraction["bbox_priors"], dtype=np.float32),
        keeper_bbox_priors=np.asarray(
            extraction["keeper_bbox_priors"], dtype=np.float32
        ),
        counts=np.asarray(extraction["counts"], dtype=np.int16),
        logits=np.asarray(extraction["logits"], dtype=np.float32),
        probabilities=np.asarray(extraction["probabilities"], dtype=np.float32),
        targets=np.asarray(extraction["targets"], dtype=np.int64),
        sample_indices=np.asarray(extraction["sample_indices"], dtype=np.int64),
        review_indices=review_indices,
        review_rgb=review_rgb.astype(np.uint8, copy=False),
    )


def _load_token_cache(path: Path) -> Dict[str, np.ndarray]:
    with np.load(path, allow_pickle=False) as payload:
        return {key: payload[key] for key in payload.files}


def _save_decomposition_cache(path: Path, bank: Mapping[str, object]) -> None:
    descriptors = bank["descriptors"]
    np.savez_compressed(
        path,
        raw_surface_control=np.asarray(descriptors["raw_surface_control"], dtype=np.float64),
        svd_rank8_control=np.asarray(descriptors["svd_rank8_control"], dtype=np.float64),
        nmf_candidate=np.asarray(descriptors["nmf_candidate"], dtype=np.float64),
        nmf_seed_repeat=np.asarray(descriptors["nmf_seed_repeat"], dtype=np.float64),
        candidate_objectives=np.asarray(bank["candidate_objectives"], dtype=np.float64),
        repeat_objectives=np.asarray(bank["repeat_objectives"], dtype=np.float64),
        factor_minima=np.asarray(bank["factor_minima"], dtype=np.float64),
    )


def _load_decomposition_cache(path: Path) -> Dict[str, np.ndarray]:
    with np.load(path, allow_pickle=False) as payload:
        return {key: payload[key] for key in payload.files}


def _resize_map(values: np.ndarray, size: Tuple[int, int]) -> np.ndarray:
    array = np.asarray(values, dtype=np.float32)
    image = Image.fromarray(array)
    resampling = getattr(Image, "Resampling", Image)
    return np.asarray(image.resize(size, resampling.NEAREST), dtype=np.float32)


def _heat_overlay(rgb: np.ndarray, heat: np.ndarray) -> Image.Image:
    base = np.asarray(rgb, dtype=np.float32)
    value = np.asarray(heat, dtype=np.float32)
    maximum = float(value.max())
    if maximum > 0.0:
        value = value / maximum
    color = np.stack((value, 0.25 * value, 1.0 - value), axis=2) * 255.0
    alpha = (0.62 * value)[..., None]
    result = base * (1.0 - alpha) + color * alpha
    return Image.fromarray(result.clip(0.0, 255.0).round().astype(np.uint8))


def _categorical_overlay(rgb: np.ndarray, categories: np.ndarray) -> Image.Image:
    palette = np.asarray(
        [
            (230, 25, 75),
            (60, 180, 75),
            (0, 130, 200),
            (245, 130, 48),
            (145, 30, 180),
            (70, 240, 240),
            (240, 50, 230),
            (210, 245, 60),
        ],
        dtype=np.float32,
    )
    base = np.asarray(rgb, dtype=np.float32)
    labels = np.asarray(categories, dtype=np.int16)
    valid = labels >= 0
    color = np.zeros_like(base)
    color[valid] = palette[labels[valid] % len(palette)]
    alpha = (0.55 * valid.astype(np.float32))[..., None]
    result = base * (1.0 - alpha) + color * alpha
    return Image.fromarray(result.clip(0.0, 255.0).round().astype(np.uint8))


def render_contact_sheet(
    path: Path,
    *,
    token_cache: Mapping[str, np.ndarray],
    cohort: Sequence[CleanTrainRow],
) -> Dict[str, object]:
    review_indices = [int(value) for value in token_cache["review_indices"].tolist()]
    rgb_lookup = {
        sample_index: token_cache["review_rgb"][position]
        for position, sample_index in enumerate(review_indices)
    }
    position_by_index = {row.sample_index: position for position, row in enumerate(cohort)}
    tile = 256
    top = 42
    left = 220
    columns = (
        "input",
        "object-token support",
        "dominant NMF component",
        "reconstruction strength",
        "residual heat",
    )
    canvas = Image.new(
        "RGB", (left + tile * len(columns), top + tile * len(review_indices)), "white"
    )
    draw = ImageDraw.Draw(canvas)
    font = ImageFont.load_default()
    for column_index, label in enumerate(columns):
        draw.text((left + column_index * tile + 6, 12), label, fill="black", font=font)
    manifest: List[Dict[str, object]] = []
    for review_row, sample_index in enumerate(review_indices):
        position = position_by_index[sample_index]
        cohort_row = cohort[position]
        count = int(token_cache["counts"][position])
        x = token_cache["projected_tokens"][position, :count].astype(np.float64)
        patch_indices = token_cache["patch_indices"][position, :count].astype(np.int64)
        result = hamburger_nmf_numpy(x, sample_index=sample_index)
        coefficient = result["coef"]
        component_order = np.argsort(-coefficient.sum(axis=0), kind="stable")
        inverse_order = np.empty_like(component_order)
        inverse_order[component_order] = np.arange(component_order.size)
        dominant = inverse_order[np.argmax(coefficient, axis=1)]
        reconstruction_strength = np.linalg.norm(result["reconstruction"], axis=1)
        residual = np.linalg.norm(x - result["reconstruction"], axis=1)
        support_grid = np.zeros((16, 16), dtype=np.float32)
        component_grid = np.full((16, 16), -1, dtype=np.int16)
        reconstruction_grid = np.zeros((16, 16), dtype=np.float32)
        residual_grid = np.zeros((16, 16), dtype=np.float32)
        for local, patch_index in enumerate(patch_indices.tolist()):
            y, x_index = divmod(int(patch_index), 16)
            support_grid[y, x_index] = 1.0
            component_grid[y, x_index] = int(dominant[local])
            reconstruction_grid[y, x_index] = float(reconstruction_strength[local])
            residual_grid[y, x_index] = float(residual[local])
        rgb = rgb_lookup[sample_index]
        size = (int(rgb.shape[1]), int(rgb.shape[0]))
        support = _resize_map(support_grid, size)
        component = np.asarray(
            Image.fromarray(component_grid).resize(
                size, getattr(Image, "Resampling", Image).NEAREST
            ),
            dtype=np.int16,
        )
        recon_map = _resize_map(reconstruction_grid, size)
        residual_map = _resize_map(residual_grid, size)
        support_overlay = _heat_overlay(rgb, support)
        component_overlay = _categorical_overlay(rgb, component)
        tiles = (
            Image.fromarray(rgb),
            support_overlay,
            component_overlay,
            _heat_overlay(rgb, recon_map),
            _heat_overlay(rgb, residual_map),
        )
        y0 = top + review_row * tile
        draw.text(
            (4, y0 + 8),
            f"target={cohort_row.target}\nidx={sample_index}\nfold={cohort_row.fold}\ntokens={count}",
            fill="black",
            font=font,
        )
        for column_index, image in enumerate(tiles):
            canvas.paste(image.resize((tile, tile)), (left + column_index * tile, y0))
        manifest.append(
            {
                "row": int(review_row),
                "sample_index": int(sample_index),
                "target": int(cohort_row.target),
                "fold": int(cohort_row.fold),
                "surface_tokens": count,
                "columns": list(columns),
            }
        )
    canvas.save(path)
    manifest_path = path.with_suffix(".json")
    manifest_path.write_text(json.dumps(manifest, indent=2), encoding="utf-8")
    return {
        "path": str(path.resolve()),
        "sha256": _sha256(path),
        "manifest_path": str(manifest_path.resolve()),
        "manifest_sha256": _sha256(manifest_path),
        "rows": len(manifest),
        "columns": list(columns),
    }


def _positive_threshold(scores: np.ndarray, labels: np.ndarray) -> float:
    values = np.asarray(scores, dtype=np.float64).reshape(-1)
    binary = np.asarray(labels, dtype=np.int64).reshape(-1)
    positives = np.sort(values[binary == 1])
    if positives.size == 0:
        raise ValueError("Readout fit folds have no positive rows")
    allowed_breaks = int(math.floor((1.0 - MIN_FIT_TP_RETENTION) * positives.size))
    threshold = float(positives[min(allowed_breaks, positives.size - 1)])
    if float((positives >= threshold).mean()) + 1e-12 < MIN_FIT_TP_RETENTION:
        raise RuntimeError("Locked fit-fold threshold violates TP retention")
    return threshold


def _positive_probability(model: LogisticRegression, features: np.ndarray) -> np.ndarray:
    probabilities = model.predict_proba(features)
    classes = np.asarray(model.classes_, dtype=np.int64)
    matches = np.flatnonzero(classes == 1)
    if matches.size != 1:
        raise ValueError("Binary readout lacks exactly one positive class")
    return np.asarray(probabilities[:, int(matches[0])], dtype=np.float64)


def fit_clean_oof_readouts(
    features: Mapping[str, np.ndarray],
    labels: np.ndarray,
    folds: np.ndarray,
    source_stems: Sequence[str],
) -> Tuple[Dict[str, np.ndarray], Dict[str, np.ndarray], Dict[str, object]]:
    if set(features) != set(ROLE_NAMES):
        raise ValueError("Readout feature roles differ from Hamburger-NMF lock")
    binary = np.asarray(labels, dtype=np.int64).reshape(-1)
    fold_values = np.asarray(folds, dtype=np.int64).reshape(-1)
    sources = np.asarray([str(value) for value in source_stems], dtype=str)
    rows = int(binary.size)
    scores = {role: np.full(rows, np.nan, dtype=np.float64) for role in ROLE_NAMES}
    actions = {role: np.zeros(rows, dtype=np.bool_) for role in ROLE_NAMES}
    states: Dict[str, object] = {role: {"folds": []} for role in ROLE_NAMES}
    for role in ROLE_NAMES:
        values = np.asarray(features[role], dtype=np.float64)
        for held_fold in FOLDS:
            held = fold_values == int(held_fold)
            fit = ~held
            if not bool(held.any()) or np.unique(binary[fit]).size != 2:
                raise ValueError(f"Readout fold {held_fold} lacks binary support")
            fit_sources = set(sources[fit].tolist())
            held_sources = set(sources[held].tolist())
            if fit_sources.intersection(held_sources):
                raise ValueError(f"Readout fold {held_fold} has source leakage")
            scaler = StandardScaler()
            fit_scaled = scaler.fit_transform(values[fit])
            with warnings.catch_warnings(record=True) as caught:
                warnings.simplefilter("always", ConvergenceWarning)
                model = LogisticRegression(
                    C=READOUT_C,
                    penalty="l2",
                    class_weight="balanced",
                    max_iter=READOUT_MAX_ITER,
                    tol=READOUT_TOLERANCE,
                    random_state=SEED,
                    solver="lbfgs",
                )
                model.fit(fit_scaled, binary[fit])
            fit_scores = _positive_probability(model, fit_scaled)
            threshold = _positive_threshold(fit_scores, binary[fit])
            held_scores = _positive_probability(model, scaler.transform(values[held]))
            scores[role][held] = held_scores
            actions[role][held] = held_scores >= threshold
            convergence_warnings = [
                str(item.message)
                for item in caught
                if issubclass(item.category, ConvergenceWarning)
            ]
            maximum_iterations = int(np.max(model.n_iter_))
            states[role]["folds"].append(
                {
                    "held_fold": int(held_fold),
                    "fit_folds": [int(value) for value in FOLDS if value != held_fold],
                    "fit_rows": int(fit.sum()),
                    "held_rows": int(held.sum()),
                    "fit_sources": int(len(fit_sources)),
                    "held_sources": int(len(held_sources)),
                    "source_overlap": 0,
                    "classes": model.classes_.astype(int).tolist(),
                    "threshold": threshold,
                    "scaler_mean": scaler.mean_.tolist(),
                    "scaler_scale": scaler.scale_.tolist(),
                    "coefficient": model.coef_.tolist(),
                    "intercept": model.intercept_.tolist(),
                    "iterations": model.n_iter_.astype(int).tolist(),
                    "convergence_warnings": convergence_warnings,
                    "converged": not convergence_warnings
                    and maximum_iterations < READOUT_MAX_ITER,
                    "fit_tp_retention": float(
                        (fit_scores[binary[fit] == 1] >= threshold).mean()
                    ),
                }
            )
    if any(not np.isfinite(values).all() for values in scores.values()):
        raise RuntimeError("OOF readout did not score every row")
    return scores, actions, states


def apply_readout_states(
    features: Mapping[str, np.ndarray],
    folds: np.ndarray,
    states: Mapping[str, object],
) -> Tuple[Dict[str, np.ndarray], Dict[str, np.ndarray]]:
    fold_values = np.asarray(folds, dtype=np.int64).reshape(-1)
    rows = int(fold_values.size)
    scores = {role: np.full(rows, np.nan, dtype=np.float64) for role in ROLE_NAMES}
    actions = {role: np.zeros(rows, dtype=np.bool_) for role in ROLE_NAMES}
    for role in ROLE_NAMES:
        values = np.asarray(features[role], dtype=np.float64)
        for state in states[role]["folds"]:
            held_fold = int(state["held_fold"])
            held = fold_values == held_fold
            mean = np.asarray(state["scaler_mean"], dtype=np.float64)
            scale = np.asarray(state["scaler_scale"], dtype=np.float64)
            coefficient = np.asarray(state["coefficient"], dtype=np.float64)
            intercept = np.asarray(state["intercept"], dtype=np.float64)
            transformed = (values[held] - mean) / scale
            decision = transformed @ coefficient.T + intercept
            if decision.shape[1] != 1:
                raise ValueError("Locked binary readout must have one decision column")
            held_scores = 1.0 / (1.0 + np.exp(-decision[:, 0]))
            scores[role][held] = held_scores
            actions[role][held] = held_scores >= float(state["threshold"])
    if any(not np.isfinite(values).all() for values in scores.values()):
        raise RuntimeError("Persisted readout state did not score every row")
    return scores, actions


def _rival_predictions(probabilities: np.ndarray) -> np.ndarray:
    values = np.asarray(probabilities, dtype=np.float64)
    restricted = values[:, RESTRICTED_NEGATIVE_CLASSES]
    local = restricted.argmax(axis=1)
    classes = np.asarray(RESTRICTED_NEGATIVE_CLASSES, dtype=np.int64)
    return classes[local]


def _role_metrics(
    *,
    labels: np.ndarray,
    targets: np.ndarray,
    folds: np.ndarray,
    scores: np.ndarray,
    actions: np.ndarray,
    probabilities: np.ndarray,
) -> Dict[str, object]:
    binary = np.asarray(labels, dtype=np.int64).reshape(-1)
    target_values = np.asarray(targets, dtype=np.int64).reshape(-1)
    fold_values = np.asarray(folds, dtype=np.int64).reshape(-1)
    score_values = np.asarray(scores, dtype=np.float64).reshape(-1)
    accepted = np.asarray(actions, dtype=np.bool_).reshape(-1)
    if not (
        binary.size
        == target_values.size
        == fold_values.size
        == score_values.size
        == accepted.size
    ):
        raise ValueError("Role-metric inputs do not align")
    positive = binary == 1
    negative = binary == 0
    rivals = _rival_predictions(probabilities)
    action_predictions = np.where(accepted, FOCUS_CLASS, rivals)
    corrections = negative & (action_predictions == target_values)
    harms = positive & ~accepted
    per_fold = []
    for fold in FOLDS:
        selected = fold_values == int(fold)
        fold_positive = positive & selected
        fold_negative = negative & selected
        per_fold.append(
            {
                "fold": int(fold),
                "rows": int(selected.sum()),
                "auroc": float(roc_auc_score(binary[selected], score_values[selected])),
                "tp_retention": float(accepted[fold_positive].mean()),
                "fp_rejection": float((~accepted[fold_negative]).mean()),
                "tp_breaks": int((~accepted[fold_positive]).sum()),
                "fp_rejects": int((~accepted[fold_negative]).sum()),
                "corrections": int(corrections[selected].sum()),
                "harms": int(harms[selected].sum()),
            }
        )
    per_target = {}
    for target in RESTRICTED_NEGATIVE_CLASSES:
        selected = target_values == int(target)
        per_target[str(target)] = {
            "rows": int(selected.sum()),
            "rejected": int((~accepted[selected]).sum()),
            "rejection_rate": float((~accepted[selected]).mean()),
            "corrected": int(corrections[selected].sum()),
        }
    return {
        "auroc": float(roc_auc_score(binary, score_values)),
        "tp_retention": float(accepted[positive].mean()),
        "fp_rejection": float((~accepted[negative]).mean()),
        "tp_breaks": int((~accepted[positive]).sum()),
        "fp_rejects": int((~accepted[negative]).sum()),
        "corrections": int(corrections.sum()),
        "harms": int(harms.sum()),
        "neutral_rejections": int(((~accepted) & ~corrections & ~harms).sum()),
        "action_semantics_exact": bool(
            np.all(action_predictions[accepted] == FOCUS_CLASS)
            and np.array_equal(action_predictions[~accepted], rivals[~accepted])
        ),
        "action_prediction_sha256": _array_sha256(action_predictions),
        "per_fold": per_fold,
        "per_target": per_target,
    }


def build_clean_analysis(
    *,
    labels: np.ndarray,
    targets: np.ndarray,
    folds: np.ndarray,
    scores: Mapping[str, np.ndarray],
    actions: Mapping[str, np.ndarray],
    probabilities: np.ndarray,
    candidate_descriptor_stats: Mapping[str, object],
    decomposition_diagnostics: Mapping[str, object],
) -> Dict[str, object]:
    metrics = {
        role: _role_metrics(
            labels=labels,
            targets=targets,
            folds=folds,
            scores=scores[role],
            actions=actions[role],
            probabilities=probabilities,
        )
        for role in ROLE_NAMES
    }
    candidate = metrics["nmf_candidate"]
    candidate_auc = float(candidate["auroc"])
    gains = {
        role: candidate_auc - float(metrics[role]["auroc"])
        for role in (*CONTROL_ROLES, "source_deranged")
    }
    control_fp_rejects = max(
        int(metrics[role]["fp_rejects"]) for role in CONTROL_ROLES
    )
    fold_wins = 0
    candidate_folds = {int(row["fold"]): row for row in candidate["per_fold"]}
    raw_folds = {
        int(row["fold"]): row for row in metrics["raw_surface_control"]["per_fold"]
    }
    svd_folds = {
        int(row["fold"]): row for row in metrics["svd_rank8_control"]["per_fold"]
    }
    for fold in FOLDS:
        fold_wins += int(
            float(candidate_folds[fold]["auroc"]) > float(raw_folds[fold]["auroc"])
            and float(candidate_folds[fold]["auroc"]) > float(svd_folds[fold]["auroc"])
        )
    seed_auc_delta = abs(
        candidate_auc - float(metrics["nmf_seed_repeat"]["auroc"])
    )
    seed_action_agreement = float(
        np.mean(
            np.asarray(actions["nmf_candidate"], dtype=np.bool_)
            == np.asarray(actions["nmf_seed_repeat"], dtype=np.bool_)
        )
    )
    gates = {
        "candidate_auroc_ge_085": candidate_auc >= 0.85,
        "candidate_gain_vs_base_ge_002": gains["base_only"] >= 0.02,
        "candidate_gain_vs_raw_ge_001": gains["raw_surface_control"] >= 0.01,
        "candidate_gain_vs_svd_ge_001": gains["svd_rank8_control"] >= 0.01,
        "candidate_gain_vs_source_deranged_ge_002": gains["source_deranged"] >= 0.02,
        "candidate_tp_retention_ge_095": float(candidate["tp_retention"]) >= 0.95,
        "candidate_fp_rejection_ge_025": float(candidate["fp_rejection"]) >= 0.25,
        "candidate_fp_rejects_10_above_strongest_control": int(
            candidate["fp_rejects"]
        )
        >= control_fp_rejects + 10,
        "candidate_corrections_ge_2x_harms": int(candidate["corrections"])
        >= 2 * int(candidate["harms"]),
        "candidate_wins_raw_and_svd_in_4_of_5_folds": fold_wins >= 4,
        "minimum_fold_tp_retention_ge_090": min(
            float(row["tp_retention"]) for row in candidate["per_fold"]
        )
        >= 0.90,
        "target0_fp_rejection_ge_015": float(
            candidate["per_target"]["0"]["rejection_rate"]
        )
        >= 0.15,
        "target2_fp_rejection_ge_015": float(
            candidate["per_target"]["2"]["rejection_rate"]
        )
        >= 0.15,
        "target4_fp_rejects_ge_1": int(candidate["per_target"]["4"]["rejected"])
        >= 1,
        "candidate_descriptor_effective_rank_ge_12": float(
            candidate_descriptor_stats["effective_rank"]
        )
        >= 12.0,
        "median_seed_repeat_descriptor_cosine_ge_095": float(
            decomposition_diagnostics["median_seed_repeat_descriptor_cosine"]
        )
        >= 0.95,
        "seed_repeat_auroc_difference_le_001": seed_auc_delta <= 0.01,
        "seed_repeat_action_agreement_ge_095": seed_action_agreement >= 0.95,
        "nmf_objectives_monotonic": bool(
            decomposition_diagnostics["candidate_objectives_monotonic_all_rows"]
        )
        and bool(decomposition_diagnostics["repeat_objectives_monotonic_all_rows"]),
        "nmf_values_nonnegative": bool(
            decomposition_diagnostics["all_nonnegative_within_1e_12"]
        ),
        "all_action_semantics_exact": all(
            bool(metrics[role]["action_semantics_exact"]) for role in ROLE_NAMES
        ),
    }
    return {
        "metrics": metrics,
        "candidate_auroc_gains": gains,
        "candidate_additional_fp_rejects_vs_strongest_control": int(
            candidate["fp_rejects"]
        )
        - control_fp_rejects,
        "candidate_fold_wins_over_raw_and_svd": int(fold_wins),
        "seed_repeat_auroc_absolute_difference": seed_auc_delta,
        "seed_repeat_action_agreement": seed_action_agreement,
        "mechanism_gates": gates,
        "mechanism_gates_passed": all(gates.values()),
    }


def build_condition_analysis(
    *,
    labels: np.ndarray,
    targets: np.ndarray,
    folds: np.ndarray,
    scores: Mapping[str, np.ndarray],
    actions: Mapping[str, np.ndarray],
    probabilities: np.ndarray,
) -> Dict[str, object]:
    metrics = {
        role: _role_metrics(
            labels=labels,
            targets=targets,
            folds=folds,
            scores=scores[role],
            actions=actions[role],
            probabilities=probabilities,
        )
        for role in ROLE_NAMES
    }
    candidate = metrics["nmf_candidate"]
    strongest_control_auc = max(
        float(metrics[role]["auroc"]) for role in CONTROL_ROLES
    )
    seed_action_agreement = float(
        np.mean(
            np.asarray(actions["nmf_candidate"], dtype=np.bool_)
            == np.asarray(actions["nmf_seed_repeat"], dtype=np.bool_)
        )
    )
    gates = {
        "candidate_auroc_ge_078": float(candidate["auroc"]) >= 0.78,
        "candidate_gain_vs_strongest_control_ge_0005": float(candidate["auroc"])
        - strongest_control_auc
        >= 0.005,
        "candidate_tp_retention_ge_092": float(candidate["tp_retention"]) >= 0.92,
        "candidate_fp_rejection_ge_015": float(candidate["fp_rejection"]) >= 0.15,
        "candidate_corrections_not_below_harms": int(candidate["corrections"])
        >= int(candidate["harms"]),
        "target0_fp_rejection_ge_008": float(
            candidate["per_target"]["0"]["rejection_rate"]
        )
        >= 0.08,
        "target2_fp_rejection_ge_008": float(
            candidate["per_target"]["2"]["rejection_rate"]
        )
        >= 0.08,
        "seed_repeat_action_agreement_ge_090": seed_action_agreement >= 0.90,
    }
    return {
        "metrics": metrics,
        "strongest_control_auroc": strongest_control_auc,
        "candidate_gain_vs_strongest_control": float(candidate["auroc"])
        - strongest_control_auc,
        "seed_repeat_action_agreement": seed_action_agreement,
        "gates": gates,
        "passed": all(gates.values()),
    }


def _readout_states_converged(states: Mapping[str, object]) -> bool:
    return all(
        bool(row["converged"])
        for role in ROLE_NAMES
        for row in states[role]["folds"]
    )


def _feature_dimensions(features: Mapping[str, np.ndarray]) -> Dict[str, int]:
    return {role: int(np.asarray(value).shape[1]) for role, value in features.items()}


def _maximum_role_difference(
    left: Mapping[str, np.ndarray], right: Mapping[str, np.ndarray]
) -> float:
    return max(
        float(
            np.max(
                np.abs(
                    np.asarray(left[role], dtype=np.float64)
                    - np.asarray(right[role], dtype=np.float64)
                )
            )
        )
        for role in ROLE_NAMES
    )


def _recursive_numeric_difference(left: object, right: object) -> float:
    if isinstance(left, Mapping) and isinstance(right, Mapping):
        if set(left) != set(right):
            return float("inf")
        return max(
            (_recursive_numeric_difference(left[key], right[key]) for key in left),
            default=0.0,
        )
    if isinstance(left, (list, tuple)) and isinstance(right, (list, tuple)):
        if len(left) != len(right):
            return float("inf")
        return max(
            (_recursive_numeric_difference(a, b) for a, b in zip(left, right)),
            default=0.0,
        )
    if isinstance(left, (bool, str)) or left is None:
        return 0.0 if left == right else float("inf")
    if isinstance(left, (int, float, np.number)) and isinstance(
        right, (int, float, np.number)
    ):
        a = float(left)
        b = float(right)
        if math.isnan(a) and math.isnan(b):
            return 0.0
        return abs(a - b)
    return 0.0 if left == right else float("inf")


def _write_json(path: Path, payload: Mapping[str, object]) -> None:
    path.write_text(
        json.dumps(to_serializable(dict(payload)), indent=2, sort_keys=True),
        encoding="utf-8",
    )


def _write_csv(path: Path, rows: Sequence[Mapping[str, object]]) -> None:
    values = list(rows)
    if not values:
        raise ValueError(f"Cannot write empty CSV: {path}")
    fields = list(values[0])
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields)
        writer.writeheader()
        writer.writerows(values)


def _prepare_output_dir(path: Path) -> Path:
    resolved = Path(path).expanduser().resolve()
    if resolved.exists():
        raise FileExistsError(f"Formal output already exists: {resolved}")
    resolved.mkdir(parents=True, exist_ok=False)
    return resolved


def _write_manifest(output_dir: Path) -> Path:
    manifest_path = output_dir / "artifact_manifest.json"
    rows = []
    for path in sorted(output_dir.rglob("*")):
        if path.is_file() and path != manifest_path:
            rows.append(
                {
                    "path": path.relative_to(output_dir).as_posix(),
                    "bytes": int(path.stat().st_size),
                    "sha256": _sha256(path),
                }
            )
    payload = {
        "schema": "trkh_hamburger_nmf_surface_a0_artifact_manifest_v1",
        "files": rows,
    }
    _write_json(manifest_path, payload)
    return manifest_path


def _verify_manifest(output_dir: Path) -> Dict[str, object]:
    manifest_path = output_dir / "artifact_manifest.json"
    payload = json.loads(manifest_path.read_text(encoding="utf-8"))
    expected = {row["path"]: row for row in payload["files"]}
    observed_paths = {
        path.relative_to(output_dir).as_posix()
        for path in output_dir.rglob("*")
        if path.is_file() and path != manifest_path
    }
    if set(expected) != observed_paths:
        raise ValueError("Artifact manifest file set differs from disk")
    for relative, row in expected.items():
        path = output_dir / relative
        if int(path.stat().st_size) != int(row["bytes"]) or _sha256(path) != row["sha256"]:
            raise ValueError(f"Artifact manifest payload differs: {relative}")
    return {
        "passed": True,
        "payload_count": len(expected),
        "manifest_sha256": _sha256(manifest_path),
    }


def _prediction_rows(
    *,
    cohort: Sequence[CleanTrainRow],
    scores: Mapping[str, np.ndarray],
    actions: Mapping[str, np.ndarray],
) -> List[Dict[str, object]]:
    rows = []
    for position, source in enumerate(cohort):
        row: Dict[str, object] = {
            "position": int(position),
            "sample_index": int(source.sample_index),
            "source_stem": source.source_stem,
            "fold": int(source.fold),
            "target": int(source.target),
        }
        for role in ROLE_NAMES:
            row[f"{role}_score"] = float(scores[role][position])
            row[f"{role}_accept"] = int(actions[role][position])
        rows.append(row)
    return rows


def _fold_summary_rows(analysis: Mapping[str, object]) -> List[Dict[str, object]]:
    metrics = analysis["metrics"]
    rows = []
    for fold in FOLDS:
        row: Dict[str, object] = {"fold": int(fold)}
        for role in ROLE_NAMES:
            fold_row = metrics[role]["per_fold"][fold]
            for key in (
                "rows",
                "auroc",
                "tp_retention",
                "fp_rejection",
                "tp_breaks",
                "fp_rejects",
                "corrections",
                "harms",
            ):
                row[f"{role}_{key}"] = fold_row[key]
        rows.append(row)
    return rows


def _descriptor_mapping(bank: Mapping[str, object]) -> Dict[str, np.ndarray]:
    descriptors = bank["descriptors"]
    return {
        role: np.asarray(descriptors[role], dtype=np.float64)
        for role in (
            "raw_surface_control",
            "svd_rank8_control",
            "nmf_candidate",
            "nmf_seed_repeat",
        )
    }


def _cached_descriptor_mapping(cache: Mapping[str, np.ndarray]) -> Dict[str, np.ndarray]:
    return {
        role: np.asarray(cache[role], dtype=np.float64)
        for role in (
            "raw_surface_control",
            "svd_rank8_control",
            "nmf_candidate",
            "nmf_seed_repeat",
        )
    }


def _decomposition_replay_differences(
    persisted: Mapping[str, np.ndarray], replay: Mapping[str, object]
) -> Dict[str, float]:
    descriptors = _descriptor_mapping(replay)
    differences = {
        f"descriptor_{role}": float(
            np.max(
                np.abs(
                    np.asarray(persisted[role], dtype=np.float64)
                    - descriptors[role]
                )
            )
        )
        for role in descriptors
    }
    for key in ("candidate_objectives", "repeat_objectives", "factor_minima"):
        differences[key] = float(
            np.max(
                np.abs(
                    np.asarray(persisted[key], dtype=np.float64)
                    - np.asarray(replay[key], dtype=np.float64)
                )
            )
        )
    return differences


def _descriptor_statistics_payload(
    descriptors: Mapping[str, np.ndarray]
) -> Dict[str, object]:
    return {
        role: _descriptor_statistics(descriptors[role])
        for role in (
            "raw_surface_control",
            "svd_rank8_control",
            "nmf_candidate",
            "nmf_seed_repeat",
        )
    }


def _read_assigned_positions(path: Path, rows: int) -> np.ndarray:
    assigned = np.full(rows, -1, dtype=np.int64)
    with path.open("r", encoding="utf-8", newline="") as handle:
        for row in csv.DictReader(handle):
            assigned[int(row["destination_position"])] = int(row["source_position"])
    if bool((assigned < 0).any()) or np.unique(assigned).size != rows:
        raise ValueError("Persisted source-derangement assignment is incomplete")
    return assigned


def _load_prediction_scores(path: Path) -> Tuple[Dict[str, np.ndarray], Dict[str, np.ndarray]]:
    with path.open("r", encoding="utf-8", newline="") as handle:
        rows = list(csv.DictReader(handle))
    scores = {
        role: np.asarray([float(row[f"{role}_score"]) for row in rows], dtype=np.float64)
        for role in ROLE_NAMES
    }
    actions = {
        role: np.asarray([int(row[f"{role}_accept"]) != 0 for row in rows], dtype=np.bool_)
        for role in ROLE_NAMES
    }
    return scores, actions


def _run_shift_condition(
    *,
    name: str,
    brightness: float,
    contrast: float,
    model: nn.Module,
    base_dataset: MangoYOLOCropDataset,
    transform,
    cohort: Sequence[CleanTrainRow],
    device: torch.device,
    semantics: Mapping[str, object],
    output_dir: Path,
    folds: np.ndarray,
    source_stems: Sequence[str],
    labels: np.ndarray,
    readout_states: Mapping[str, object],
    assigned: np.ndarray,
) -> Dict[str, object]:
    loader, loader_summary = _make_condition_loader(
        base_dataset=base_dataset,
        transform=transform,
        indices=[row.sample_index for row in cohort],
        brightness=brightness,
        contrast=contrast,
        batch_size=BATCH_SIZE,
        num_workers=NUM_WORKERS,
        context=f"hamburger_nmf_surface_a0_{name}",
    )
    extraction = extract_condition(
        model=model,
        loader=loader,
        loader_summary=loader_summary,
        cohort=cohort,
        device=device,
        semantics=semantics,
        condition=name,
        review_indices=_review_indices(cohort),
    )
    token_path = output_dir / f"token_cache_{name}.npz"
    decomposition_path = output_dir / f"decomposition_{name}.npz"
    _save_token_cache(token_path, extraction)
    bank = build_decomposition_bank(
        extraction["projected_tokens"],
        extraction["counts"],
        extraction["sample_indices"],
        oracle=False,
        progress_label=f"hamburger_nmf_decomposition_{name}",
    )
    _save_decomposition_cache(decomposition_path, bank)
    descriptors = _descriptor_mapping(bank)
    features, observed_assigned, mapping = build_readout_features(
        logits=extraction["logits"],
        probabilities=extraction["probabilities"],
        descriptors=descriptors,
        folds=folds,
        source_stems=source_stems,
        sample_indices=extraction["sample_indices"],
        assigned=assigned,
    )
    if mapping or not np.array_equal(observed_assigned, assigned):
        raise RuntimeError("Lighting condition changed the clean source mapping")
    scores, actions = apply_readout_states(features, folds, readout_states)
    targets = np.asarray(extraction["targets"], dtype=np.int64)
    analysis = build_condition_analysis(
        labels=labels,
        targets=targets,
        folds=folds,
        scores=scores,
        actions=actions,
        probabilities=extraction["probabilities"],
    )
    prediction_path = output_dir / f"oof_predictions_{name}.csv"
    fold_path = output_dir / f"fold_summary_{name}.csv"
    descriptor_path = output_dir / f"descriptor_statistics_{name}.json"
    _write_csv(prediction_path, _prediction_rows(cohort=cohort, scores=scores, actions=actions))
    _write_csv(fold_path, _fold_summary_rows(analysis))
    descriptor_statistics = _descriptor_statistics_payload(descriptors)
    _write_json(descriptor_path, descriptor_statistics)
    return {
        "condition": name,
        "brightness": float(brightness),
        "contrast": float(contrast),
        "analysis": analysis,
        "runtime": extraction["runtime"],
        "token_statistics": extraction["token_statistics"],
        "decomposition_diagnostics": bank["diagnostics"],
        "descriptor_statistics": descriptor_statistics,
        "artifacts": {
            "token_cache": token_path.name,
            "decomposition": decomposition_path.name,
            "predictions": prediction_path.name,
            "fold_summary": fold_path.name,
            "descriptor_statistics": descriptor_path.name,
        },
    }


def preflight(args: argparse.Namespace) -> Dict[str, object]:
    _validate_locked_args(args)
    output_dir = Path(args.output_dir).expanduser().resolve()
    if output_dir.exists():
        raise FileExistsError(f"Preflight output must not exist: {output_dir}")
    provenance = verify_locked_inputs(args)
    engineering = engineering_checks()
    if not bool(engineering["passed"]):
        raise RuntimeError(f"Hamburger-NMF engineering checks failed: {engineering}")
    return {
        "mode": "hamburger_nmf_surface_a0_preflight",
        "output_created": False,
        "model_loaded": False,
        "dataset_pixels_loaded": False,
        "validation_data_used": False,
        "test_data_used": False,
        "locked_inputs_verified": True,
        "engineering": engineering,
        "repo_state_observed_not_gated_until_formal": _repo_state(),
        "provenance": provenance,
    }


def engineering_forward(args: argparse.Namespace) -> Dict[str, object]:
    _validate_locked_args(args)
    if not torch.cuda.is_available():
        raise RuntimeError("CUDA is unavailable for Hamburger-NMF engineering forward")
    provenance = verify_locked_inputs(args)
    device = torch.device("cuda")
    model, checkpoint, _ = load_model(Path(provenance["files"]["keeper"]["path"]), device)
    all_rows = _read_clean_train_rows(Path(provenance["files"]["cidt_predictions"]["path"]))
    cohort = _cohort_from_rows(all_rows)[:2]
    base_dataset, transform, dataset_summary = _build_dataset(
        checkpoint, all_rows, Path(provenance["files"]["data_yaml"]["path"])
    )
    loader, loader_summary = _make_condition_loader(
        base_dataset=base_dataset,
        transform=transform,
        indices=[row.sample_index for row in cohort],
        brightness=1.0,
        contrast=1.0,
        batch_size=BATCH_SIZE,
        num_workers=NUM_WORKERS,
        context="hamburger_nmf_surface_a0_engineering_forward",
    )
    extraction = extract_condition(
        model=model,
        loader=loader,
        loader_summary=loader_summary,
        cohort=cohort,
        device=device,
        semantics=dataset_summary["semantics"],
        condition="engineering_clean",
        review_indices=[row.sample_index for row in cohort],
    )
    bank = build_decomposition_bank(
        extraction["projected_tokens"],
        extraction["counts"],
        extraction["sample_indices"],
        oracle=False,
    )
    return {
        "mode": "hamburger_nmf_surface_a0_engineering_forward",
        "passed": bool(
            extraction["runtime"]["model_state_before_sha256"]
            == extraction["runtime"]["model_state_after_sha256"]
            and extraction["token_statistics"]["all_at_least_16"]
            and bank["diagnostics"]["candidate_objectives_monotonic_all_rows"]
            and bank["diagnostics"]["repeat_objectives_monotonic_all_rows"]
            and bank["diagnostics"]["all_nonnegative_within_1e_12"]
        ),
        "rows": len(cohort),
        "token_statistics": extraction["token_statistics"],
        "runtime": extraction["runtime"],
        "decomposition_diagnostics": bank["diagnostics"],
        "validation_data_used": False,
        "test_data_used": False,
        "output_created": False,
    }


def run_audit(args: argparse.Namespace) -> Dict[str, object]:
    _validate_locked_args(args)
    if os.environ.get("TRKH_HAMBURGER_NMF_A0_PREFLIGHT") != "passed":
        raise RuntimeError(
            "Formal Hamburger-NMF A0 must use the locked PowerShell preflight"
        )
    if not torch.cuda.is_available():
        raise RuntimeError("CUDA is unavailable for formal Hamburger-NMF A0")
    provenance = verify_locked_inputs(args)
    repo_state = _repo_state()
    if not bool(repo_state["tracked_worktree_clean"]) or not bool(
        repo_state["head_matches_upstream"]
    ):
        raise ValueError(
            f"Formal Hamburger-NMF A0 requires clean pushed tracked state: {repo_state}"
        )
    output_dir = _prepare_output_dir(args.output_dir)
    set_seed(SEED, deterministic=True)
    torch.backends.cuda.matmul.allow_tf32 = False
    torch.backends.cudnn.allow_tf32 = True
    engineering = engineering_checks()
    if not bool(engineering["passed"]):
        raise RuntimeError("Hamburger-NMF engineering checks changed before formal run")
    device = torch.device("cuda")
    model, checkpoint, class_names = load_model(
        Path(provenance["files"]["keeper"]["path"]), device
    )
    all_rows = _read_clean_train_rows(
        Path(provenance["files"]["cidt_predictions"]["path"])
    )
    cohort = _cohort_from_rows(all_rows)
    base_dataset, transform, dataset_summary = _build_dataset(
        checkpoint, all_rows, Path(provenance["files"]["data_yaml"]["path"])
    )
    folds = np.asarray([row.fold for row in cohort], dtype=np.int64)
    source_stems = [row.source_stem for row in cohort]
    expected_indices = np.asarray([row.sample_index for row in cohort], dtype=np.int64)
    expected_targets = np.asarray([row.target for row in cohort], dtype=np.int64)
    review_indices = _review_indices(cohort)
    clean_loader, clean_loader_summary = _make_condition_loader(
        base_dataset=base_dataset,
        transform=transform,
        indices=expected_indices.tolist(),
        brightness=1.0,
        contrast=1.0,
        batch_size=BATCH_SIZE,
        num_workers=NUM_WORKERS,
        context="hamburger_nmf_surface_a0_clean",
    )
    clean_extraction = extract_condition(
        model=model,
        loader=clean_loader,
        loader_summary=clean_loader_summary,
        cohort=cohort,
        device=device,
        semantics=dataset_summary["semantics"],
        condition="clean",
        review_indices=review_indices,
    )
    if not np.array_equal(clean_extraction["sample_indices"], expected_indices):
        raise ValueError("Formal clean extraction changed cohort order")
    token_path = output_dir / "token_cache_clean.npz"
    decomposition_path = output_dir / "decomposition_clean.npz"
    _save_token_cache(token_path, clean_extraction)
    clean_bank = build_decomposition_bank(
        clean_extraction["projected_tokens"],
        clean_extraction["counts"],
        clean_extraction["sample_indices"],
        oracle=False,
        progress_label="hamburger_nmf_decomposition_clean",
    )
    _save_decomposition_cache(decomposition_path, clean_bank)
    clean_descriptors = _descriptor_mapping(clean_bank)
    clean_features, assigned, derangement_mapping = build_readout_features(
        logits=clean_extraction["logits"],
        probabilities=clean_extraction["probabilities"],
        descriptors=clean_descriptors,
        folds=folds,
        source_stems=source_stems,
        sample_indices=expected_indices,
    )
    derangement_path = output_dir / "source_derangement_clean.csv"
    _write_csv(derangement_path, derangement_mapping)

    if not np.array_equal(clean_extraction["targets"], expected_targets):
        raise ValueError("Formal clean extraction changed cohort targets")
    labels = (expected_targets == FOCUS_CLASS).astype(np.int64)
    expected_probabilities = np.asarray(
        [row.keeper_probabilities for row in cohort], dtype=np.float64
    )
    clean_probabilities = np.asarray(clean_extraction["probabilities"], dtype=np.float64)
    cidt_probability_error = float(
        np.max(np.abs(clean_probabilities - expected_probabilities))
    )
    cidt_argmax_exact = bool(
        np.array_equal(
            clean_probabilities.argmax(axis=1),
            np.asarray([row.keeper_prediction for row in cohort], dtype=np.int64),
        )
    )
    scores, actions, readout_states = fit_clean_oof_readouts(
        clean_features, labels, folds, source_stems
    )
    descriptor_statistics = _descriptor_statistics_payload(clean_descriptors)
    clean_analysis = build_clean_analysis(
        labels=labels,
        targets=expected_targets,
        folds=folds,
        scores=scores,
        actions=actions,
        probabilities=clean_probabilities,
        candidate_descriptor_stats=descriptor_statistics["nmf_candidate"],
        decomposition_diagnostics=clean_bank["diagnostics"],
    )
    readout_path = output_dir / "readout_state_clean.json"
    prediction_path = output_dir / "oof_predictions_clean.csv"
    fold_path = output_dir / "fold_summary_clean.csv"
    descriptor_path = output_dir / "descriptor_statistics_clean.json"
    _write_json(readout_path, readout_states)
    _write_csv(prediction_path, _prediction_rows(cohort=cohort, scores=scores, actions=actions))
    _write_csv(fold_path, _fold_summary_rows(clean_analysis))
    _write_json(descriptor_path, descriptor_statistics)
    contact_path = output_dir / "hamburger_nmf_contact_sheet_clean.png"
    contact_sheet = render_contact_sheet(
        contact_path,
        token_cache=_load_token_cache(token_path),
        cohort=cohort,
    )

    persisted_tokens = _load_token_cache(token_path)
    persisted_decomposition = _load_decomposition_cache(decomposition_path)
    oracle_bank = build_decomposition_bank(
        persisted_tokens["projected_tokens"],
        persisted_tokens["counts"],
        persisted_tokens["sample_indices"],
        oracle=True,
        progress_label="hamburger_nmf_internal_numpy_replay_clean",
    )
    decomposition_differences = _decomposition_replay_differences(
        persisted_decomposition, oracle_bank
    )
    oracle_descriptors = _descriptor_mapping(oracle_bank)
    oracle_features, oracle_assigned, _ = build_readout_features(
        logits=persisted_tokens["logits"],
        probabilities=persisted_tokens["probabilities"],
        descriptors=oracle_descriptors,
        folds=folds,
        source_stems=source_stems,
        sample_indices=persisted_tokens["sample_indices"],
        assigned=assigned,
    )
    oracle_scores, oracle_actions, oracle_states = fit_clean_oof_readouts(
        oracle_features, labels, folds, source_stems
    )
    applied_scores, applied_actions = apply_readout_states(
        oracle_features, folds, readout_states
    )
    oracle_descriptor_statistics = _descriptor_statistics_payload(oracle_descriptors)
    oracle_analysis = build_clean_analysis(
        labels=labels,
        targets=expected_targets,
        folds=folds,
        scores=oracle_scores,
        actions=oracle_actions,
        probabilities=persisted_tokens["probabilities"],
        candidate_descriptor_stats=oracle_descriptor_statistics["nmf_candidate"],
        decomposition_diagnostics=oracle_bank["diagnostics"],
    )
    internal_replay = {
        "decomposition_maximum_abs_differences": decomposition_differences,
        "decomposition_maximum_abs_difference": max(decomposition_differences.values()),
        "analysis_maximum_numeric_difference": _recursive_numeric_difference(
            clean_analysis, oracle_analysis
        ),
        "readout_state_maximum_numeric_difference": _recursive_numeric_difference(
            readout_states, oracle_states
        ),
        "refit_score_maximum_abs_difference": _maximum_role_difference(
            scores, oracle_scores
        ),
        "applied_score_maximum_abs_difference": _maximum_role_difference(
            scores, applied_scores
        ),
        "refit_actions_exact": all(
            np.array_equal(actions[role], oracle_actions[role]) for role in ROLE_NAMES
        ),
        "applied_actions_exact": all(
            np.array_equal(actions[role], applied_actions[role]) for role in ROLE_NAMES
        ),
        "derangement_assignment_exact": np.array_equal(assigned, oracle_assigned),
    }
    internal_replay["passed"] = bool(
        float(internal_replay["decomposition_maximum_abs_difference"]) <= 1e-10
        and float(internal_replay["analysis_maximum_numeric_difference"]) <= 1e-10
        and float(internal_replay["readout_state_maximum_numeric_difference"]) <= 1e-10
        and float(internal_replay["refit_score_maximum_abs_difference"]) <= 1e-10
        and float(internal_replay["applied_score_maximum_abs_difference"]) <= 1e-10
        and bool(internal_replay["refit_actions_exact"])
        and bool(internal_replay["applied_actions_exact"])
        and bool(internal_replay["derangement_assignment_exact"])
    )

    descriptor_hashes = {
        role: _array_sha256(value) for role, value in clean_descriptors.items()
    }
    role_hashes_distinct = len(set(descriptor_hashes.values())) == len(descriptor_hashes)
    source_fold_overlap = 0
    for held_fold in FOLDS:
        held_sources = set(np.asarray(source_stems)[folds == held_fold].tolist())
        fit_sources = set(np.asarray(source_stems)[folds != held_fold].tolist())
        source_fold_overlap += len(held_sources.intersection(fit_sources))
    effective_workers = int(
        clean_extraction["runtime"]["loader"].get("effective_num_workers", -1)
    )
    structural_gates = {
        "locked_hashes_verified": True,
        "official_repository_exact_and_clean": bool(
            provenance["official_repository"]["worktree_clean"]
        ),
        "protected_untracked_identity_exact": bool(
            provenance["protected_untracked"]["passed"]
        ),
        "repo_clean_and_pushed": bool(repo_state["tracked_worktree_clean"])
        and bool(repo_state["head_matches_upstream"]),
        "launcher_preflight_attested": os.environ.get(
            "TRKH_HAMBURGER_NMF_A0_PREFLIGHT"
        )
        == "passed",
        "train_split_only": bool(dataset_summary["train_paths_only"]),
        "dataset_mapping_exact": bool(dataset_summary["paths_exact"]),
        "cohort_rows_order_support_exact": len(cohort) == EXPECTED_COHORT_ROWS
        and int(labels.sum()) == EXPECTED_POSITIVES
        and int((labels == 0).sum()) == EXPECTED_NEGATIVES
        and _cohort_index_sha256(expected_indices.tolist())
        == EXPECTED_ORDERED_INDEX_SHA256,
        "five_source_folds_and_zero_overlap": set(folds.tolist()) == set(FOLDS)
        and source_fold_overlap == 0,
        "cidt_probability_error_le_3e_5": cidt_probability_error
        <= MAX_CIDT_PROBABILITY_ERROR,
        "cidt_argmax_exact": cidt_argmax_exact,
        "projection_exact": bool(engineering["projection"]["passed"]),
        "surface_token_support_exact": bool(
            clean_extraction["token_statistics"]["all_at_least_16"]
        )
        and bool(clean_extraction["token_statistics"]["all_at_most_256"])
        and bool(clean_extraction["token_statistics"]["indices_sorted_unique"])
        and float(clean_extraction["token_statistics"]["selected_prior_minimum"])
        >= BBOX_PRIOR_THRESHOLD,
        "role_feature_dimensions_exact": _feature_dimensions(clean_features)
        == {
            "base_only": BASE_DIM,
            "raw_surface_control": ROLE_DIM,
            "svd_rank8_control": ROLE_DIM,
            "nmf_candidate": ROLE_DIM,
            "nmf_seed_repeat": ROLE_DIM,
            "source_deranged": ROLE_DIM,
        },
        "candidate_and_controls_distinct": role_hashes_distinct,
        "all_30_readouts_converged_without_retry": _readout_states_converged(
            readout_states
        ),
        "model_state_bit_exact": clean_extraction["runtime"][
            "model_state_before_sha256"
        ]
        == clean_extraction["runtime"]["model_state_after_sha256"],
        "requested_workers_2_effective_workers_2": NUM_WORKERS == 2
        and effective_workers == 2,
        "peak_cuda_allocation_le_3_5_gib": int(
            clean_extraction["runtime"]["peak_cuda_bytes"]
        )
        <= int(MAX_PEAK_CUDA_GIB * 1024**3),
        "production_uses_only_local_rng": bool(
            clean_bank["diagnostics"]["local_rng_only"]
        ),
        "internal_numpy_replay_passed": bool(internal_replay["passed"]),
        "external_second_process_replay_exact": False,
        "validation_data_unused": True,
        "test_data_unused": True,
        "raw_dataset_unmodified": True,
    }
    stage_b_preconditions = {
        key: value
        for key, value in structural_gates.items()
        if key != "external_second_process_replay_exact"
    }
    stage_b_authorized = bool(
        clean_analysis["mechanism_gates_passed"]
        and all(stage_b_preconditions.values())
    )
    condition_results: Dict[str, object] = {}
    if stage_b_authorized:
        for name, brightness, contrast in CONDITIONS:
            condition_results[name] = _run_shift_condition(
                name=name,
                brightness=brightness,
                contrast=contrast,
                model=model,
                base_dataset=base_dataset,
                transform=transform,
                cohort=cohort,
                device=device,
                semantics=dataset_summary["semantics"],
                output_dir=output_dir,
                folds=folds,
                source_stems=source_stems,
                labels=labels,
                readout_states=readout_states,
                assigned=assigned,
            )
    stage_b_passed = bool(
        stage_b_authorized
        and len(condition_results) == len(CONDITIONS)
        and all(bool(value["analysis"]["passed"]) for value in condition_results.values())
    )
    summary: Dict[str, object] = {
        "mode": MODE,
        "method": METHOD,
        "status": (
            "awaiting_external_replay"
            if stage_b_passed
            else "rejected_clean_automated_gate"
            if not stage_b_authorized
            else "rejected_lighting_automated_gate"
        ),
        "class_names": class_names,
        "cohort": {
            "rows": len(cohort),
            "positives": int(labels.sum()),
            "negatives": int((labels == 0).sum()),
            "ordered_sample_index_sha256": _cohort_index_sha256(expected_indices.tolist()),
        },
        "provenance": provenance,
        "repo_state": repo_state,
        "dataset": dataset_summary,
        "engineering": engineering,
        "clean_runtime": clean_extraction["runtime"],
        "clean_token_statistics": clean_extraction["token_statistics"],
        "clean_decomposition_diagnostics": clean_bank["diagnostics"],
        "descriptor_statistics": descriptor_statistics,
        "descriptor_sha256": descriptor_hashes,
        "source_derangement": {
            "rows": len(derangement_mapping),
            "mapping_sha256": _records_sha256(derangement_mapping),
            "assignment_sha256": _array_sha256(assigned),
            "different_source_all_rows": all(
                row["destination_source_stem"] != row["descriptor_source_stem"]
                for row in derangement_mapping
            ),
        },
        "cidt_replay": {
            "maximum_probability_error": cidt_probability_error,
            "argmax_exact": cidt_argmax_exact,
        },
        "clean_analysis": clean_analysis,
        "internal_replay": internal_replay,
        "structural_gates": structural_gates,
        "structural_gates_passed": False,
        "stage_b_authorized_by_clean_gate": stage_b_authorized,
        "stage_b_executed": bool(condition_results),
        "stage_b_results": condition_results,
        "stage_b_passed": stage_b_passed,
        "external_replay": {"required": True, "completed": False, "passed": False},
        "visual_review": {"required": True, "completed": False, "passed": False},
        "automated_gate_passed": False,
        "a0_passed": False,
        "trainer_integration_authorized": False,
        "matched_short_smoke_authorized": False,
        "full_train_authorized": False,
        "validation_access_authorized": False,
        "test_access_authorized": False,
        "current_command_update_authorized": False,
        "contact_sheet": contact_sheet,
        "artifacts": {
            "token_cache_clean": token_path.name,
            "decomposition_clean": decomposition_path.name,
            "derangement_clean": derangement_path.name,
            "readout_state_clean": readout_path.name,
            "predictions_clean": prediction_path.name,
            "fold_summary_clean": fold_path.name,
            "descriptor_statistics_clean": descriptor_path.name,
        },
    }
    summary_path = output_dir / "summary.json"
    _write_json(summary_path, summary)
    manifest_path = _write_manifest(output_dir)
    return {
        "mode": MODE,
        "status": summary["status"],
        "clean_mechanism_gates_passed": bool(
            clean_analysis["mechanism_gates_passed"]
        ),
        "stage_b_executed": bool(condition_results),
        "stage_b_passed": stage_b_passed,
        "summary_path": str(summary_path),
        "summary_sha256": _sha256(summary_path),
        "manifest_path": str(manifest_path),
        "manifest_sha256": _sha256(manifest_path),
        "full_train_authorized": False,
        "current_command_update_authorized": False,
    }


def replay_summary(summary_path: Path) -> Dict[str, object]:
    resolved = Path(summary_path).expanduser().resolve()
    summary = json.loads(resolved.read_text(encoding="utf-8"))
    if summary.get("mode") != MODE:
        raise ValueError("Summary is not the locked Hamburger-NMF A0 audit")
    manifest_before = _verify_manifest(resolved.parent)
    all_rows = _read_clean_train_rows(
        Path(summary["provenance"]["files"]["cidt_predictions"]["path"])
    )
    cohort = _cohort_from_rows(all_rows)
    folds = np.asarray([row.fold for row in cohort], dtype=np.int64)
    source_stems = [row.source_stem for row in cohort]
    targets = np.asarray([row.target for row in cohort], dtype=np.int64)
    labels = (targets == FOCUS_CLASS).astype(np.int64)
    assigned = _read_assigned_positions(
        resolved.parent / summary["artifacts"]["derangement_clean"], len(cohort)
    )
    readout_states = json.loads(
        (resolved.parent / summary["artifacts"]["readout_state_clean"]).read_text(
            encoding="utf-8"
        )
    )
    token_cache = _load_token_cache(
        resolved.parent / summary["artifacts"]["token_cache_clean"]
    )
    decomposition_cache = _load_decomposition_cache(
        resolved.parent / summary["artifacts"]["decomposition_clean"]
    )
    oracle_bank = build_decomposition_bank(
        token_cache["projected_tokens"],
        token_cache["counts"],
        token_cache["sample_indices"],
        oracle=True,
        progress_label="hamburger_nmf_external_numpy_replay_clean",
    )
    decomposition_differences = _decomposition_replay_differences(
        decomposition_cache, oracle_bank
    )
    descriptors = _descriptor_mapping(oracle_bank)
    features, replay_assigned, _ = build_readout_features(
        logits=token_cache["logits"],
        probabilities=token_cache["probabilities"],
        descriptors=descriptors,
        folds=folds,
        source_stems=source_stems,
        sample_indices=token_cache["sample_indices"],
        assigned=assigned,
    )
    refit_scores, refit_actions, refit_states = fit_clean_oof_readouts(
        features, labels, folds, source_stems
    )
    applied_scores, applied_actions = apply_readout_states(
        features, folds, readout_states
    )
    persisted_scores, persisted_actions = _load_prediction_scores(
        resolved.parent / summary["artifacts"]["predictions_clean"]
    )
    descriptor_statistics = _descriptor_statistics_payload(descriptors)
    clean_analysis = build_clean_analysis(
        labels=labels,
        targets=targets,
        folds=folds,
        scores=refit_scores,
        actions=refit_actions,
        probabilities=token_cache["probabilities"],
        candidate_descriptor_stats=descriptor_statistics["nmf_candidate"],
        decomposition_diagnostics=oracle_bank["diagnostics"],
    )
    clean_checks = {
        "decomposition_maximum_abs_difference": max(
            decomposition_differences.values()
        ),
        "analysis_maximum_numeric_difference": _recursive_numeric_difference(
            summary["clean_analysis"], clean_analysis
        ),
        "readout_state_maximum_numeric_difference": _recursive_numeric_difference(
            readout_states, refit_states
        ),
        "persisted_score_maximum_abs_difference": _maximum_role_difference(
            persisted_scores, refit_scores
        ),
        "applied_score_maximum_abs_difference": _maximum_role_difference(
            refit_scores, applied_scores
        ),
        "persisted_actions_exact": all(
            np.array_equal(persisted_actions[role], refit_actions[role])
            for role in ROLE_NAMES
        ),
        "applied_actions_exact": all(
            np.array_equal(applied_actions[role], refit_actions[role])
            for role in ROLE_NAMES
        ),
        "derangement_assignment_exact": np.array_equal(assigned, replay_assigned),
    }
    clean_passed = bool(
        float(clean_checks["decomposition_maximum_abs_difference"]) <= 1e-10
        and float(clean_checks["analysis_maximum_numeric_difference"]) <= 1e-10
        and float(clean_checks["readout_state_maximum_numeric_difference"]) <= 1e-10
        and float(clean_checks["persisted_score_maximum_abs_difference"]) <= 1e-10
        and float(clean_checks["applied_score_maximum_abs_difference"]) <= 1e-10
        and bool(clean_checks["persisted_actions_exact"])
        and bool(clean_checks["applied_actions_exact"])
        and bool(clean_checks["derangement_assignment_exact"])
    )
    condition_replay: Dict[str, object] = {}
    for name, _, _ in CONDITIONS:
        if name not in summary["stage_b_results"]:
            continue
        condition_record = summary["stage_b_results"][name]
        artifacts = condition_record["artifacts"]
        condition_tokens = _load_token_cache(resolved.parent / artifacts["token_cache"])
        condition_decomposition = _load_decomposition_cache(
            resolved.parent / artifacts["decomposition"]
        )
        condition_oracle = build_decomposition_bank(
            condition_tokens["projected_tokens"],
            condition_tokens["counts"],
            condition_tokens["sample_indices"],
            oracle=True,
            progress_label=f"hamburger_nmf_external_numpy_replay_{name}",
        )
        condition_differences = _decomposition_replay_differences(
            condition_decomposition, condition_oracle
        )
        condition_features, condition_assigned, _ = build_readout_features(
            logits=condition_tokens["logits"],
            probabilities=condition_tokens["probabilities"],
            descriptors=_descriptor_mapping(condition_oracle),
            folds=folds,
            source_stems=source_stems,
            sample_indices=condition_tokens["sample_indices"],
            assigned=assigned,
        )
        condition_scores, condition_actions = apply_readout_states(
            condition_features, folds, readout_states
        )
        recorded_scores, recorded_actions = _load_prediction_scores(
            resolved.parent / artifacts["predictions"]
        )
        condition_analysis = build_condition_analysis(
            labels=labels,
            targets=targets,
            folds=folds,
            scores=condition_scores,
            actions=condition_actions,
            probabilities=condition_tokens["probabilities"],
        )
        checks = {
            "decomposition_maximum_abs_difference": max(
                condition_differences.values()
            ),
            "analysis_maximum_numeric_difference": _recursive_numeric_difference(
                condition_record["analysis"], condition_analysis
            ),
            "persisted_score_maximum_abs_difference": _maximum_role_difference(
                recorded_scores, condition_scores
            ),
            "persisted_actions_exact": all(
                np.array_equal(recorded_actions[role], condition_actions[role])
                for role in ROLE_NAMES
            ),
            "derangement_assignment_exact": np.array_equal(
                assigned, condition_assigned
            ),
        }
        checks["passed"] = bool(
            float(checks["decomposition_maximum_abs_difference"]) <= 1e-10
            and float(checks["analysis_maximum_numeric_difference"]) <= 1e-10
            and float(checks["persisted_score_maximum_abs_difference"]) <= 1e-10
            and bool(checks["persisted_actions_exact"])
            and bool(checks["derangement_assignment_exact"])
        )
        condition_replay[name] = checks
    condition_set_exact = set(condition_replay) == set(summary["stage_b_results"])
    external_passed = bool(
        clean_passed
        and condition_set_exact
        and all(bool(value["passed"]) for value in condition_replay.values())
    )
    replay = {
        "required": True,
        "completed": True,
        "second_process": True,
        "passed": external_passed,
        "clean": clean_checks,
        "conditions": condition_replay,
        "condition_set_exact": condition_set_exact,
        "manifest_sha256_before_replay": manifest_before["manifest_sha256"],
    }
    summary["external_replay"] = replay
    summary["structural_gates"]["external_second_process_replay_exact"] = external_passed
    structural_passed = all(
        bool(value) for value in summary["structural_gates"].values()
    )
    automated_passed = bool(
        structural_passed
        and summary["clean_analysis"]["mechanism_gates_passed"]
        and summary["stage_b_passed"]
    )
    summary["structural_gates_passed"] = structural_passed
    summary["automated_gate_passed"] = automated_passed
    summary["status"] = (
        "awaiting_visual_review"
        if automated_passed
        else "rejected_clean_automated_gate"
        if not summary["stage_b_authorized_by_clean_gate"]
        else "rejected_lighting_or_replay_gate"
    )
    _write_json(resolved, summary)
    manifest_after = _write_manifest(resolved.parent)
    return {
        "mode": "hamburger_nmf_surface_a0_external_replay",
        "passed": external_passed,
        "structural_gates_passed": structural_passed,
        "automated_gate_passed": automated_passed,
        "status": summary["status"],
        "summary_sha256": _sha256(resolved),
        "manifest_sha256": _sha256(manifest_after),
        "full_train_authorized": False,
        "current_command_update_authorized": False,
    }


def finalize_visual_review(
    summary_path: Path,
    *,
    result: str,
    expected_summary_sha256: str,
) -> Dict[str, object]:
    resolved = Path(summary_path).expanduser().resolve()
    observed_summary_sha = _sha256(resolved)
    if observed_summary_sha != str(expected_summary_sha256).strip().casefold():
        raise ValueError(
            "Visual-review summary SHA differs: "
            f"expected={expected_summary_sha256}, observed={observed_summary_sha}"
        )
    manifest_before = _verify_manifest(resolved.parent)
    summary = json.loads(resolved.read_text(encoding="utf-8"))
    if summary.get("mode") != MODE:
        raise ValueError("Visual-review target is not Hamburger-NMF A0")
    if not bool(summary.get("external_replay", {}).get("completed", False)):
        raise ValueError("Visual review requires completed second-process replay")
    contact_path = Path(summary["contact_sheet"]["path"])
    contact_sha = _sha256(contact_path)
    if contact_sha != summary["contact_sheet"]["sha256"]:
        raise ValueError("Visual-review contact sheet differs from summary lock")
    visual_passed = str(result) == "pass"
    automated_passed = bool(summary["automated_gate_passed"])
    a0_passed = bool(visual_passed and automated_passed)
    summary["visual_review"] = {
        "required": True,
        "completed": True,
        "passed": visual_passed,
        "decision": str(result),
        "reviewed_summary_sha256": observed_summary_sha,
        "contact_sheet_sha256": contact_sha,
        "manifest_sha256_before_review": manifest_before["manifest_sha256"],
        "cannot_rescue_automated_failure": True,
    }
    summary["a0_passed"] = a0_passed
    summary["trainer_integration_authorized"] = a0_passed
    summary["matched_short_smoke_authorized"] = a0_passed
    summary["full_train_authorized"] = False
    summary["validation_access_authorized"] = False
    summary["test_access_authorized"] = False
    summary["current_command_update_authorized"] = False
    summary["status"] = (
        "passed_a0_default_off_integration_and_matched_smoke_authorized"
        if a0_passed
        else "rejected_visual_gate"
        if automated_passed and not visual_passed
        else "rejected_automated_gate_visual_review_recorded"
    )
    _write_json(resolved, summary)
    manifest_after = _write_manifest(resolved.parent)
    return {
        "mode": "hamburger_nmf_surface_a0_visual_review",
        "decision": str(result),
        "automated_gate_passed": automated_passed,
        "a0_passed": a0_passed,
        "trainer_integration_authorized": a0_passed,
        "matched_short_smoke_authorized": a0_passed,
        "full_train_authorized": False,
        "current_command_update_authorized": False,
        "reviewed_summary_sha256": observed_summary_sha,
        "summary_sha256": _sha256(resolved),
        "manifest_sha256": _sha256(manifest_after),
    }


def main(argv: Optional[Sequence[str]] = None) -> None:
    args = parse_args(argv)
    if args.finalize_visual_review:
        if args.replay_summary is None or not args.expected_summary_sha256:
            raise ValueError(
                "Visual finalization requires --replay-summary and --expected-summary-sha256"
            )
        result = finalize_visual_review(
            args.replay_summary,
            result=args.finalize_visual_review,
            expected_summary_sha256=args.expected_summary_sha256,
        )
    elif args.replay_summary is not None:
        result = replay_summary(args.replay_summary)
    elif args.preflight_only:
        result = preflight(args)
    elif args.engineering_forward:
        result = engineering_forward(args)
    else:
        result = run_audit(args)
    print(json.dumps(to_serializable(result), indent=2, sort_keys=True), flush=True)


if __name__ == "__main__":
    main()
