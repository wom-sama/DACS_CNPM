from __future__ import annotations

import argparse
import csv
from functools import lru_cache
import hashlib
import json
import math
import os
from pathlib import Path
import subprocess
import time
from typing import Dict, List, Mapping, Optional, Sequence, Tuple
import warnings

import numpy as np
from PIL import Image, ImageDraw, ImageFont

os.environ.setdefault("CUBLAS_WORKSPACE_CONFIG", ":4096:8")

import torch
from sklearn.exceptions import ConvergenceWarning
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import roc_auc_score
from sklearn.preprocessing import StandardScaler
from torch import Tensor, nn
import torch.nn.functional as F

from trkh.core.config import to_serializable
from trkh.core.utils import set_seed
from trkh.evaluation.robustness_eval import _forward_classification_with_metadata
from trkh.inference.inference import load_checkpoint
from trkh.models.model import (
    classification_logits_from_features,
    extract_bbox_from_model_output,
)
from trkh.tools.audit_more_model_rebalancing_readiness import (
    CleanTrainRow,
    _ordered_index_sha256,
    _read_clean_train_rows,
)
from trkh.tools.audit_pixel_difference_stem_signal import (
    _build_dataset,
    _load_keeper_model,
    _make_condition_loader,
)


METHOD = "ielt_mhv_signal_a0"
SEED = 20260720
BATCH_SIZE = 32
NUM_WORKERS = 4
FOCUS_CLASS = 1
RESTRICTED_NEGATIVE_CLASSES = (0, 2, 4)
FIT_FOLDS = (1, 2, 3, 4)
EXPECTED_COHORT_ROWS = 607
EXPECTED_POSITIVES = 421
EXPECTED_NEGATIVES = 186
EXPECTED_FOLD_COUNTS = {
    1: {"tp": 112, "fp": 45},
    2: {"tp": 100, "fp": 48},
    3: {"tp": 101, "fp": 52},
    4: {"tp": 108, "fp": 41},
}
EXPECTED_ORDERED_INDEX_SHA256 = (
    "a2689d1be8579386eea9ef02a826822a5e7d78e2d29439e8947c1731ccc738bd"
)
LOCKED_DECLARATION_EXCEPTION = {
    "sample_index": 3657,
    "target": 2,
    "expected_prediction": 1,
    "observed_prediction": 2,
}
LOCKED_BBOX_BYTES_SHA256 = (
    "e9b2143c9dbc7c6f5bf3a38f80a43483bd436e3b5ce99116c3d5989f79892f6b"
)

GRID_SIZE = 16
PATCH_COUNT = GRID_SIZE * GRID_SIZE
BLOCK_COUNT = 8
HEAD_COUNT = 8
FEATURE_DIM = 256
PROJECTION_DIM = 32
ROLE_DIM = BLOCK_COUNT * PROJECTION_DIM * 2
ROLE_NAMES = (
    "cls_mean_smooth",
    "vote_raw",
    "vote_dephased_smooth",
    "mhv_smooth",
)
CONTROL_NAMES = ("base_only",) + ROLE_NAMES[:-1]
ALL_READOUT_ROLES = ("base_only",) + ROLE_NAMES
LAYER_QUOTAS = (5, 4, 3, 3, 3, 3, 4, 5)
EXPECTED_ENTERING_PATCH_COUNTS = (256, 256, 218, 218, 218, 167, 167, 167)
EXPECTED_AFTER_PATCH_COUNTS = (256, 218, 218, 218, 167, 167, 167, 167)
DEPHASE_OFFSETS = (
    (-3, -1),
    (-2, 2),
    (-1, -3),
    (1, 3),
    (2, -2),
    (3, 1),
    (-3, 3),
    (3, -3),
)
MHV_KERNEL = (
    (1, 2, 1),
    (2, 4, 2),
    (1, 2, 1),
)
CONDITIONS = (
    ("lighting_dim", 0.70, 0.90),
    ("lighting_bright", 1.25, 1.10),
    ("low_contrast", 1.00, 0.65),
)

READOUT_C = 0.1
READOUT_MAX_ITER = 4000
READOUT_TOLERANCE = 1e-9
MIN_FIT_TP_RETENTION = 0.97
MAX_HOOK_PROBABILITY_ERROR = 1e-6
MAX_PEAK_CUDA_GIB = 3.5

LOCKED_KEEPER_SHA256 = "1f49d577240c69dc63c30af70db52ec2aa9da65a17aef1c4b1c09ece6c482677"
LOCKED_LAUNCHER_ARGS_SHA256 = "908a05cf66b2a01162cae62e4ff2251eaae1297d31e70510144e4954159b7eff"
LOCKED_DATA_SHA256 = "716e33df24c63a9e9920f97b685199707fb84ab4c7154544f5dd9a3e00d884ef"
LOCKED_CIDT_SUMMARY_SHA256 = "d4891edf2963ab12385b7ce5bdc812ec3e19c5c098acd25c66eb557af541d7ad"
LOCKED_CIDT_PREDICTIONS_SHA256 = "2e0993752d58d99ea429bfefe1e2bfe6fa949e45aea1a26cc4bdfee97d4db21c"
LOCKED_PROTOCOL_SHA256 = "0e4984a003845312cf5725a98e4be7e220084037417ec24050734755a6665e58"
LOCKED_CURRENT_COMMAND_SHA256 = "36b9aa1a21b765829acf4c8321be147bd76297de4ccdb8a40e6dee8e37940faf"
LOCKED_COMMAND_HISTORY_SHA256 = "39bd2879ce66fddf36a953021ea1e40f8d9de6cb4334b9b825011b2b8dc98f53"
LOCKED_OFFICIAL_COMMIT = "b185111782d4a5f1e835060377022a7355bd5eec"
LOCKED_OFFICIAL_TREE = "713ed85495c3719a06f3a87fd2c4fd83dccfd050"
LOCKED_OFFICIAL_HASHES = {
    "models/IELT.py": "cab0b8b28caab9f315ec9b214be820f82fd482bbc76cb175f1223e47020ef7de",
    "models/build.py": "5d0cec610a1baec85d75c1b091c7a889d457882774297a478aeb91d235c6d042",
    "configs/cub.yaml": "0ab2f5c4eddd57e0d8792cd388c01669d357d6b3349743a2711b19c46dab8a81",
    "README.md": "e7a1ef914498835516aec066c02700dfd48d3dbc92d618006e13475000c13cea",
    "LICENSE": "0dd3b7c901d75b63f24f43955d0b121a4edc59ae78be7be753d355be492f9069",
    "main.py": "f127ae2ac569c0aa102567c599f6f2c0eb8df74330cc39e2b5c8430cfce3eaf6",
}

REPO_ROOT = Path(__file__).resolve().parents[2]
OFFICIAL_ROOT = Path(r"D:\DataAI\external_sources\official\ielt-tmm2023")
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
        description="Locked train-only IELT MHV information gate."
    )
    parser.add_argument(
        "--checkpoint",
        type=Path,
        default=REPO_ROOT
        / "runs"
        / "probe_v8_yolof_pairroute_teacherfocusbinary015_boundarydrop_bboxprior_120b_2e_20260701"
        / "checkpoints"
        / "best.pt",
    )
    parser.add_argument(
        "--launcher-args",
        type=Path,
        default=REPO_ROOT
        / "runs"
        / "probe_v8_yolof_pairroute_teacherfocusbinary015_boundarydrop_bboxprior_120b_2e_20260701"
        / "launcher_args.json",
    )
    parser.add_argument(
        "--data",
        type=Path,
        default=Path(r"D:\DataAI\AIEx\newdataset\yolo_f\data.yaml"),
    )
    parser.add_argument(
        "--cidt-summary",
        type=Path,
        default=REPO_ROOT
        / "runs"
        / "audit_cidt_readiness_full_train_20260714"
        / "summary.json",
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
        / "TRKH_5CLASS_IELT_MHV_SIGNAL_A0_PROTOCOL_20260720.md",
    )
    parser.add_argument("--official-root", type=Path, default=OFFICIAL_ROOT)
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=REPO_ROOT / "runs" / "audit_ielt_mhv_signal_a0_20260720",
    )
    parser.add_argument("--preflight-only", action="store_true")
    parser.add_argument("--replay-summary", type=Path)
    parser.add_argument("--device", choices=("cuda",), default="cuda")
    parser.add_argument("--batch-size", type=int, default=BATCH_SIZE)
    parser.add_argument("--num-workers", type=int, default=NUM_WORKERS)
    parser.add_argument("--seed", type=int, default=SEED)
    return parser.parse_args(argv)


def _validate_locked_args(args: argparse.Namespace) -> None:
    if str(args.device) != "cuda":
        raise ValueError("IELT MHV A0 is locked to CUDA")
    if int(args.batch_size) != BATCH_SIZE or int(args.num_workers) != NUM_WORKERS:
        raise ValueError(
            f"IELT MHV A0 locks batch-size={BATCH_SIZE}, num-workers={NUM_WORKERS}"
        )
    if int(args.seed) != SEED:
        raise ValueError(f"IELT MHV A0 locks seed={SEED}")


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
    tracked = _git_value(
        REPO_ROOT, "status", "--short", "--untracked-files=no"
    )
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
        for relative in (
            "deep-research-report (9).md",
            "deep-research-report (10).md",
        )
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
    untracked = {
        value.decode("utf-8") for value in raw.split(b"\0") if value
    }
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
        "data_yaml": _verify_hash(args.data, LOCKED_DATA_SHA256, "data YAML"),
        "cidt_summary": _verify_hash(
            args.cidt_summary, LOCKED_CIDT_SUMMARY_SHA256, "CIDT summary"
        ),
        "cidt_predictions": _verify_hash(
            args.cidt_predictions,
            LOCKED_CIDT_PREDICTIONS_SHA256,
            "CIDT predictions",
        ),
        "protocol": _verify_hash(
            args.protocol, LOCKED_PROTOCOL_SHA256, "IELT MHV protocol"
        ),
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
    }
    official_root = Path(args.official_root).resolve()
    if not official_root.is_dir():
        raise FileNotFoundError(f"Official IELT repository is missing: {official_root}")
    official_files = {
        relative: _verify_hash(
            official_root / relative, expected, f"official IELT {relative}"
        )
        for relative, expected in LOCKED_OFFICIAL_HASHES.items()
    }
    official = {
        "path": str(official_root),
        "remote": _git_value(official_root, "remote", "get-url", "origin"),
        "commit": _git_value(official_root, "rev-parse", "HEAD"),
        "tree": _git_value(official_root, "rev-parse", "HEAD^{tree}"),
        "status": _git_value(official_root, "status", "--porcelain"),
        "files": official_files,
    }
    if (
        official["commit"] != LOCKED_OFFICIAL_COMMIT
        or official["tree"] != LOCKED_OFFICIAL_TREE
        or official["status"]
    ):
        raise ValueError(f"Official IELT repository lock differs: {official}")
    protected = _protected_untracked_state()
    if not bool(protected["passed"]):
        raise ValueError(f"Protected user payload identity differs: {protected}")
    return {
        "files": files,
        "official_repository": official,
        "protected_untracked": protected,
    }


def _prepare_output_dir(path: Path) -> Path:
    resolved = Path(path).expanduser().resolve()
    if resolved.exists():
        raise FileExistsError(f"Formal output already exists: {resolved}")
    resolved.mkdir(parents=True, exist_ok=False)
    return resolved


def _write_json(path: Path, payload: object) -> None:
    Path(path).write_text(
        json.dumps(to_serializable(payload), indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )


def _write_csv(path: Path, rows: Sequence[Mapping[str, object]]) -> None:
    if not rows:
        raise ValueError(f"Cannot write empty CSV: {path}")
    fields = list(rows[0].keys())
    with Path(path).open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields)
        writer.writeheader()
        for row in rows:
            if list(row.keys()) != fields:
                raise ValueError("CSV row fields are not stable")
            writer.writerow(row)


def _model_state_sha256(model: nn.Module) -> str:
    digest = hashlib.sha256()
    for name, tensor in sorted(model.state_dict().items()):
        value = tensor.detach().cpu().contiguous()
        digest.update(name.encode("utf-8"))
        digest.update(str(value.dtype).encode("ascii"))
        digest.update(np.asarray(value.shape, dtype=np.int64).tobytes())
        digest.update(value.reshape(-1).view(torch.uint8).numpy().tobytes())
    return digest.hexdigest()


def _artifact_record(path: Path, *, root: Path) -> Dict[str, object]:
    resolved = Path(path).resolve()
    return {
        "path": resolved.relative_to(Path(root).resolve()).as_posix(),
        "bytes": int(resolved.stat().st_size),
        "sha256": _sha256(resolved),
    }


def _write_manifest(output_dir: Path) -> Path:
    root = Path(output_dir).resolve()
    rows = []
    for path in sorted(root.rglob("*")):
        if not path.is_file() or path.name == "artifact_manifest.json":
            continue
        rows.append(_artifact_record(path, root=root))
    manifest = root / "artifact_manifest.json"
    _write_json(manifest, {"method": METHOD, "payload_count": len(rows), "files": rows})
    return manifest


def _verify_manifest(output_dir: Path) -> Dict[str, object]:
    root = Path(output_dir).resolve()
    manifest_path = root / "artifact_manifest.json"
    payload = json.loads(manifest_path.read_text(encoding="utf-8"))
    expected = {
        str(row["path"]): (int(row["bytes"]), str(row["sha256"]))
        for row in payload["files"]
    }
    observed = {}
    for path in sorted(root.rglob("*")):
        if not path.is_file() or path.name == "artifact_manifest.json":
            continue
        relative = path.relative_to(root).as_posix()
        observed[relative] = (int(path.stat().st_size), _sha256(path))
    if observed != expected:
        raise ValueError("Artifact manifest payload differs from disk")
    return {
        "passed": True,
        "payload_count": len(observed),
        "manifest_sha256": _sha256(manifest_path),
    }


def scaled_vote_count(active_patch_count: int) -> int:
    return max(1, int(round(24.0 * int(active_patch_count) / 784.0)))


def _stable_descending_local(scores: Tensor, original_indices: Tensor) -> Tensor:
    if scores.ndim != 2 or original_indices.shape != scores.shape:
        raise ValueError("Stable ranking requires matching [B,P] tensors")
    spatial = torch.argsort(original_indices, dim=1, stable=True)
    spatial_scores = scores.gather(1, spatial)
    ranked_spatial = torch.argsort(
        spatial_scores, dim=1, descending=True, stable=True
    )
    return spatial.gather(1, ranked_spatial)


def _smooth_grid_torch(values: Tensor) -> Tensor:
    if values.ndim != 3 or tuple(values.shape[-2:]) != (GRID_SIZE, GRID_SIZE):
        raise ValueError("MHV smoothing requires [B,16,16]")
    kernel = torch.tensor(
        MHV_KERNEL, device=values.device, dtype=torch.float32
    ).reshape(1, 1, 3, 3)
    return F.conv2d(values.float().unsqueeze(1), kernel, padding=1).squeeze(1)


def _smooth_grid_numpy(values: np.ndarray) -> np.ndarray:
    array = np.asarray(values)
    if array.ndim != 3 or tuple(array.shape[-2:]) != (GRID_SIZE, GRID_SIZE):
        raise ValueError("NumPy MHV smoothing requires [B,16,16]")
    padded = np.pad(array, ((0, 0), (1, 1), (1, 1)), mode="constant")
    result = np.zeros(array.shape, dtype=np.float64)
    kernel = np.asarray(MHV_KERNEL, dtype=np.float64)
    for row in range(3):
        for column in range(3):
            result += kernel[row, column] * padded[
                :, row : row + GRID_SIZE, column : column + GRID_SIZE
            ]
    return result


def _vote_maps_torch(
    cls_attention: Tensor, original_indices: Tensor
) -> Tuple[Tensor, Tensor]:
    if cls_attention.ndim != 3:
        raise ValueError("CLS attention must have shape [B,H,P]")
    batch, heads, patches = (int(value) for value in cls_attention.shape)
    if heads != HEAD_COUNT or tuple(original_indices.shape) != (batch, patches):
        raise ValueError("CLS attention/lineage shape differs from the locked keeper")
    expanded_indices = original_indices[:, None, :].expand(batch, heads, patches)
    flattened_scores = cls_attention.reshape(batch * heads, patches)
    flattened_indices = expanded_indices.reshape(batch * heads, patches)
    order = _stable_descending_local(flattened_scores, flattened_indices)
    count = scaled_vote_count(patches)
    local = order[:, :count].reshape(batch, heads, count)
    active_votes = torch.zeros(
        (batch, heads, patches), device=cls_attention.device, dtype=torch.float32
    )
    active_votes.scatter_(2, local, 1.0)
    full_votes = torch.zeros(
        (batch, heads, PATCH_COUNT),
        device=cls_attention.device,
        dtype=torch.float32,
    )
    full_votes.scatter_(2, expanded_indices, active_votes)
    return active_votes, full_votes


def selector_bank_torch(
    attention: Tensor,
    original_indices: Tensor,
    *,
    prefix_count: int,
    quota: int,
) -> Dict[str, object]:
    value = attention.detach().float()
    indices = original_indices.to(device=value.device, dtype=torch.long)
    if value.ndim != 4 or int(value.size(1)) != HEAD_COUNT:
        raise ValueError("Attention must have shape [B,8,N,N]")
    patch_count = int(indices.size(1))
    if int(value.size(-1)) != int(prefix_count) + patch_count:
        raise ValueError("Attention token count differs from patch lineage")
    if not bool((indices[:, 1:] > indices[:, :-1]).all()):
        raise ValueError("Patch lineage must be strictly ascending")
    cls_attention = value[:, :, 0, int(prefix_count) :]
    active_votes, full_votes = _vote_maps_torch(cls_attention, indices)
    raw = full_votes.sum(dim=1).reshape(-1, GRID_SIZE, GRID_SIZE)
    dephased_heads = []
    for head, offset in enumerate(DEPHASE_OFFSETS):
        grid = full_votes[:, head].reshape(-1, GRID_SIZE, GRID_SIZE)
        dephased_heads.append(torch.roll(grid, shifts=offset, dims=(-2, -1)))
    dephased = torch.stack(dephased_heads, dim=1).sum(dim=1)
    cls_full = torch.zeros(
        (int(value.size(0)), PATCH_COUNT), device=value.device, dtype=torch.float32
    )
    cls_full.scatter_(1, indices, cls_attention.mean(dim=1))
    scores = {
        "cls_mean_smooth": _smooth_grid_torch(
            cls_full.reshape(-1, GRID_SIZE, GRID_SIZE)
        ).reshape(-1, PATCH_COUNT),
        "vote_raw": raw.reshape(-1, PATCH_COUNT),
        "vote_dephased_smooth": _smooth_grid_torch(dephased).reshape(
            -1, PATCH_COUNT
        ),
        "mhv_smooth": _smooth_grid_torch(raw).reshape(-1, PATCH_COUNT),
    }
    selected_local: Dict[str, Tensor] = {}
    selected_original: Dict[str, Tensor] = {}
    for role in ROLE_NAMES:
        active_scores = scores[role].gather(1, indices)
        order = _stable_descending_local(active_scores, indices)
        selected_local[role] = order[:, : int(quota)]
        selected_original[role] = indices.gather(1, selected_local[role])
    return {
        "scores": scores,
        "selected_local": selected_local,
        "selected_original": selected_original,
        "active_votes": active_votes,
        "full_votes": full_votes,
        "cls_attention": cls_attention,
    }


def selector_bank_numpy(
    cls_attention: np.ndarray,
    original_indices: np.ndarray,
    *,
    quota: int,
) -> Dict[str, object]:
    attention = np.asarray(cls_attention, dtype=np.float64)
    indices = np.asarray(original_indices, dtype=np.int64)
    if attention.ndim != 3:
        raise ValueError("NumPy CLS attention must have shape [B,H,P]")
    batch, heads, patches = attention.shape
    if heads != HEAD_COUNT or indices.shape != (batch, patches):
        raise ValueError("NumPy attention/lineage shape differs")
    active_votes = np.zeros((batch, heads, patches), dtype=np.int64)
    full_votes = np.zeros((batch, heads, PATCH_COUNT), dtype=np.int64)
    count = scaled_vote_count(patches)
    for sample in range(batch):
        for head in range(heads):
            order = np.lexsort((indices[sample], -attention[sample, head]))
            local = order[:count]
            active_votes[sample, head, local] = 1
            full_votes[sample, head, indices[sample, local]] = 1
    raw = full_votes.sum(axis=1).reshape(batch, GRID_SIZE, GRID_SIZE)
    dephased_heads = []
    for head, offset in enumerate(DEPHASE_OFFSETS):
        dephased_heads.append(
            np.roll(
                full_votes[:, head].reshape(batch, GRID_SIZE, GRID_SIZE),
                shift=offset,
                axis=(-2, -1),
            )
        )
    dephased = np.stack(dephased_heads, axis=1).sum(axis=1)
    scores = {
        "vote_raw": raw.reshape(batch, PATCH_COUNT).astype(np.float64),
        "vote_dephased_smooth": _smooth_grid_numpy(dephased).reshape(
            batch, PATCH_COUNT
        ),
        "mhv_smooth": _smooth_grid_numpy(raw).reshape(batch, PATCH_COUNT),
    }
    selected = {}
    for role, full_score in scores.items():
        rows = []
        for sample in range(batch):
            active_score = full_score[sample, indices[sample]]
            order = np.lexsort((indices[sample], -active_score))
            rows.append(indices[sample, order[: int(quota)]])
        selected[role] = np.stack(rows, axis=0)
    return {
        "scores": scores,
        "selected_original": selected,
        "active_votes": active_votes,
        "full_votes": full_votes,
    }


@lru_cache(maxsize=BLOCK_COUNT)
def _projection_matrix(block_index: int) -> Tensor:
    generator = np.random.default_rng(SEED + int(block_index))
    matrix = generator.standard_normal((FEATURE_DIM, PROJECTION_DIM))
    q, r = np.linalg.qr(matrix, mode="reduced")
    signs = np.where(np.diag(r) < 0.0, -1.0, 1.0)
    q = q * signs.reshape(1, -1)
    return torch.from_numpy(q.astype(np.float32, copy=False))


def block_role_descriptor(
    patch_tokens: Tensor,
    selected_local: Tensor,
    *,
    block_index: int,
) -> Tensor:
    tokens = patch_tokens.detach().float()
    local = selected_local.to(device=tokens.device, dtype=torch.long)
    selected = tokens.gather(
        1, local.unsqueeze(-1).expand(-1, -1, int(tokens.size(-1)))
    )
    normalized = F.layer_norm(selected, (FEATURE_DIM,), eps=1e-6)
    projected = normalized @ _projection_matrix(block_index).to(tokens.device)
    descriptor = torch.cat((projected.mean(dim=1), projected.amax(dim=1)), dim=1)
    if tuple(descriptor.shape) != (int(tokens.size(0)), PROJECTION_DIM * 2):
        raise ValueError("IELT MHV block descriptor shape differs")
    if not bool(torch.isfinite(descriptor).all()):
        raise ValueError("IELT MHV descriptor contains non-finite values")
    return descriptor


def mhv_engineering_checks() -> Dict[str, object]:
    generator = torch.Generator().manual_seed(SEED + 911)
    cases = []
    all_passed = True
    for patches in (256, 218, 167):
        prefix = 7
        attention = torch.rand(
            (2, HEAD_COUNT, prefix + patches, prefix + patches),
            generator=generator,
            dtype=torch.float32,
        )
        attention = attention / attention.sum(dim=-1, keepdim=True)
        indices = torch.arange(PATCH_COUNT, dtype=torch.long).reshape(1, -1)
        if patches != PATCH_COUNT:
            indices = indices[:, :patches]
        indices = indices.expand(2, -1).contiguous()
        # Force deterministic ties in two heads to exercise original-index order.
        attention[:, :2, 0, prefix : prefix + min(12, patches)] = 0.25
        torch_result = selector_bank_torch(
            attention,
            indices,
            prefix_count=prefix,
            quota=min(5, patches),
        )
        numpy_result = selector_bank_numpy(
            torch_result["cls_attention"].numpy(),
            indices.numpy(),
            quota=min(5, patches),
        )
        score_exact = all(
            np.array_equal(
                torch_result["scores"][role].numpy(),
                numpy_result["scores"][role],
            )
            for role in ("vote_raw", "vote_dephased_smooth", "mhv_smooth")
        )
        selected_exact = all(
            np.array_equal(
                torch_result["selected_original"][role].numpy(),
                numpy_result["selected_original"][role],
            )
            for role in ("vote_raw", "vote_dephased_smooth", "mhv_smooth")
        )
        vote_mass_exact = bool(
            torch_result["active_votes"].sum().item()
            == 2 * HEAD_COUNT * scaled_vote_count(patches)
        )
        row = {
            "patches": patches,
            "vote_per_head": scaled_vote_count(patches),
            "score_exact": score_exact,
            "selected_exact": selected_exact,
            "vote_mass_exact": vote_mass_exact,
        }
        row["passed"] = all(
            bool(row[key])
            for key in ("score_exact", "selected_exact", "vote_mass_exact")
        )
        cases.append(row)
        all_passed = all_passed and bool(row["passed"])
    quota_exact = list(LAYER_QUOTAS) == [5, 4, 3, 3, 3, 3, 4, 5]
    quota_total_exact = sum(LAYER_QUOTAS) == 30
    return {
        "cases": cases,
        "quota": list(LAYER_QUOTAS),
        "quota_exact": quota_exact,
        "quota_total_exact": quota_total_exact,
        "dephase_offsets": [list(value) for value in DEPHASE_OFFSETS],
        "all_passed": bool(all_passed and quota_exact and quota_total_exact),
    }


def _positive_threshold(scores: np.ndarray, labels: np.ndarray) -> float:
    value = np.asarray(scores, dtype=np.float64).reshape(-1)
    target = np.asarray(labels, dtype=np.int64).reshape(-1)
    positives = np.sort(value[target == 1])
    if positives.size == 0:
        raise ValueError("Readout fit fold has no positive rows")
    allowed_breaks = int(math.floor((1.0 - MIN_FIT_TP_RETENTION) * positives.size))
    threshold = float(positives[min(allowed_breaks, positives.size - 1)])
    retained = float((positives >= threshold).mean())
    if retained + 1e-12 < MIN_FIT_TP_RETENTION:
        raise RuntimeError("Locked fit-fold threshold violates TP retention")
    return threshold


def _positive_probability(model: LogisticRegression, features: np.ndarray) -> np.ndarray:
    probabilities = model.predict_proba(features)
    classes = np.asarray(model.classes_, dtype=np.int64)
    matches = np.flatnonzero(classes == 1)
    if matches.size == 0:
        return np.zeros(int(features.shape[0]), dtype=np.float64)
    return np.asarray(probabilities[:, int(matches[0])], dtype=np.float64)


def build_readout_features(
    descriptors: Mapping[str, np.ndarray], log_probabilities: np.ndarray
) -> Dict[str, np.ndarray]:
    logs = np.asarray(log_probabilities, dtype=np.float64)
    if logs.ndim != 2 or int(logs.shape[1]) != 5:
        raise ValueError("Keeper log probabilities must have shape [N,5]")
    if set(descriptors) != set(ROLE_NAMES):
        raise ValueError("Descriptor role bank differs from the locked MHV roles")
    result = {"base_only": logs}
    for role in ROLE_NAMES:
        descriptor = np.asarray(descriptors[role], dtype=np.float64)
        if descriptor.shape != (logs.shape[0], ROLE_DIM):
            raise ValueError(f"Descriptor shape differs for {role}: {descriptor.shape}")
        result[role] = np.concatenate((descriptor, logs), axis=1)
    return result


def fit_clean_oof_readouts(
    features: Mapping[str, np.ndarray],
    labels: np.ndarray,
    folds: np.ndarray,
    *,
    seed: int = SEED,
) -> Tuple[Dict[str, np.ndarray], Dict[str, np.ndarray], Dict[str, object]]:
    binary = np.asarray(labels, dtype=np.int64).reshape(-1)
    fold_values = np.asarray(folds, dtype=np.int64).reshape(-1)
    rows = int(binary.size)
    if set(features) != set(ALL_READOUT_ROLES):
        raise ValueError("Readout feature roles differ from the locked bank")
    scores = {
        role: np.full(rows, np.nan, dtype=np.float64) for role in ALL_READOUT_ROLES
    }
    actions = {
        role: np.zeros(rows, dtype=np.bool_) for role in ALL_READOUT_ROLES
    }
    states: Dict[str, object] = {role: {"folds": []} for role in ALL_READOUT_ROLES}
    for role in ALL_READOUT_ROLES:
        values = np.asarray(features[role], dtype=np.float64)
        for held_fold in FIT_FOLDS:
            held = fold_values == int(held_fold)
            fit = np.isin(fold_values, FIT_FOLDS) & ~held
            if not bool(held.any()) or len(np.unique(binary[fit])) != 2:
                raise ValueError(f"Readout fold {held_fold} lacks binary support")
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
                    random_state=int(seed),
                    solver="lbfgs",
                )
                model.fit(fit_scaled, binary[fit])
            fit_scores = _positive_probability(model, fit_scaled)
            threshold = _positive_threshold(fit_scores, binary[fit])
            held_scores = _positive_probability(model, scaler.transform(values[held]))
            scores[role][held] = held_scores
            actions[role][held] = held_scores >= threshold
            state = {
                "held_fold": int(held_fold),
                "fit_folds": [
                    int(value) for value in FIT_FOLDS if int(value) != held_fold
                ],
                "fit_rows": int(fit.sum()),
                "held_rows": int(held.sum()),
                "classes": model.classes_.astype(int).tolist(),
                "threshold": threshold,
                "scaler_mean": scaler.mean_.tolist(),
                "scaler_scale": scaler.scale_.tolist(),
                "coefficient": model.coef_.tolist(),
                "intercept": model.intercept_.tolist(),
                "iterations": model.n_iter_.astype(int).tolist(),
                "converged": not any(
                    issubclass(item.category, ConvergenceWarning) for item in caught
                ),
                "fit_tp_retention": float(
                    (fit_scores[binary[fit] == 1] >= threshold).mean()
                ),
            }
            states[role]["folds"].append(state)
    if any(not np.isfinite(value).all() for value in scores.values()):
        raise ValueError("OOF readout did not score every row exactly once")
    return scores, actions, states


def apply_clean_readout_states(
    features: Mapping[str, np.ndarray],
    folds: np.ndarray,
    states: Mapping[str, object],
) -> Tuple[Dict[str, np.ndarray], Dict[str, np.ndarray]]:
    fold_values = np.asarray(folds, dtype=np.int64).reshape(-1)
    rows = int(fold_values.size)
    scores = {
        role: np.full(rows, np.nan, dtype=np.float64) for role in ALL_READOUT_ROLES
    }
    actions = {
        role: np.zeros(rows, dtype=np.bool_) for role in ALL_READOUT_ROLES
    }
    for role in ALL_READOUT_ROLES:
        values = np.asarray(features[role], dtype=np.float64)
        role_state = states[role]
        for state in role_state["folds"]:
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
    if any(not np.isfinite(value).all() for value in scores.values()):
        raise ValueError("Condition replay did not score every held row")
    return scores, actions


def descriptor_statistics(descriptors: np.ndarray) -> Dict[str, object]:
    value = np.asarray(descriptors, dtype=np.float64)
    if value.ndim != 2 or int(value.shape[1]) != ROLE_DIM:
        raise ValueError(f"Descriptor statistics require [N,{ROLE_DIM}]")
    centered = value - value.mean(axis=0, keepdims=True)
    singular = np.linalg.svd(centered, full_matrices=False, compute_uv=False)
    energy = np.square(singular)
    probability = energy / max(float(energy.sum()), 1e-30)
    nonzero = probability > 0.0
    effective_rank = float(
        np.exp(-np.sum(probability[nonzero] * np.log(probability[nonzero])))
    )
    blocks = []
    width = PROJECTION_DIM * 2
    for block_index in range(BLOCK_COUNT):
        block = value[:, block_index * width : (block_index + 1) * width]
        standard_deviation = block.std(axis=0)
        blocks.append(
            {
                "block": block_index,
                "active_dimensions": int((standard_deviation > 1e-8).sum()),
                "minimum_std": float(standard_deviation.min()),
                "maximum_std": float(standard_deviation.max()),
                "constant": bool(np.all(standard_deviation <= 1e-8)),
            }
        )
    return {
        "shape": [int(value.shape[0]), int(value.shape[1])],
        "sha256": _array_sha256(value.astype(np.float32)),
        "effective_rank": effective_rank,
        "blocks": blocks,
        "all_values_finite": bool(np.isfinite(value).all()),
    }


def role_similarity(descriptors: Mapping[str, np.ndarray]) -> Dict[str, object]:
    result: Dict[str, object] = {}
    candidate = np.asarray(descriptors["mhv_smooth"], dtype=np.float64)
    candidate_norm = np.linalg.norm(candidate, axis=1)
    for role in ROLE_NAMES[:-1]:
        control = np.asarray(descriptors[role], dtype=np.float64)
        denominator = np.maximum(
            candidate_norm * np.linalg.norm(control, axis=1), 1e-30
        )
        cosine = np.sum(candidate * control, axis=1) / denominator
        left = candidate.reshape(-1)
        right = control.reshape(-1)
        correlation = float(np.corrcoef(left, right)[0, 1])
        result[role] = {
            "mean_row_cosine": float(cosine.mean()),
            "minimum_row_cosine": float(cosine.min()),
            "flattened_pearson": correlation,
        }
    return result


def _binary_role_metrics(
    labels: np.ndarray,
    targets: np.ndarray,
    folds: np.ndarray,
    scores: np.ndarray,
    actions: np.ndarray,
) -> Dict[str, object]:
    binary = np.asarray(labels, dtype=np.int64).reshape(-1)
    target_values = np.asarray(targets, dtype=np.int64).reshape(-1)
    fold_values = np.asarray(folds, dtype=np.int64).reshape(-1)
    score_values = np.asarray(scores, dtype=np.float64).reshape(-1)
    action_values = np.asarray(actions, dtype=np.bool_).reshape(-1)
    positive = binary == 1
    negative = binary == 0
    per_fold = []
    for fold in FIT_FOLDS:
        selected = fold_values == int(fold)
        fold_positive = positive & selected
        fold_negative = negative & selected
        per_fold.append(
            {
                "fold": int(fold),
                "rows": int(selected.sum()),
                "auroc": float(roc_auc_score(binary[selected], score_values[selected])),
                "tp_retention": float(action_values[fold_positive].mean()),
                "fp_rejection": float((~action_values[fold_negative]).mean()),
                "tp_breaks": int((~action_values[fold_positive]).sum()),
                "fp_rejects": int((~action_values[fold_negative]).sum()),
            }
        )
    per_target = {}
    for target in RESTRICTED_NEGATIVE_CLASSES:
        selected = target_values == int(target)
        per_target[str(target)] = {
            "rows": int(selected.sum()),
            "rejected": int((~action_values[selected]).sum()),
            "rejection_rate": float((~action_values[selected]).mean()),
        }
    return {
        "auroc": float(roc_auc_score(binary, score_values)),
        "tp_retention": float(action_values[positive].mean()),
        "fp_rejection": float((~action_values[negative]).mean()),
        "tp_breaks": int((~action_values[positive]).sum()),
        "fp_rejects": int((~action_values[negative]).sum()),
        "per_fold": per_fold,
        "per_target": per_target,
    }


def build_analysis(
    *,
    labels: np.ndarray,
    targets: np.ndarray,
    folds: np.ndarray,
    scores: Mapping[str, np.ndarray],
    actions: Mapping[str, np.ndarray],
    descriptor_stats: Mapping[str, Mapping[str, object]],
    selector_summary: Mapping[str, object],
) -> Dict[str, object]:
    if set(scores) != set(ALL_READOUT_ROLES) or set(actions) != set(
        ALL_READOUT_ROLES
    ):
        raise ValueError("Analysis role bank differs from the locked MHV roles")
    binary = np.asarray(labels, dtype=np.int64).reshape(-1)
    metrics = {
        role: _binary_role_metrics(
            binary, targets, folds, scores[role], actions[role]
        )
        for role in ALL_READOUT_ROLES
    }
    base_actions = np.asarray(actions["base_only"], dtype=np.bool_)
    transitions = {}
    for role in ROLE_NAMES:
        role_actions = np.asarray(actions[role], dtype=np.bool_)
        base_correct = base_actions == binary.astype(np.bool_)
        role_correct = role_actions == binary.astype(np.bool_)
        transitions[role] = {
            "corrections": int((~base_correct & role_correct).sum()),
            "harms": int((base_correct & ~role_correct).sum()),
            "changed_actions": int((base_actions != role_actions).sum()),
        }

    candidate = metrics["mhv_smooth"]
    gains = {
        role: float(candidate["auroc"] - metrics[role]["auroc"])
        for role in CONTROL_NAMES
    }
    fold_wins = 0
    candidate_folds = {row["fold"]: row for row in candidate["per_fold"]}
    for fold in FIT_FOLDS:
        candidate_auc = float(candidate_folds[int(fold)]["auroc"])
        control_auc = max(
            float(
                next(
                    row["auroc"]
                    for row in metrics[role]["per_fold"]
                    if int(row["fold"]) == int(fold)
                )
            )
            for role in CONTROL_NAMES
        )
        fold_wins += int(candidate_auc > control_auc)

    candidate_selector = selector_summary["roles"]["mhv_smooth"]
    dephased_selector = selector_summary["roles"]["vote_dephased_smooth"]
    candidate_stats = descriptor_stats["mhv_smooth"]
    mechanism_gates = {
        "candidate_auroc_ge_065": float(candidate["auroc"]) >= 0.65,
        "gain_vs_base_ge_003": gains["base_only"] >= 0.03,
        "gain_vs_cls_mean_ge_002": gains["cls_mean_smooth"] >= 0.02,
        "gain_vs_vote_raw_ge_001": gains["vote_raw"] >= 0.01,
        "gain_vs_dephased_ge_003": gains["vote_dephased_smooth"] >= 0.03,
        "tp_retention_ge_095": float(candidate["tp_retention"]) >= 0.95,
        "fp_rejection_ge_020": float(candidate["fp_rejection"]) >= 0.20,
        "tp_breaks_not_above_any_control": all(
            int(candidate["tp_breaks"]) <= int(metrics[role]["tp_breaks"])
            for role in CONTROL_NAMES
        ),
        "fp_rejects_at_least_10_above_each_control": all(
            int(candidate["fp_rejects"]) >= int(metrics[role]["fp_rejects"]) + 10
            for role in CONTROL_NAMES
        ),
        "wins_at_least_3_of_4_folds": fold_wins >= 3,
        "minimum_fold_tp_retention_ge_090": min(
            float(row["tp_retention"]) for row in candidate["per_fold"]
        )
        >= 0.90,
        "candidate_effective_rank_ge_16": float(candidate_stats["effective_rank"])
        >= 16.0,
        "all_candidate_block_groups_nonconstant": all(
            not bool(row["constant"]) for row in candidate_stats["blocks"]
        ),
        "candidate_top_score_tie_not_majority": float(
            candidate_selector["top_score_tie_fraction"]
        )
        < 0.50,
        "candidate_alignment_above_dephased": float(
            candidate_selector["alignment_score"]
        )
        > float(dephased_selector["alignment_score"]),
    }
    return {
        "metrics": metrics,
        "candidate_auroc_gains": gains,
        "candidate_fold_wins_over_all_controls": int(fold_wins),
        "transitions_vs_base_only": transitions,
        "mechanism_gates": mechanism_gates,
        "mechanism_gates_passed": all(mechanism_gates.values()),
    }


def condition_gate(analysis: Mapping[str, object]) -> Dict[str, object]:
    metrics = analysis["metrics"]
    candidate = metrics["mhv_smooth"]
    best_control_auroc = max(
        float(metrics[role]["auroc"]) for role in CONTROL_NAMES
    )
    gates = {
        "candidate_auroc_ge_060": float(candidate["auroc"]) >= 0.60,
        "candidate_tp_retention_ge_092": float(candidate["tp_retention"]) >= 0.92,
        "candidate_auroc_above_best_control": float(candidate["auroc"])
        > best_control_auroc,
    }
    return {
        "candidate_auroc": float(candidate["auroc"]),
        "best_control_auroc": best_control_auroc,
        "candidate_minus_best_control": float(candidate["auroc"])
        - best_control_auroc,
        "gates": gates,
        "passed": all(gates.values()),
    }


def _recursive_numeric_difference(left: object, right: object) -> float:
    if isinstance(left, Mapping) and isinstance(right, Mapping):
        if set(left) != set(right):
            return math.inf
        return max(
            (_recursive_numeric_difference(left[key], right[key]) for key in left),
            default=0.0,
        )
    if isinstance(left, Sequence) and not isinstance(left, (str, bytes)):
        if not isinstance(right, Sequence) or isinstance(right, (str, bytes)):
            return math.inf
        if len(left) != len(right):
            return math.inf
        return max(
            (_recursive_numeric_difference(a, b) for a, b in zip(left, right)),
            default=0.0,
        )
    if isinstance(left, (int, float, np.number)) and isinstance(
        right, (int, float, np.number)
    ):
        return abs(float(left) - float(right))
    return 0.0 if left == right else math.inf


class FrozenMHVCapture:
    def __init__(self, model: nn.Module) -> None:
        blocks = getattr(model, "blocks", None)
        if not isinstance(blocks, nn.ModuleList) or len(blocks) != BLOCK_COUNT:
            raise TypeError("IELT MHV A0 requires the keeper's eight-block stack")
        self.model = model
        self.blocks = blocks
        self.prefix_count = int(getattr(model, "num_prefix_tokens", -1))
        if self.prefix_count != 7:
            raise ValueError("IELT MHV A0 locks seven prefix tokens")
        self.attentions: Dict[int, Tensor] = {}
        self.tokens: Dict[int, Tensor] = {}
        self.enabled = False
        self._before_counts = self._hook_counts()
        self.handles = []
        for block_index, block in enumerate(blocks):
            attention_dropout = getattr(getattr(block, "attn", None), "attention_dropout", None)
            if not isinstance(attention_dropout, nn.Module):
                raise TypeError(f"Block {block_index} lacks attention dropout")
            self.handles.append(
                attention_dropout.register_forward_hook(
                    self._attention_hook(block_index)
                )
            )
            self.handles.append(block.register_forward_hook(self._block_hook(block_index)))

    def _hook_counts(self) -> Dict[str, int]:
        result = {}
        for block_index, block in enumerate(self.blocks):
            result[f"block_{block_index}"] = len(block._forward_hooks)
            result[f"attention_{block_index}"] = len(
                block.attn.attention_dropout._forward_hooks
            )
        return result

    def _attention_hook(self, block_index: int):
        def capture(_module, _inputs, output) -> None:
            if not self.enabled:
                return
            if not torch.is_tensor(output):
                raise TypeError(f"Block {block_index} attention is not a tensor")
            self.attentions[block_index] = (
                output.detach().float().cpu().contiguous()
            )

        return capture

    def _block_hook(self, block_index: int):
        def capture(_module, _inputs, output) -> None:
            if not self.enabled:
                return
            value = output[0] if isinstance(output, tuple) else output
            if not torch.is_tensor(value):
                raise TypeError(f"Block {block_index} output is not a tensor")
            self.tokens[block_index] = value.detach().float().cpu().contiguous()

        return capture

    def start(self) -> None:
        self.attentions.clear()
        self.tokens.clear()
        self.enabled = True

    def consume(self, trace: Mapping[str, object]) -> Dict[str, object]:
        self.enabled = False
        expected = set(range(BLOCK_COUNT))
        if set(self.attentions) != expected or set(self.tokens) != expected:
            raise RuntimeError(
                "Incomplete MHV hook capture: "
                f"attention={sorted(self.attentions)}, tokens={sorted(self.tokens)}"
            )
        after_indices_raw = trace.get("block_patch_indices")
        if not isinstance(after_indices_raw, Sequence) or len(after_indices_raw) != BLOCK_COUNT:
            raise ValueError("Trace lacks eight block patch-index tensors")
        after_indices = [value.detach().cpu().long() for value in after_indices_raw]
        batch = int(after_indices[0].size(0))
        initial = torch.arange(PATCH_COUNT, dtype=torch.long).reshape(1, -1)
        before_indices = [initial.expand(batch, -1).contiguous()]
        before_indices.extend(after_indices[:-1])
        entering_counts = tuple(int(value.size(1)) for value in before_indices)
        after_counts = tuple(int(value.size(1)) for value in after_indices)
        if entering_counts != EXPECTED_ENTERING_PATCH_COUNTS:
            raise ValueError(f"Entering patch lineage differs: {entering_counts}")
        if after_counts != EXPECTED_AFTER_PATCH_COUNTS:
            raise ValueError(f"After-block patch lineage differs: {after_counts}")
        patch_tokens = {}
        for block_index in range(BLOCK_COUNT):
            patches = int(before_indices[block_index].size(1))
            attention = self.attentions[block_index]
            tokens = self.tokens[block_index]
            if int(attention.size(-1)) != self.prefix_count + patches:
                raise ValueError(f"Block {block_index} attention/lineage mismatch")
            if int(tokens.size(1)) != self.prefix_count + patches:
                raise ValueError(f"Block {block_index} token/lineage mismatch")
            patch_tokens[block_index] = tokens[:, self.prefix_count :]
        return {
            "attentions": dict(self.attentions),
            "patch_tokens": patch_tokens,
            "before_indices": before_indices,
            "after_indices": after_indices,
            "entering_patch_counts": entering_counts,
            "after_patch_counts": after_counts,
        }

    def close(self) -> Dict[str, object]:
        self.enabled = False
        for handle in self.handles:
            handle.remove()
        self.handles.clear()
        after = self._hook_counts()
        return {
            "before": self._before_counts,
            "after": after,
            "no_leak": after == self._before_counts,
        }

    def __enter__(self) -> "FrozenMHVCapture":
        return self

    def __exit__(self, _exc_type, _exc, _traceback) -> None:
        self.close()


def _metadata_tensors(
    metadata: Optional[Mapping[str, object]], device: torch.device
) -> Tuple[Optional[Tensor], Optional[Tensor]]:
    bbox = metadata.get("bbox") if isinstance(metadata, Mapping) else None
    image_mask = metadata.get("image_mask") if isinstance(metadata, Mapping) else None
    bbox_value = (
        bbox.to(device=device, dtype=torch.float32, non_blocking=True)
        if torch.is_tensor(bbox)
        else None
    )
    mask_value = (
        image_mask.to(device=device, dtype=torch.bool, non_blocking=True)
        if torch.is_tensor(image_mask)
        else None
    )
    return bbox_value, mask_value


def _trace_forward(
    model: nn.Module,
    images: Tensor,
    metadata: Optional[Mapping[str, object]],
    *,
    device: torch.device,
) -> Tuple[Tensor, Dict[str, object]]:
    bbox, image_mask = _metadata_tensors(metadata, device)
    features = model.forward_features(
        images,
        image_valid_mask=image_mask,
        bbox_token_prior=bbox,
        return_trace=True,
    )
    if bbox is not None:
        features["bbox"] = bbox
    forward_heads = getattr(model, "forward_heads", None)
    if callable(forward_heads):
        output = forward_heads(features)
    else:
        output = classification_logits_from_features(model, features)
    logits, _ = extract_bbox_from_model_output(output)
    return logits, features


def _cohort_from_rows(rows: Sequence[CleanTrainRow]) -> List[CleanTrainRow]:
    cohort = [
        row
        for row in rows
        if row.fold in FIT_FOLDS
        and row.keeper_prediction == FOCUS_CLASS
        and (
            row.target == FOCUS_CLASS or row.target in RESTRICTED_NEGATIVE_CLASSES
        )
    ]
    positives = sum(row.target == FOCUS_CLASS for row in cohort)
    negatives = len(cohort) - positives
    if (len(cohort), positives, negatives) != (
        EXPECTED_COHORT_ROWS,
        EXPECTED_POSITIVES,
        EXPECTED_NEGATIVES,
    ):
        raise ValueError(
            "IELT MHV cohort differs: "
            f"rows/TP/FP={(len(cohort), positives, negatives)}"
        )
    fold_counts = {}
    for fold in FIT_FOLDS:
        selected = [row for row in cohort if row.fold == fold]
        fold_counts[fold] = {
            "tp": sum(row.target == FOCUS_CLASS for row in selected),
            "fp": sum(row.target != FOCUS_CLASS for row in selected),
        }
    if fold_counts != EXPECTED_FOLD_COUNTS:
        raise ValueError(f"IELT MHV fold counts differ: {fold_counts}")
    index_hash = _ordered_index_sha256([row.sample_index for row in cohort])
    if index_hash != EXPECTED_ORDERED_INDEX_SHA256:
        raise ValueError(f"IELT MHV cohort order hash differs: {index_hash}")
    source_folds: Dict[str, set[int]] = {}
    for row in cohort:
        source_folds.setdefault(row.source_stem, set()).add(int(row.fold))
    overlaps = {
        source: sorted(folds)
        for source, folds in source_folds.items()
        if len(folds) > 1
    }
    if overlaps:
        raise ValueError(f"CIDT cohort has cross-fold source overlap: {overlaps}")
    return cohort


def _review_indices(cohort: Sequence[CleanTrainRow]) -> List[int]:
    categories = (
        ("tp", lambda row: row.target == FOCUS_CLASS),
        ("fp_target_0", lambda row: row.target == 0),
        ("fp_target_2", lambda row: row.target == 2),
        ("fp_target_4", lambda row: row.target == 4),
    )
    selected = []
    for _, predicate in categories:
        values = sorted(row.sample_index for row in cohort if predicate(row))[:3]
        if len(values) != 3:
            raise ValueError("Fixed MHV review cohort lacks three category rows")
        selected.extend(values)
    return selected


def _mean_head_jaccard(active_votes: Tensor) -> Tensor:
    votes = active_votes.to(dtype=torch.bool)
    rows = []
    for first in range(HEAD_COUNT):
        for second in range(first + 1, HEAD_COUNT):
            intersection = (votes[:, first] & votes[:, second]).sum(dim=1).float()
            union = (votes[:, first] | votes[:, second]).sum(dim=1).float()
            rows.append(intersection / union.clamp_min(1.0))
    return torch.stack(rows, dim=1).mean(dim=1)


def _normalized_entropy(active_scores: Tensor) -> Tensor:
    values = active_scores.float().clamp_min(0.0)
    probability = values / values.sum(dim=1, keepdim=True).clamp_min(1e-12)
    entropy = -(probability * probability.clamp_min(1e-12).log()).sum(dim=1)
    return entropy / math.log(max(2, int(values.size(1))))


def _selected_prior_metrics(
    selected_original: Tensor,
    original_indices: Tensor,
    foreground_prior: Tensor,
    bbox_prior: Tensor,
) -> Dict[str, Tensor]:
    selected = selected_original.long()
    foreground = foreground_prior.float()
    bbox = bbox_prior.float()
    selected_foreground = foreground.gather(1, selected)
    selected_bbox = bbox.gather(1, selected)
    active_foreground = foreground.gather(1, original_indices.long())
    active_bbox = bbox.gather(1, original_indices.long())
    foreground_mean = selected_foreground.mean(dim=1)
    bbox_mean = selected_bbox.mean(dim=1)
    return {
        "foreground_mean": foreground_mean,
        "bbox_mean": bbox_mean,
        "foreground_mass_ratio": selected_foreground.sum(dim=1)
        / active_foreground.sum(dim=1).clamp_min(1e-12),
        "bbox_mass_ratio": selected_bbox.sum(dim=1)
        / active_bbox.sum(dim=1).clamp_min(1e-12),
        "alignment": 0.5 * (foreground_mean + bbox_mean),
    }


def _normalize_map(values: np.ndarray) -> np.ndarray:
    array = np.asarray(values, dtype=np.float32)
    minimum = float(array.min())
    maximum = float(array.max())
    if maximum - minimum <= 1e-12:
        return np.zeros_like(array)
    return (array - minimum) / (maximum - minimum)


def extract_condition(
    *,
    model: nn.Module,
    loader,
    loader_summary: Mapping[str, object],
    cohort: Sequence[CleanTrainRow],
    device: torch.device,
    condition: str,
    check_ordinary: bool,
    review_indices: Sequence[int],
) -> Dict[str, object]:
    model = model.to(device).eval()
    state_before = _model_state_sha256(model)
    if device.type == "cuda":
        torch.cuda.empty_cache()
        torch.cuda.reset_peak_memory_stats(device)
        torch.cuda.synchronize(device)
    started = time.perf_counter()
    cohort_by_index = {row.sample_index: row for row in cohort}
    expected_order = [row.sample_index for row in cohort]
    observed_indices: List[int] = []
    observed_targets: List[int] = []
    observed_folds: List[int] = []
    probability_chunks: List[np.ndarray] = []
    descriptor_chunks: Dict[str, List[np.ndarray]] = {
        role: [] for role in ROLE_NAMES
    }
    telemetry: List[Dict[str, object]] = []
    visual_cache: Dict[int, Dict[str, object]] = {}
    review_set = set(int(value) for value in review_indices)
    bbox_batches: List[Tensor] = []
    declaration_mismatches: List[Dict[str, int]] = []
    hook_probability_error = 0.0
    hook_argmax_exact = True
    cidt_probability_error = 0.0
    oracle_score_error = 0.0
    oracle_selected_exact = True
    oracle_checks = 0
    token_counts = None
    capture = FrozenMHVCapture(model)
    hook_lifecycle: Dict[str, object] = {}
    try:
        with torch.inference_mode():
            for images, targets, metadata in loader:
                sample_indices = metadata.get("sample_index")
                if not torch.is_tensor(sample_indices):
                    raise ValueError("MHV loader metadata lacks sample_index")
                batch_indices = [int(value) for value in sample_indices.tolist()]
                batch_targets = [int(value) for value in targets.tolist()]
                expected_rows = [cohort_by_index[index] for index in batch_indices]
                if batch_targets != [row.target for row in expected_rows]:
                    raise ValueError("MHV loader targets differ from CIDT declarations")
                observed_indices.extend(batch_indices)
                observed_targets.extend(batch_targets)
                observed_folds.extend(int(row.fold) for row in expected_rows)
                if check_ordinary:
                    crop_bbox = metadata.get("crop_bbox")
                    if not torch.is_tensor(crop_bbox):
                        raise ValueError("Clean MHV loader lacks crop_bbox metadata")
                    bbox_batches.append(crop_bbox[:, :4].float().cpu().contiguous())

                images_cpu = images.detach().float().cpu()
                images_device = images.to(
                    device=device, dtype=torch.float32, non_blocking=True
                )
                ordinary_probabilities = None
                if check_ordinary:
                    with torch.autocast(
                        device_type="cuda", dtype=torch.bfloat16, enabled=True
                    ):
                        ordinary_logits, ordinary_features = (
                            _forward_classification_with_metadata(
                                model, images_device, metadata, device=device
                            )
                        )
                    ordinary_probabilities = ordinary_logits.float().softmax(dim=1)
                    del ordinary_features, ordinary_logits

                capture.start()
                with torch.autocast(
                    device_type="cuda", dtype=torch.bfloat16, enabled=True
                ):
                    traced_logits, features = _trace_forward(
                        model, images_device, metadata, device=device
                    )
                captured = capture.consume(features["trace"])
                probabilities = traced_logits.float().softmax(dim=1).cpu()
                if not bool(torch.isfinite(probabilities).all()):
                    raise ValueError("Traced keeper probabilities are non-finite")
                if ordinary_probabilities is not None:
                    ordinary_cpu = ordinary_probabilities.cpu()
                    hook_probability_error = max(
                        hook_probability_error,
                        float((ordinary_cpu - probabilities).abs().max().item()),
                    )
                    hook_argmax_exact = hook_argmax_exact and bool(
                        torch.equal(
                            ordinary_cpu.argmax(dim=1), probabilities.argmax(dim=1)
                        )
                    )
                probability_chunks.append(probabilities.numpy())

                expected_probabilities = np.asarray(
                    [row.keeper_probabilities for row in expected_rows],
                    dtype=np.float64,
                )
                cidt_probability_error = max(
                    cidt_probability_error,
                    float(
                        np.max(
                            np.abs(
                                probabilities.numpy().astype(np.float64)
                                - expected_probabilities
                            )
                        )
                    ),
                )
                observed_predictions = probabilities.argmax(dim=1).tolist()
                for source, prediction in zip(expected_rows, observed_predictions):
                    if int(prediction) != int(source.keeper_prediction):
                        declaration_mismatches.append(
                            {
                                "sample_index": int(source.sample_index),
                                "target": int(source.target),
                                "expected_prediction": int(
                                    source.keeper_prediction
                                ),
                                "observed_prediction": int(prediction),
                            }
                        )

                trace = features["trace"]
                foreground_prior = trace.get("foreground_prior")
                bbox_prior = trace.get("bbox_patch_prior")
                if not torch.is_tensor(foreground_prior) or not torch.is_tensor(
                    bbox_prior
                ):
                    raise ValueError("Keeper trace lacks full foreground/bbox priors")
                foreground_cpu = foreground_prior.detach().float().cpu()
                bbox_cpu = bbox_prior.detach().float().cpu()
                if tuple(foreground_cpu.shape) != (
                    len(batch_indices),
                    PATCH_COUNT,
                ) or tuple(bbox_cpu.shape) != (len(batch_indices), PATCH_COUNT):
                    raise ValueError("Keeper prior grid shape differs from 16x16")

                role_blocks: Dict[str, List[Tensor]] = {
                    role: [] for role in ROLE_NAMES
                }
                for block_index in range(BLOCK_COUNT):
                    attention = captured["attentions"][block_index]
                    patch_tokens = captured["patch_tokens"][block_index]
                    original_indices = captured["before_indices"][block_index]
                    selector = selector_bank_torch(
                        attention,
                        original_indices,
                        prefix_count=capture.prefix_count,
                        quota=LAYER_QUOTAS[block_index],
                    )
                    oracle = selector_bank_numpy(
                        selector["cls_attention"].numpy(),
                        original_indices.numpy(),
                        quota=LAYER_QUOTAS[block_index],
                    )
                    for oracle_role in (
                        "vote_raw",
                        "vote_dephased_smooth",
                        "mhv_smooth",
                    ):
                        oracle_score_error = max(
                            oracle_score_error,
                            float(
                                np.max(
                                    np.abs(
                                        selector["scores"][oracle_role].numpy()
                                        - oracle["scores"][oracle_role]
                                    )
                                )
                            ),
                        )
                        oracle_selected_exact = oracle_selected_exact and bool(
                            np.array_equal(
                                selector["selected_original"][
                                    oracle_role
                                ].numpy(),
                                oracle["selected_original"][oracle_role],
                            )
                        )
                        oracle_checks += len(batch_indices)

                    jaccard = _mean_head_jaccard(selector["active_votes"])
                    candidate_active_scores = selector["scores"][
                        "mhv_smooth"
                    ].gather(1, original_indices)
                    vote_entropy = _normalized_entropy(candidate_active_scores)
                    role_prior_metrics = {}
                    for role in ROLE_NAMES:
                        role_blocks[role].append(
                            block_role_descriptor(
                                patch_tokens,
                                selector["selected_local"][role],
                                block_index=block_index,
                            )
                        )
                        role_prior_metrics[role] = _selected_prior_metrics(
                            selector["selected_original"][role],
                            original_indices,
                            foreground_cpu,
                            bbox_cpu,
                        )

                    for sample_position, sample_index in enumerate(batch_indices):
                        row: Dict[str, object] = {
                            "condition": condition,
                            "sample_index": int(sample_index),
                            "fold": int(expected_rows[sample_position].fold),
                            "target": int(expected_rows[sample_position].target),
                            "block": int(block_index),
                            "active_patches": int(original_indices.size(1)),
                            "vote_per_head": scaled_vote_count(
                                int(original_indices.size(1))
                            ),
                            "quota": int(LAYER_QUOTAS[block_index]),
                            "head_topk_mean_jaccard": float(jaccard[sample_position]),
                            "mhv_vote_entropy": float(vote_entropy[sample_position]),
                        }
                        for role in ROLE_NAMES:
                            active_score = selector["scores"][role][
                                sample_position
                            ].gather(0, original_indices[sample_position])
                            top_ties = int(
                                torch.isclose(
                                    active_score,
                                    active_score.max(),
                                    rtol=0.0,
                                    atol=0.0,
                                ).sum()
                            )
                            metrics = role_prior_metrics[role]
                            row[f"{role}_top_ties"] = top_ties
                            row[f"{role}_foreground_mean"] = float(
                                metrics["foreground_mean"][sample_position]
                            )
                            row[f"{role}_bbox_mean"] = float(
                                metrics["bbox_mean"][sample_position]
                            )
                            row[f"{role}_foreground_mass_ratio"] = float(
                                metrics["foreground_mass_ratio"][sample_position]
                            )
                            row[f"{role}_bbox_mass_ratio"] = float(
                                metrics["bbox_mass_ratio"][sample_position]
                            )
                            row[f"{role}_alignment"] = float(
                                metrics["alignment"][sample_position]
                            )
                            row[f"{role}_selected_original"] = ";".join(
                                str(int(value))
                                for value in selector["selected_original"][role][
                                    sample_position
                                ].tolist()
                            )
                        telemetry.append(row)

                        if sample_index in review_set:
                            cache = visual_cache.setdefault(
                                sample_index,
                                {
                                    "image": images_cpu[sample_position].numpy(),
                                    "target": int(
                                        expected_rows[sample_position].target
                                    ),
                                    "fold": int(expected_rows[sample_position].fold),
                                    "maps": {
                                        role: np.zeros(
                                            (GRID_SIZE, GRID_SIZE), dtype=np.float32
                                        )
                                        for role in ROLE_NAMES
                                    },
                                },
                            )
                            for role in ROLE_NAMES:
                                score_map = selector["scores"][role][
                                    sample_position
                                ].reshape(GRID_SIZE, GRID_SIZE)
                                cache["maps"][role] += _normalize_map(
                                    score_map.numpy()
                                ) / float(BLOCK_COUNT)

                for role in ROLE_NAMES:
                    descriptor = torch.cat(role_blocks[role], dim=1)
                    if tuple(descriptor.shape) != (len(batch_indices), ROLE_DIM):
                        raise ValueError(f"Complete descriptor shape differs for {role}")
                    descriptor_chunks[role].append(descriptor.numpy())
                if token_counts is None:
                    token_counts = {
                        "entering": list(captured["entering_patch_counts"]),
                        "after": list(captured["after_patch_counts"]),
                    }
                del features, trace, traced_logits, captured, images_device
    finally:
        hook_lifecycle = capture.close()

    if device.type == "cuda":
        torch.cuda.synchronize(device)
        peak_cuda_bytes = int(torch.cuda.max_memory_allocated(device))
    else:
        peak_cuda_bytes = 0
    elapsed = time.perf_counter() - started
    state_after = _model_state_sha256(model)
    if observed_indices != expected_order:
        raise ValueError("MHV loader order differs from the locked cohort")
    expected_targets = [row.target for row in cohort]
    expected_folds = [row.fold for row in cohort]
    if observed_targets != expected_targets or observed_folds != expected_folds:
        raise ValueError("MHV observed targets/folds differ from CIDT")

    descriptors = {
        role: np.concatenate(descriptor_chunks[role], axis=0).astype(
            np.float32, copy=False
        )
        for role in ROLE_NAMES
    }
    probabilities_array = np.concatenate(probability_chunks, axis=0).astype(
        np.float64, copy=False
    )
    log_probabilities = np.log(np.clip(probabilities_array, 1e-8, 1.0))
    selector_roles = {}
    for role in ROLE_NAMES:
        selector_roles[role] = {
            "foreground_mean": float(
                np.mean([row[f"{role}_foreground_mean"] for row in telemetry])
            ),
            "bbox_mean": float(
                np.mean([row[f"{role}_bbox_mean"] for row in telemetry])
            ),
            "foreground_mass_ratio": float(
                np.mean(
                    [row[f"{role}_foreground_mass_ratio"] for row in telemetry]
                )
            ),
            "bbox_mass_ratio": float(
                np.mean([row[f"{role}_bbox_mass_ratio"] for row in telemetry])
            ),
            "alignment_score": float(
                np.mean([row[f"{role}_alignment"] for row in telemetry])
            ),
            "top_score_tie_fraction": float(
                np.mean([int(row[f"{role}_top_ties"]) > 1 for row in telemetry])
            ),
            "mean_top_score_ties": float(
                np.mean([row[f"{role}_top_ties"] for row in telemetry])
            ),
        }
    selector_summary = {
        "roles": selector_roles,
        "mean_head_topk_jaccard": float(
            np.mean([row["head_topk_mean_jaccard"] for row in telemetry])
        ),
        "mean_mhv_vote_entropy": float(
            np.mean([row["mhv_vote_entropy"] for row in telemetry])
        ),
        "oracle_score_max_abs_error": float(oracle_score_error),
        "oracle_selected_indices_exact": bool(oracle_selected_exact),
        "oracle_row_role_block_checks": int(oracle_checks),
    }
    bbox_hash = None
    if check_ordinary:
        bboxes = torch.cat(bbox_batches, dim=0).contiguous()
        if tuple(bboxes.shape) != (EXPECTED_COHORT_ROWS, 4):
            raise ValueError(f"Locked bbox shape differs: {tuple(bboxes.shape)}")
        bbox_hash = hashlib.sha256(bboxes.numpy().tobytes(order="C")).hexdigest()

    return {
        "condition": condition,
        "indices": np.asarray(observed_indices, dtype=np.int64),
        "targets": np.asarray(observed_targets, dtype=np.int64),
        "folds": np.asarray(observed_folds, dtype=np.int64),
        "probabilities": probabilities_array,
        "log_probabilities": log_probabilities,
        "descriptors": descriptors,
        "telemetry": telemetry,
        "selector_summary": selector_summary,
        "visual_cache": visual_cache,
        "runtime": {
            "elapsed_seconds": float(elapsed),
            "rows_per_second": float(len(observed_indices) / max(elapsed, 1e-9)),
            "peak_cuda_bytes": peak_cuda_bytes,
            "peak_cuda_gib": float(peak_cuda_bytes / 1024**3),
            "loader": dict(loader_summary),
            "token_counts": token_counts,
            "model_state_before_sha256": state_before,
            "model_state_after_sha256": state_after,
            "hook_lifecycle": hook_lifecycle,
        },
        "forward_replay": {
            "ordinary_checked": bool(check_ordinary),
            "hook_probability_max_abs_error": float(hook_probability_error),
            "hook_argmax_exact": bool(hook_argmax_exact),
            "cidt_probability_max_abs_error": float(cidt_probability_error),
            "cidt_declaration_mismatches": declaration_mismatches,
            "cidt_declarations_match_locked_exception": declaration_mismatches
            == [LOCKED_DECLARATION_EXCEPTION],
            "bbox_bytes_sha256": bbox_hash,
        },
    }


def _denormalized_image(image: np.ndarray, semantics: Mapping[str, object]) -> Image.Image:
    value = np.asarray(image, dtype=np.float32).transpose(1, 2, 0)
    mean = np.asarray(semantics["input_mean"], dtype=np.float32).reshape(1, 1, 3)
    std = np.asarray(semantics["input_std"], dtype=np.float32).reshape(1, 1, 3)
    value = np.clip(value * std + mean, 0.0, 1.0)
    return Image.fromarray(np.rint(value * 255.0).astype(np.uint8), mode="RGB")


def _overlay_score(base: Image.Image, score: np.ndarray) -> Image.Image:
    normalized = _normalize_map(score)
    red = np.rint(255.0 * normalized).astype(np.uint8)
    green = np.rint(210.0 * np.sqrt(normalized)).astype(np.uint8)
    blue = np.rint(40.0 * (1.0 - normalized)).astype(np.uint8)
    heat = Image.fromarray(np.stack((red, green, blue), axis=-1), mode="RGB")
    heat = heat.resize(base.size, resample=Image.Resampling.BILINEAR)
    alpha = Image.fromarray(
        np.rint(170.0 * normalized).astype(np.uint8), mode="L"
    ).resize(base.size, resample=Image.Resampling.BILINEAR)
    return Image.composite(heat, base, alpha)


def render_contact_sheet(
    path: Path,
    *,
    visual_cache: Mapping[int, Mapping[str, object]],
    review_indices: Sequence[int],
    cohort: Sequence[CleanTrainRow],
    semantics: Mapping[str, object],
) -> Dict[str, object]:
    if set(visual_cache) != set(int(value) for value in review_indices):
        raise ValueError("MHV visual cache differs from the fixed review cohort")
    cell = 192
    label_height = 30
    header_height = 34
    columns = ("input",) + ROLE_NAMES
    sheet = Image.new(
        "RGB",
        (cell * len(columns), header_height + (cell + label_height) * len(review_indices)),
        "white",
    )
    draw = ImageDraw.Draw(sheet)
    font = ImageFont.load_default()
    for column, name in enumerate(columns):
        draw.text((column * cell + 6, 10), name, fill="black", font=font)
    by_index = {row.sample_index: row for row in cohort}
    rows = []
    for row_position, sample_index in enumerate(review_indices):
        source = by_index[int(sample_index)]
        cache = visual_cache[int(sample_index)]
        base = _denormalized_image(np.asarray(cache["image"]), semantics).resize(
            (cell, cell), resample=Image.Resampling.BILINEAR
        )
        top = header_height + row_position * (cell + label_height)
        sheet.paste(base, (0, top + label_height))
        for column, role in enumerate(ROLE_NAMES, start=1):
            overlay = _overlay_score(base, np.asarray(cache["maps"][role]))
            sheet.paste(overlay, (column * cell, top + label_height))
        category = "tp" if source.target == FOCUS_CLASS else f"fp_target_{source.target}"
        label = (
            f"sample={sample_index} fold={source.fold} target={source.target} "
            f"category={category}"
        )
        draw.text((6, top + 9), label, fill="black", font=font)
        rows.append(
            {
                "sample_index": int(sample_index),
                "fold": int(source.fold),
                "target": int(source.target),
                "category": category,
            }
        )
    path.parent.mkdir(parents=True, exist_ok=True)
    sheet.save(path, format="PNG", optimize=True)
    return {
        "columns": list(columns),
        "rows": rows,
        "width": int(sheet.width),
        "height": int(sheet.height),
        "review_index_sha256": _ordered_index_sha256(review_indices),
    }


def _save_descriptor_payload(path: Path, extraction: Mapping[str, object]) -> None:
    np.savez_compressed(
        path,
        indices=np.asarray(extraction["indices"], dtype=np.int64),
        targets=np.asarray(extraction["targets"], dtype=np.int64),
        folds=np.asarray(extraction["folds"], dtype=np.int64),
        probabilities=np.asarray(extraction["probabilities"], dtype=np.float64),
        log_probabilities=np.asarray(
            extraction["log_probabilities"], dtype=np.float64
        ),
        **{
            f"descriptor_{role}": np.asarray(
                extraction["descriptors"][role], dtype=np.float32
            )
            for role in ROLE_NAMES
        },
    )


def _load_descriptor_payload(path: Path) -> Dict[str, object]:
    with np.load(path, allow_pickle=False) as payload:
        return {
            "indices": payload["indices"].astype(np.int64),
            "targets": payload["targets"].astype(np.int64),
            "folds": payload["folds"].astype(np.int64),
            "probabilities": payload["probabilities"].astype(np.float64),
            "log_probabilities": payload["log_probabilities"].astype(np.float64),
            "descriptors": {
                role: payload[f"descriptor_{role}"].astype(np.float32)
                for role in ROLE_NAMES
            },
        }


def _oof_prediction_rows(
    *,
    cohort: Sequence[CleanTrainRow],
    scores: Mapping[str, np.ndarray],
    actions: Mapping[str, np.ndarray],
) -> List[Dict[str, object]]:
    rows = []
    for position, source in enumerate(cohort):
        row: Dict[str, object] = {
            "sample_index": int(source.sample_index),
            "source_stem": source.source_stem,
            "image_path": str(source.image_path),
            "fold": int(source.fold),
            "target": int(source.target),
            "binary_true_class1": int(source.target == FOCUS_CLASS),
        }
        for role in ALL_READOUT_ROLES:
            row[f"{role}_score"] = float(scores[role][position])
            row[f"{role}_keep_class1"] = int(bool(actions[role][position]))
        rows.append(row)
    return rows


def _fold_summary_rows(analysis: Mapping[str, object]) -> List[Dict[str, object]]:
    rows = []
    metrics = analysis["metrics"]
    for role in ALL_READOUT_ROLES:
        for fold in metrics[role]["per_fold"]:
            rows.append({"role": role, **fold})
    return rows


def _selector_summary_from_telemetry(
    telemetry: Sequence[Mapping[str, object]],
) -> Dict[str, object]:
    roles = {}
    for role in ROLE_NAMES:
        roles[role] = {
            "foreground_mean": float(
                np.mean([row[f"{role}_foreground_mean"] for row in telemetry])
            ),
            "bbox_mean": float(
                np.mean([row[f"{role}_bbox_mean"] for row in telemetry])
            ),
            "foreground_mass_ratio": float(
                np.mean(
                    [row[f"{role}_foreground_mass_ratio"] for row in telemetry]
                )
            ),
            "bbox_mass_ratio": float(
                np.mean([row[f"{role}_bbox_mass_ratio"] for row in telemetry])
            ),
            "alignment_score": float(
                np.mean([row[f"{role}_alignment"] for row in telemetry])
            ),
            "top_score_tie_fraction": float(
                np.mean([int(row[f"{role}_top_ties"]) > 1 for row in telemetry])
            ),
            "mean_top_score_ties": float(
                np.mean([row[f"{role}_top_ties"] for row in telemetry])
            ),
        }
    return {
        "roles": roles,
        "mean_head_topk_jaccard": float(
            np.mean([row["head_topk_mean_jaccard"] for row in telemetry])
        ),
        "mean_mhv_vote_entropy": float(
            np.mean([row["mhv_vote_entropy"] for row in telemetry])
        ),
    }


def _load_keeper(path: Path, device: torch.device):
    checkpoint = load_checkpoint(Path(path), map_location="cpu")
    model = _load_keeper_model(checkpoint).to(device).eval()
    class_names = [str(value) for value in checkpoint.get("class_names", [])]
    if len(class_names) != 5:
        raise ValueError("Keeper checkpoint must declare five classes")
    if len(getattr(model, "blocks", [])) != BLOCK_COUNT:
        raise ValueError("Keeper block count differs from the MHV lock")
    if int(getattr(model, "num_prefix_tokens", -1)) != 7:
        raise ValueError("Keeper prefix count differs from the MHV lock")
    if not bool(getattr(model, "token_pruning", False)):
        raise ValueError("Keeper token pruning must remain enabled")
    schedule = {
        int(key): float(value)
        for key, value in getattr(model, "token_prune_schedule", {}).items()
    }
    if schedule != {1: 0.85, 4: 0.65}:
        raise ValueError(f"Keeper pruning schedule differs: {schedule}")
    return model, checkpoint, class_names


def synthetic_hook_preflight(
    model: nn.Module, *, device: torch.device
) -> Dict[str, object]:
    state_before = _model_state_sha256(model)
    generator = torch.Generator(device="cpu").manual_seed(SEED + 131)
    images = torch.randn(
        (2, 3, 256, 256), generator=generator, dtype=torch.float32
    ).to(device)
    metadata = {
        "bbox": torch.tensor(
            [[0.50, 0.50, 0.60, 0.55], [0.45, 0.52, 0.48, 0.62]],
            dtype=torch.float32,
        ),
        "image_mask": torch.ones((2, 1, 256, 256), dtype=torch.bool),
    }
    if device.type == "cuda":
        torch.cuda.empty_cache()
        torch.cuda.reset_peak_memory_stats(device)
    capture = FrozenMHVCapture(model)
    try:
        with torch.inference_mode():
            with torch.autocast(
                device_type="cuda", dtype=torch.bfloat16, enabled=True
            ):
                ordinary_logits, ordinary_features = (
                    _forward_classification_with_metadata(
                        model, images, metadata, device=device
                    )
                )
            ordinary_probabilities = ordinary_logits.float().softmax(dim=1)
            del ordinary_features, ordinary_logits
            capture.start()
            with torch.autocast(
                device_type="cuda", dtype=torch.bfloat16, enabled=True
            ):
                traced_logits, features = _trace_forward(
                    model, images, metadata, device=device
                )
            captured = capture.consume(features["trace"])
            traced_probabilities = traced_logits.float().softmax(dim=1)
            descriptor_shapes = {}
            oracle_exact = True
            all_finite = bool(torch.isfinite(traced_probabilities).all())
            for block_index in range(BLOCK_COUNT):
                selector = selector_bank_torch(
                    captured["attentions"][block_index],
                    captured["before_indices"][block_index],
                    prefix_count=capture.prefix_count,
                    quota=LAYER_QUOTAS[block_index],
                )
                oracle = selector_bank_numpy(
                    selector["cls_attention"].numpy(),
                    captured["before_indices"][block_index].numpy(),
                    quota=LAYER_QUOTAS[block_index],
                )
                for role in ("vote_raw", "vote_dephased_smooth", "mhv_smooth"):
                    oracle_exact = oracle_exact and bool(
                        np.array_equal(
                            selector["scores"][role].numpy(),
                            oracle["scores"][role],
                        )
                        and np.array_equal(
                            selector["selected_original"][role].numpy(),
                            oracle["selected_original"][role],
                        )
                    )
                for role in ROLE_NAMES:
                    descriptor = block_role_descriptor(
                        captured["patch_tokens"][block_index],
                        selector["selected_local"][role],
                        block_index=block_index,
                    )
                    descriptor_shapes[f"{block_index}:{role}"] = list(
                        descriptor.shape
                    )
                    all_finite = all_finite and bool(torch.isfinite(descriptor).all())
            probability_error = float(
                (ordinary_probabilities - traced_probabilities).abs().max().item()
            )
            argmax_exact = bool(
                torch.equal(
                    ordinary_probabilities.argmax(dim=1),
                    traced_probabilities.argmax(dim=1),
                )
            )
            token_counts = {
                "entering": list(captured["entering_patch_counts"]),
                "after": list(captured["after_patch_counts"]),
            }
    finally:
        hook_lifecycle = capture.close()
    if device.type == "cuda":
        torch.cuda.synchronize(device)
        peak_cuda_bytes = int(torch.cuda.max_memory_allocated(device))
    else:
        peak_cuda_bytes = 0
    state_after = _model_state_sha256(model)
    checks = {
        "normal_forward_probability_error_le_1e_6": probability_error
        <= MAX_HOOK_PROBABILITY_ERROR,
        "normal_forward_argmax_exact": argmax_exact,
        "expected_token_lineage": token_counts
        == {
            "entering": list(EXPECTED_ENTERING_PATCH_COUNTS),
            "after": list(EXPECTED_AFTER_PATCH_COUNTS),
        },
        "numpy_oracle_exact": oracle_exact,
        "bf16_model_and_fp32_descriptors_finite": all_finite,
        "model_state_bit_exact": state_before == state_after,
        "hooks_removed_exactly": bool(hook_lifecycle["no_leak"]),
        "peak_cuda_allocation_le_3_5_gib": peak_cuda_bytes
        <= int(MAX_PEAK_CUDA_GIB * 1024**3),
    }
    return {
        "checks": checks,
        "passed": all(checks.values()),
        "probability_max_abs_error": probability_error,
        "token_counts": token_counts,
        "descriptor_shapes": descriptor_shapes,
        "peak_cuda_bytes": peak_cuda_bytes,
        "peak_cuda_gib": float(peak_cuda_bytes / 1024**3),
        "model_state_before_sha256": state_before,
        "model_state_after_sha256": state_after,
        "hook_lifecycle": hook_lifecycle,
    }


def preflight(args: argparse.Namespace) -> Dict[str, object]:
    _validate_locked_args(args)
    output_dir = Path(args.output_dir).expanduser().resolve()
    if output_dir.exists():
        raise FileExistsError(f"Preflight output must not exist: {output_dir}")
    if not torch.cuda.is_available():
        raise RuntimeError("CUDA is unavailable for IELT MHV preflight")
    provenance = verify_locked_inputs(args)
    engineering = mhv_engineering_checks()
    if not bool(engineering["all_passed"]):
        raise RuntimeError(f"MHV engineering checks failed: {engineering}")
    set_seed(SEED, deterministic=True)
    torch.backends.cuda.matmul.allow_tf32 = False
    torch.backends.cudnn.allow_tf32 = True
    device = torch.device("cuda")
    model, _, _ = _load_keeper(
        Path(provenance["files"]["keeper"]["path"]), device
    )
    hook_preflight = synthetic_hook_preflight(model, device=device)
    if not bool(hook_preflight["passed"]):
        raise RuntimeError(f"Synthetic hook preflight failed: {hook_preflight}")
    return {
        "mode": "ielt_mhv_signal_a0_preflight",
        "output_created": False,
        "dataset_pixels_loaded": False,
        "validation_data_used": False,
        "test_data_used": False,
        "locked_inputs_verified": True,
        "engineering": engineering,
        "synthetic_hook_preflight": hook_preflight,
        "repo_state_observed_not_gated_until_formal": _repo_state(),
        "provenance": provenance,
    }


def _states_converged(states: Mapping[str, object]) -> bool:
    return all(
        bool(fold["converged"])
        for role in ALL_READOUT_ROLES
        for fold in states[role]["folds"]
    )


def _maximum_score_difference(
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
        for role in ALL_READOUT_ROLES
    )


def run_audit(args: argparse.Namespace) -> Dict[str, object]:
    _validate_locked_args(args)
    if not torch.cuda.is_available():
        raise RuntimeError("CUDA is unavailable for formal IELT MHV A0")
    provenance = verify_locked_inputs(args)
    repo_state = _repo_state()
    if not bool(repo_state["tracked_worktree_clean"]) or not bool(
        repo_state["head_matches_upstream"]
    ):
        raise ValueError(
            f"Formal IELT MHV A0 requires clean pushed tracked state: {repo_state}"
        )
    output_dir = _prepare_output_dir(args.output_dir)
    set_seed(SEED, deterministic=True)
    torch.backends.cuda.matmul.allow_tf32 = False
    torch.backends.cudnn.allow_tf32 = True
    device = torch.device("cuda")
    engineering = mhv_engineering_checks()
    if not bool(engineering["all_passed"]):
        raise RuntimeError(f"MHV engineering checks failed: {engineering}")

    model, checkpoint, class_names = _load_keeper(
        Path(provenance["files"]["keeper"]["path"]), device
    )
    all_rows = _read_clean_train_rows(
        Path(provenance["files"]["cidt_predictions"]["path"])
    )
    cohort = _cohort_from_rows(all_rows)
    base_dataset, transform, dataset_summary = _build_dataset(
        checkpoint,
        all_rows,
        Path(provenance["files"]["data_yaml"]["path"]),
    )
    review_indices = _review_indices(cohort)
    clean_loader, clean_loader_summary = _make_condition_loader(
        base_dataset=base_dataset,
        transform=transform,
        indices=[row.sample_index for row in cohort],
        brightness=1.0,
        contrast=1.0,
        batch_size=BATCH_SIZE,
        num_workers=NUM_WORKERS,
        context="ielt_mhv_signal_a0_clean",
    )
    clean = extract_condition(
        model=model,
        loader=clean_loader,
        loader_summary=clean_loader_summary,
        cohort=cohort,
        device=device,
        condition="clean",
        check_ordinary=True,
        review_indices=review_indices,
    )
    targets = np.asarray(clean["targets"], dtype=np.int64)
    folds = np.asarray(clean["folds"], dtype=np.int64)
    labels = (targets == FOCUS_CLASS).astype(np.int64)
    descriptor_stats = {
        role: descriptor_statistics(clean["descriptors"][role])
        for role in ROLE_NAMES
    }
    similarities = role_similarity(clean["descriptors"])
    clean_features = build_readout_features(
        clean["descriptors"], clean["log_probabilities"]
    )
    clean_scores, clean_actions, readout_states = fit_clean_oof_readouts(
        clean_features, labels, folds, seed=SEED
    )
    clean_analysis = build_analysis(
        labels=labels,
        targets=targets,
        folds=folds,
        scores=clean_scores,
        actions=clean_actions,
        descriptor_stats=descriptor_stats,
        selector_summary=clean["selector_summary"],
    )

    descriptor_path = output_dir / "descriptors_clean.npz"
    prediction_path = output_dir / "oof_predictions_clean.csv"
    readout_path = output_dir / "readout_state_clean.json"
    telemetry_path = output_dir / "selector_telemetry_clean.csv"
    fold_path = output_dir / "fold_summary_clean.csv"
    contact_path = output_dir / "mhv_contact_sheet_clean.png"
    _save_descriptor_payload(descriptor_path, clean)
    _write_csv(
        prediction_path,
        _oof_prediction_rows(
            cohort=cohort, scores=clean_scores, actions=clean_actions
        ),
    )
    _write_json(readout_path, readout_states)
    _write_csv(telemetry_path, clean["telemetry"])
    _write_csv(fold_path, _fold_summary_rows(clean_analysis))
    contact = render_contact_sheet(
        contact_path,
        visual_cache=clean["visual_cache"],
        review_indices=review_indices,
        cohort=cohort,
        semantics=dataset_summary["semantics"],
    )

    replay_payload = _load_descriptor_payload(descriptor_path)
    replay_features = build_readout_features(
        replay_payload["descriptors"], replay_payload["log_probabilities"]
    )
    replay_scores, replay_actions, replay_states = fit_clean_oof_readouts(
        replay_features, labels, folds, seed=SEED
    )
    applied_scores, applied_actions = apply_clean_readout_states(
        replay_features, folds, readout_states
    )
    replay_descriptor_stats = {
        role: descriptor_statistics(replay_payload["descriptors"][role])
        for role in ROLE_NAMES
    }
    replay_analysis = build_analysis(
        labels=labels,
        targets=targets,
        folds=folds,
        scores=replay_scores,
        actions=replay_actions,
        descriptor_stats=replay_descriptor_stats,
        selector_summary=clean["selector_summary"],
    )
    replay_difference = _recursive_numeric_difference(
        clean_analysis, replay_analysis
    )
    replay_state_difference = _recursive_numeric_difference(
        readout_states, replay_states
    )
    replay_score_difference = _maximum_score_difference(
        clean_scores, replay_scores
    )
    applied_score_difference = _maximum_score_difference(
        clean_scores, applied_scores
    )
    applied_actions_exact = all(
        np.array_equal(clean_actions[role], applied_actions[role])
        for role in ALL_READOUT_ROLES
    )
    clean_replay_passed = bool(
        replay_difference <= 1e-12
        and replay_state_difference <= 1e-12
        and replay_score_difference <= 1e-12
        and applied_score_difference <= 1e-10
        and applied_actions_exact
    )

    loader_effective_workers = int(
        clean["runtime"]["loader"].get("effective_num_workers", -1)
    )
    clean_structural_gates = {
        "locked_hashes_verified": True,
        "official_repository_exact_and_clean": not bool(
            provenance["official_repository"]["status"]
        ),
        "protected_untracked_identity_exact": bool(
            provenance["protected_untracked"]["passed"]
        ),
        "repo_clean_and_pushed": bool(repo_state["tracked_worktree_clean"])
        and bool(repo_state["head_matches_upstream"]),
        "train_split_only": True,
        "dataset_mapping_exact": bool(dataset_summary["paths_exact"])
        and bool(dataset_summary["train_paths_only"]),
        "cohort_rows_exact": len(cohort) == EXPECTED_COHORT_ROWS,
        "cohort_order_hash_exact": _ordered_index_sha256(
            [row.sample_index for row in cohort]
        )
        == EXPECTED_ORDERED_INDEX_SHA256,
        "binary_support_exact": int(labels.sum()) == EXPECTED_POSITIVES
        and int((labels == 0).sum()) == EXPECTED_NEGATIVES,
        "normal_forward_probability_error_le_1e_6": float(
            clean["forward_replay"]["hook_probability_max_abs_error"]
        )
        <= MAX_HOOK_PROBABILITY_ERROR,
        "normal_forward_argmax_exact": bool(
            clean["forward_replay"]["hook_argmax_exact"]
        ),
        "cidt_declaration_exception_exact": bool(
            clean["forward_replay"][
                "cidt_declarations_match_locked_exception"
            ]
        ),
        "bbox_bytes_hash_exact": clean["forward_replay"]["bbox_bytes_sha256"]
        == LOCKED_BBOX_BYTES_SHA256,
        "token_lineage_exact": clean["runtime"]["token_counts"]
        == {
            "entering": list(EXPECTED_ENTERING_PATCH_COUNTS),
            "after": list(EXPECTED_AFTER_PATCH_COUNTS),
        },
        "numpy_oracle_score_exact": float(
            clean["selector_summary"]["oracle_score_max_abs_error"]
        )
        == 0.0,
        "numpy_oracle_indices_exact": bool(
            clean["selector_summary"]["oracle_selected_indices_exact"]
        ),
        "all_descriptors_finite": all(
            bool(descriptor_stats[role]["all_values_finite"])
            for role in ROLE_NAMES
        ),
        "all_20_readouts_converged": _states_converged(readout_states),
        "all_oof_scores_finite": all(
            np.isfinite(clean_scores[role]).all() for role in ALL_READOUT_ROLES
        ),
        "model_state_bit_exact": clean["runtime"][
            "model_state_before_sha256"
        ]
        == clean["runtime"]["model_state_after_sha256"],
        "hooks_removed_exactly": bool(
            clean["runtime"]["hook_lifecycle"]["no_leak"]
        ),
        "requested_workers_4_effective_workers_4": NUM_WORKERS == 4
        and loader_effective_workers == 4,
        "peak_cuda_allocation_le_3_5_gib": int(
            clean["runtime"]["peak_cuda_bytes"]
        )
        <= int(MAX_PEAK_CUDA_GIB * 1024**3),
        "descriptor_readout_replay_exact": clean_replay_passed,
        "contact_sheet_12_rows_5_columns": len(contact["rows"]) == 12
        and len(contact["columns"]) == 5,
    }
    clean_structural_passed = all(clean_structural_gates.values())
    clean_automatic_passed = clean_structural_passed and bool(
        clean_analysis["mechanism_gates_passed"]
    )

    condition_summaries: Dict[str, object] = {}
    condition_artifacts: Dict[str, object] = {}
    condition_replay: Dict[str, object] = {}
    if clean_automatic_passed:
        for name, brightness, contrast in CONDITIONS:
            loader, loader_summary = _make_condition_loader(
                base_dataset=base_dataset,
                transform=transform,
                indices=[row.sample_index for row in cohort],
                brightness=brightness,
                contrast=contrast,
                batch_size=BATCH_SIZE,
                num_workers=NUM_WORKERS,
                context=f"ielt_mhv_signal_a0_{name}",
            )
            extraction = extract_condition(
                model=model,
                loader=loader,
                loader_summary=loader_summary,
                cohort=cohort,
                device=device,
                condition=name,
                check_ordinary=False,
                review_indices=(),
            )
            condition_features = build_readout_features(
                extraction["descriptors"], extraction["log_probabilities"]
            )
            condition_scores, condition_actions = apply_clean_readout_states(
                condition_features, folds, readout_states
            )
            condition_descriptor_stats = {
                role: descriptor_statistics(extraction["descriptors"][role])
                for role in ROLE_NAMES
            }
            condition_analysis = build_analysis(
                labels=labels,
                targets=targets,
                folds=folds,
                scores=condition_scores,
                actions=condition_actions,
                descriptor_stats=condition_descriptor_stats,
                selector_summary=extraction["selector_summary"],
            )
            condition_gate_result = condition_gate(condition_analysis)
            condition_structural = {
                "indices_targets_folds_exact": bool(
                    np.array_equal(extraction["indices"], clean["indices"])
                    and np.array_equal(extraction["targets"], clean["targets"])
                    and np.array_equal(extraction["folds"], clean["folds"])
                ),
                "model_state_bit_exact": extraction["runtime"][
                    "model_state_before_sha256"
                ]
                == extraction["runtime"]["model_state_after_sha256"],
                "hooks_removed_exactly": bool(
                    extraction["runtime"]["hook_lifecycle"]["no_leak"]
                ),
                "numpy_oracle_exact": float(
                    extraction["selector_summary"]["oracle_score_max_abs_error"]
                )
                == 0.0
                and bool(
                    extraction["selector_summary"][
                        "oracle_selected_indices_exact"
                    ]
                ),
                "effective_workers_4": int(
                    extraction["runtime"]["loader"].get(
                        "effective_num_workers", -1
                    )
                )
                == 4,
                "peak_cuda_allocation_le_3_5_gib": int(
                    extraction["runtime"]["peak_cuda_bytes"]
                )
                <= int(MAX_PEAK_CUDA_GIB * 1024**3),
                "all_descriptors_finite": all(
                    bool(condition_descriptor_stats[role]["all_values_finite"])
                    for role in ROLE_NAMES
                ),
            }
            condition_passed = all(condition_structural.values()) and bool(
                condition_gate_result["passed"]
            )
            condition_descriptor_path = output_dir / f"descriptors_{name}.npz"
            condition_prediction_path = output_dir / f"oof_predictions_{name}.csv"
            condition_telemetry_path = output_dir / f"selector_telemetry_{name}.csv"
            condition_fold_path = output_dir / f"fold_summary_{name}.csv"
            _save_descriptor_payload(condition_descriptor_path, extraction)
            _write_csv(
                condition_prediction_path,
                _oof_prediction_rows(
                    cohort=cohort,
                    scores=condition_scores,
                    actions=condition_actions,
                ),
            )
            _write_csv(condition_telemetry_path, extraction["telemetry"])
            _write_csv(condition_fold_path, _fold_summary_rows(condition_analysis))

            loaded = _load_descriptor_payload(condition_descriptor_path)
            loaded_features = build_readout_features(
                loaded["descriptors"], loaded["log_probabilities"]
            )
            replay_condition_scores, replay_condition_actions = (
                apply_clean_readout_states(loaded_features, folds, readout_states)
            )
            replay_condition_analysis = build_analysis(
                labels=labels,
                targets=targets,
                folds=folds,
                scores=replay_condition_scores,
                actions=replay_condition_actions,
                descriptor_stats={
                    role: descriptor_statistics(loaded["descriptors"][role])
                    for role in ROLE_NAMES
                },
                selector_summary=extraction["selector_summary"],
            )
            difference = _recursive_numeric_difference(
                condition_analysis, replay_condition_analysis
            )
            condition_replay[name] = {
                "maximum_numeric_difference": float(difference),
                "passed": bool(difference <= 1e-12),
            }
            condition_summaries[name] = {
                "brightness": brightness,
                "contrast": contrast,
                "runtime": extraction["runtime"],
                "forward_replay": extraction["forward_replay"],
                "selector_summary": extraction["selector_summary"],
                "descriptor_statistics": condition_descriptor_stats,
                "analysis": condition_analysis,
                "condition_gate": condition_gate_result,
                "structural_gates": condition_structural,
                "passed": condition_passed
                and bool(condition_replay[name]["passed"]),
            }
            condition_artifacts[name] = {
                "descriptors": _artifact_record(
                    condition_descriptor_path, root=output_dir
                ),
                "predictions": _artifact_record(
                    condition_prediction_path, root=output_dir
                ),
                "telemetry": _artifact_record(
                    condition_telemetry_path, root=output_dir
                ),
                "fold_summary": _artifact_record(
                    condition_fold_path, root=output_dir
                ),
            }

    all_conditions_passed = bool(
        clean_automatic_passed
        and len(condition_summaries) == len(CONDITIONS)
        and all(bool(value["passed"]) for value in condition_summaries.values())
    )
    replay_summary_payload = {
        "clean": {
            "analysis_maximum_numeric_difference": float(replay_difference),
            "state_maximum_numeric_difference": float(replay_state_difference),
            "refit_score_maximum_abs_difference": float(replay_score_difference),
            "serialized_state_score_maximum_abs_difference": float(
                applied_score_difference
            ),
            "serialized_state_actions_exact": applied_actions_exact,
            "passed": clean_replay_passed,
        },
        "conditions": condition_replay,
        "passed": bool(
            clean_replay_passed
            and all(bool(value["passed"]) for value in condition_replay.values())
        ),
    }
    replay_path = output_dir / "replay_summary.json"
    _write_json(replay_path, replay_summary_payload)

    artifacts = {
        "descriptors_clean": _artifact_record(descriptor_path, root=output_dir),
        "oof_predictions_clean": _artifact_record(
            prediction_path, root=output_dir
        ),
        "readout_state_clean": _artifact_record(readout_path, root=output_dir),
        "selector_telemetry_clean": _artifact_record(
            telemetry_path, root=output_dir
        ),
        "fold_summary_clean": _artifact_record(fold_path, root=output_dir),
        "contact_sheet_clean": _artifact_record(contact_path, root=output_dir),
        "replay_summary": _artifact_record(replay_path, root=output_dir),
        "conditions": condition_artifacts,
    }
    automated_gate_passed = bool(all_conditions_passed)
    summary = {
        "mode": "ielt_mhv_signal_a0_train_information_gate",
        "status": "awaiting_visual_review"
        if automated_gate_passed
        else "rejected_automated_gate",
        "split": "train",
        "rows": EXPECTED_COHORT_ROWS,
        "validation_data_used": False,
        "test_data_used": False,
        "raw_dataset_modified": False,
        "class_names": class_names,
        "focus_class": FOCUS_CLASS,
        "restricted_negative_classes": list(RESTRICTED_NEGATIVE_CLASSES),
        "cohort": {
            "rows": len(cohort),
            "positives": int(labels.sum()),
            "negatives": int((labels == 0).sum()),
            "ordered_index_sha256": _ordered_index_sha256(
                [row.sample_index for row in cohort]
            ),
            "fold_counts": EXPECTED_FOLD_COUNTS,
        },
        "locked_adaptation": {
            "grid_size": GRID_SIZE,
            "heads": HEAD_COUNT,
            "layer_quotas": list(LAYER_QUOTAS),
            "vote_per_head_by_active_patches": {
                str(value): scaled_vote_count(value) for value in (256, 218, 167)
            },
            "kernel": [list(row) for row in MHV_KERNEL],
            "dephase_offsets": [list(value) for value in DEPHASE_OFFSETS],
            "descriptor_dimensions": ROLE_DIM,
        },
        "provenance": provenance,
        "repo_state": repo_state,
        "dataset": dataset_summary,
        "engineering": engineering,
        "clean": {
            "runtime": clean["runtime"],
            "forward_replay": clean["forward_replay"],
            "selector_summary": clean["selector_summary"],
            "descriptor_statistics": descriptor_stats,
            "role_similarity": similarities,
            "analysis": clean_analysis,
            "structural_gates": clean_structural_gates,
            "structural_gates_passed": clean_structural_passed,
            "automatic_gate_passed": clean_automatic_passed,
        },
        "conditions_status": "completed"
        if clean_automatic_passed
        else "skipped_clean_gate_failed",
        "conditions": condition_summaries,
        "all_conditions_passed": all_conditions_passed,
        "contact_sheet": contact,
        "replay": replay_summary_payload,
        "automated_gate_passed": automated_gate_passed,
        "visual_review": {
            "required": automated_gate_passed,
            "completed": False,
            "passed": None,
        },
        "trainer_integration_authorized": False,
        "short_pair_authorized": False,
        "probe_authorized": False,
        "full_train_authorized": False,
        "current_command_update_authorized": False,
        "artifacts": artifacts,
        "guardrail": (
            "A0 may only authorize one later default-off implementation and matched "
            "short pair after an automated pass plus manual fixed-sheet review. "
            "Validation, test, full train, and current-best command promotion remain "
            "forbidden here."
        ),
    }
    summary_path = output_dir / "summary.json"
    _write_json(summary_path, summary)
    _write_manifest(output_dir)
    return summary


def replay_summary(summary_path: Path) -> Dict[str, object]:
    resolved = Path(summary_path).expanduser().resolve()
    if not resolved.is_file() or resolved.name != "summary.json":
        raise FileNotFoundError(f"IELT MHV summary is missing: {resolved}")
    root = resolved.parent
    manifest = _verify_manifest(root)
    summary = json.loads(resolved.read_text(encoding="utf-8"))
    artifacts = summary["artifacts"]
    clean_payload = _load_descriptor_payload(
        root / artifacts["descriptors_clean"]["path"]
    )
    states = json.loads(
        (root / artifacts["readout_state_clean"]["path"]).read_text(
            encoding="utf-8"
        )
    )
    targets = np.asarray(clean_payload["targets"], dtype=np.int64)
    folds = np.asarray(clean_payload["folds"], dtype=np.int64)
    labels = (targets == FOCUS_CLASS).astype(np.int64)
    features = build_readout_features(
        clean_payload["descriptors"], clean_payload["log_probabilities"]
    )
    scores, actions, replay_states = fit_clean_oof_readouts(
        features, labels, folds, seed=SEED
    )
    applied_scores, applied_actions = apply_clean_readout_states(
        features, folds, states
    )
    stats = {
        role: descriptor_statistics(clean_payload["descriptors"][role])
        for role in ROLE_NAMES
    }
    replayed_analysis = build_analysis(
        labels=labels,
        targets=targets,
        folds=folds,
        scores=scores,
        actions=actions,
        descriptor_stats=stats,
        selector_summary=summary["clean"]["selector_summary"],
    )
    clean_difference = _recursive_numeric_difference(
        summary["clean"]["analysis"], replayed_analysis
    )
    state_difference = _recursive_numeric_difference(states, replay_states)
    serialized_score_difference = _maximum_score_difference(scores, applied_scores)
    serialized_actions_exact = all(
        np.array_equal(actions[role], applied_actions[role])
        for role in ALL_READOUT_ROLES
    )
    conditions = {}
    for name, condition_summary in summary.get("conditions", {}).items():
        condition_record = artifacts["conditions"][name]["descriptors"]
        payload = _load_descriptor_payload(root / condition_record["path"])
        condition_features = build_readout_features(
            payload["descriptors"], payload["log_probabilities"]
        )
        condition_scores, condition_actions = apply_clean_readout_states(
            condition_features, folds, states
        )
        condition_analysis = build_analysis(
            labels=labels,
            targets=targets,
            folds=folds,
            scores=condition_scores,
            actions=condition_actions,
            descriptor_stats={
                role: descriptor_statistics(payload["descriptors"][role])
                for role in ROLE_NAMES
            },
            selector_summary=condition_summary["selector_summary"],
        )
        difference = _recursive_numeric_difference(
            condition_summary["analysis"], condition_analysis
        )
        conditions[name] = {
            "maximum_numeric_difference": float(difference),
            "passed": bool(difference <= 1e-12),
        }
    passed = bool(
        clean_difference <= 1e-12
        and state_difference <= 1e-12
        and serialized_score_difference <= 1e-10
        and serialized_actions_exact
        and all(bool(value["passed"]) for value in conditions.values())
    )
    return {
        "mode": "ielt_mhv_signal_a0_external_replay",
        "passed": passed,
        "clean_analysis_maximum_numeric_difference": float(clean_difference),
        "clean_state_maximum_numeric_difference": float(state_difference),
        "serialized_state_score_maximum_abs_difference": float(
            serialized_score_difference
        ),
        "serialized_state_actions_exact": serialized_actions_exact,
        "conditions": conditions,
        "manifest": manifest,
        "validation_data_used": False,
        "test_data_used": False,
    }


def main(argv: Optional[Sequence[str]] = None) -> int:
    args = parse_args(argv)
    if args.preflight_only and args.replay_summary is not None:
        raise ValueError("preflight-only cannot be combined with replay-summary")
    if args.preflight_only:
        print(json.dumps(to_serializable(preflight(args)), indent=2), flush=True)
        return 0
    if args.replay_summary is not None:
        result = replay_summary(args.replay_summary)
        print(json.dumps(to_serializable(result), indent=2), flush=True)
        return 0
    summary = run_audit(args)
    print(
        json.dumps(
            to_serializable(
                {
                    "status": summary["status"],
                    "rows": summary["rows"],
                    "clean_candidate_metrics": summary["clean"]["analysis"][
                        "metrics"
                    ]["mhv_smooth"],
                    "clean_candidate_auroc_gains": summary["clean"][
                        "analysis"
                    ]["candidate_auroc_gains"],
                    "failed_clean_structural_gates": sorted(
                        key
                        for key, passed in summary["clean"][
                            "structural_gates"
                        ].items()
                        if not passed
                    ),
                    "failed_clean_mechanism_gates": sorted(
                        key
                        for key, passed in summary["clean"]["analysis"][
                            "mechanism_gates"
                        ].items()
                        if not passed
                    ),
                    "conditions_status": summary["conditions_status"],
                    "full_train_authorized": False,
                    "current_command_update_authorized": False,
                }
            ),
            indent=2,
        ),
        flush=True,
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
