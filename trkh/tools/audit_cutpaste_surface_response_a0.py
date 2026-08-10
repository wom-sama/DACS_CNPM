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

import torch
import torch.nn.functional as F
import torchvision
from sklearn.exceptions import ConvergenceWarning
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import roc_auc_score
from sklearn.preprocessing import StandardScaler
from torch import Tensor, nn
from torch.utils.data import DataLoader
from torchvision.transforms import InterpolationMode
from torchvision.transforms import functional as TVF

from trkh.core.config import load_data_spec, to_serializable
from trkh.core.utils import build_safe_dataloader_kwargs, load_checkpoint, set_seed
from trkh.data.dataset import (
    MangoYOLOCropDataset,
    _surface_detail_foreground_mask_array,
)
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


METHOD = "cutpaste_surface_response_a0"
SEED = 20260720
BATCH_SIZE = 32
NUM_WORKERS = 4
FOCUS_CLASS = 1
RESTRICTED_NEGATIVE_CLASSES = (0, 2, 4)
FOLDS = (0, 1, 2, 3, 4)
FAMILIES = ("regular", "scar")
DRAW_COUNT = 4
SURFACE_MASK_MARGIN = 0.08
SUPPORT_ELLIPSE_RADIUS_FACTOR = 0.92
SUPPORT_EROSION_PIXELS = 2
RESPONSE_NAMES = (
    "class1_logit_delta",
    "class1_restricted_margin_delta",
    "probability_js_divergence",
    "pooled_cosine_distance",
    "pooled_l2_distance",
)
AGGREGATIONS = ("mean", "std_population", "minimum", "maximum")
ROLE_NAMES = (
    "base_only",
    "cutout_control",
    "cutpaste_candidate",
    "paired_contrast",
    "source_deranged",
)
BASE_DIM = 6
RESPONSE_DIM = len(FAMILIES) * len(RESPONSE_NAMES) * len(AGGREGATIONS)
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

READOUT_C = 0.1
READOUT_MAX_ITER = 4000
READOUT_TOLERANCE = 1e-9
MIN_FIT_TP_RETENTION = 0.97
LOG_PROBABILITY_CLIP = 1e-8
MAX_CIDT_PROBABILITY_ERROR = 2e-5
MAX_PEAK_CUDA_GIB = 3.5
CONDITIONS = (
    ("lighting_dim", 0.70, 0.90, 1),
    ("lighting_bright", 1.25, 1.10, 2),
    ("low_contrast", 1.00, 0.65, 3),
)

LOCKED_KEEPER_SHA256 = "1f49d577240c69dc63c30af70db52ec2aa9da65a17aef1c4b1c09ece6c482677"
LOCKED_LAUNCHER_ARGS_SHA256 = "908a05cf66b2a01162cae62e4ff2251eaae1297d31e70510144e4954159b7eff"
LOCKED_RESOLVED_CONFIG_SHA256 = "e9c4f48917e333d2f34f61806bb54041b35f2217ebb23afb3bd0ced969854674"
LOCKED_DATA_SHA256 = "716e33df24c63a9e9920f97b685199707fb84ab4c7154544f5dd9a3e00d884ef"
LOCKED_CIDT_SUMMARY_SHA256 = "d4891edf2963ab12385b7ce5bdc812ec3e19c5c098acd25c66eb557af541d7ad"
LOCKED_CIDT_PREDICTIONS_SHA256 = "2e0993752d58d99ea429bfefe1e2bfe6fa949e45aea1a26cc4bdfee97d4db21c"
LOCKED_PROTOCOL_SHA256 = "31017e378308a962c00640233185e6c36db094adbded21433cc1b474ef2256d1"
LOCKED_CURRENT_COMMAND_SHA256 = "36b9aa1a21b765829acf4c8321be147bd76297de4ccdb8a40e6dee8e37940faf"
LOCKED_COMMAND_HISTORY_SHA256 = "39bd2879ce66fddf36a953021ea1e40f8d9de6cb4334b9b825011b2b8dc98f53"
LOCKED_PAPER_SHA256 = "e40ec13e20ded2a5307cd2cd4ddb04027ad3f657a2b5490bdcbb5892d81617f4"
LOCKED_SUPPLEMENTAL_SHA256 = "fa89f983a759076082b5be87c92cbe4fdf45ba958175f4b48fed736051f41ea6"
LOCKED_TORCHVISION_COMMIT = "7af698794eded568735f9519593603c1ec889eba"
LOCKED_TORCHVISION_TREE = "9188eecffeb7a0febd04dea72028a9b59dfece0a"
LOCKED_TORCHVISION_LICENSE_SHA256 = "c06363f9d33627dc5173b13522444cc85fbb4739f63e16d4587d5c0a165b5b1e"
LOCKED_TORCHVISION_TRANSFORMS_SHA256 = "75c1d80d921b60e5bab0ecb01f8d6e3a205df6d3df12039d7c56d2cea38db5f0"

REPO_ROOT = Path(__file__).resolve().parents[2]
OFFICIAL_ROOT = Path(r"D:\DataAI\external_sources\official\torchvision-v0.21.0")
PAPER_ROOT = Path(r"D:\DataAI\external_sources\official\cutpaste-cvpr2021")
KEEPER_ROOT = (
    REPO_ROOT
    / "runs"
    / "probe_v8_yolof_pairroute_teacherfocusbinary015_boundarydrop_bboxprior_120b_2e_20260701"
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
        description="Locked train-only object-interior CutPaste response gate."
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
        / "TRKH_5CLASS_CUTPASTE_SURFACE_RESPONSE_A0_PROTOCOL_20260720.md",
    )
    parser.add_argument("--official-root", type=Path, default=OFFICIAL_ROOT)
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=REPO_ROOT / "runs" / "audit_cutpaste_surface_response_a0_20260720",
    )
    parser.add_argument("--batch-size", type=int, default=BATCH_SIZE)
    parser.add_argument("--num-workers", type=int, default=NUM_WORKERS)
    parser.add_argument("--seed", type=int, default=SEED)
    parser.add_argument("--device", choices=("cuda",), default="cuda")
    parser.add_argument("--preflight-only", action="store_true")
    parser.add_argument("--replay-summary", type=Path)
    parser.add_argument("--geometry-replay-summary", type=Path)
    parser.add_argument("--finalize-visual-review", choices=("pass", "fail"))
    parser.add_argument("--expected-summary-sha256")
    return parser.parse_args(argv)


def _validate_locked_args(args: argparse.Namespace) -> None:
    if str(args.device) != "cuda":
        raise ValueError("CutPaste A0 is locked to CUDA")
    if int(args.batch_size) != BATCH_SIZE or int(args.num_workers) != NUM_WORKERS:
        raise ValueError(
            f"CutPaste A0 locks batch-size={BATCH_SIZE}, num-workers={NUM_WORKERS}"
        )
    if int(args.seed) != SEED:
        raise ValueError(f"CutPaste A0 locks seed={SEED}")


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
            args.resolved_config,
            LOCKED_RESOLVED_CONFIG_SHA256,
            "keeper resolved config",
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
            args.protocol, LOCKED_PROTOCOL_SHA256, "CutPaste protocol"
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
        "cutpaste_paper": _verify_hash(
            PAPER_ROOT / "Li_CutPaste_CVPR_2021.pdf",
            LOCKED_PAPER_SHA256,
            "CutPaste paper",
        ),
        "cutpaste_supplemental": _verify_hash(
            PAPER_ROOT / "Li_CutPaste_CVPR_2021_supplemental.pdf",
            LOCKED_SUPPLEMENTAL_SHA256,
            "CutPaste supplemental",
        ),
        "torchvision_license": _verify_hash(
            Path(args.official_root) / "LICENSE",
            LOCKED_TORCHVISION_LICENSE_SHA256,
            "torchvision license",
        ),
        "torchvision_transforms": _verify_hash(
            Path(args.official_root) / "torchvision" / "transforms" / "transforms.py",
            LOCKED_TORCHVISION_TRANSFORMS_SHA256,
            "torchvision transforms source",
        ),
    }
    official_root = Path(args.official_root).expanduser().resolve()
    commit = _git_value(official_root, "rev-parse", "HEAD")
    tree = _git_value(official_root, "rev-parse", "HEAD^{tree}")
    status = _git_value(official_root, "status", "--porcelain")
    remote = _git_value(official_root, "remote", "get-url", "origin")
    if commit != LOCKED_TORCHVISION_COMMIT or tree != LOCKED_TORCHVISION_TREE or status:
        raise ValueError(
            "Official torchvision repository differs from lock: "
            f"commit={commit}, tree={tree}, status={status!r}"
        )
    if torchvision.__version__ != "0.21.0+cu124":
        raise ValueError(
            f"Installed torchvision differs from lock: {torchvision.__version__}"
        )
    protected = _protected_untracked_state()
    if not bool(protected["passed"]):
        raise ValueError(f"Protected user payloads differ from lock: {protected}")
    return {
        "files": files,
        "official_repository": {
            "path": str(official_root),
            "remote": remote,
            "commit": commit,
            "tree": tree,
            "status": status,
            "worktree_clean": status == "",
        },
        "installed_runtime": {
            "torch": torch.__version__,
            "torchvision": torchvision.__version__,
        },
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


def _artifact_record(path: Path, *, root: Path) -> Dict[str, object]:
    resolved = Path(path).resolve()
    return {
        "path": resolved.relative_to(Path(root).resolve()).as_posix(),
        "bytes": int(resolved.stat().st_size),
        "sha256": _sha256(resolved),
    }


def _write_manifest(output_dir: Path) -> Path:
    root = Path(output_dir).resolve()
    files = []
    for path in sorted(root.rglob("*")):
        if path.is_file() and path.name != "artifact_manifest.json":
            files.append(_artifact_record(path, root=root))
    manifest = root / "artifact_manifest.json"
    _write_json(manifest, {"method": METHOD, "payload_count": len(files), "files": files})
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
        if path.is_file() and path.name != "artifact_manifest.json":
            observed[path.relative_to(root).as_posix()] = (
                int(path.stat().st_size),
                _sha256(path),
            )
    if observed != expected:
        raise ValueError("Artifact manifest payload differs from disk")
    return {
        "passed": True,
        "payload_count": len(observed),
        "manifest_sha256": _sha256(manifest_path),
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
            "CutPaste cohort differs: "
            f"rows/TP/FP={(len(cohort), positives, negatives)}"
        )
    observed = {}
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
        raise ValueError(f"CutPaste cohort fold counts differ: {observed}")
    ordered_hash = _cohort_index_sha256([row.sample_index for row in cohort])
    if ordered_hash != EXPECTED_ORDERED_INDEX_SHA256:
        raise ValueError(f"CutPaste cohort order hash differs: {ordered_hash}")
    source_folds: Dict[str, set[int]] = {}
    for row in cohort:
        source_folds.setdefault(row.source_stem, set()).add(int(row.fold))
    overlap = {
        source: sorted(values)
        for source, values in source_folds.items()
        if len(values) > 1
    }
    if overlap:
        raise ValueError(f"CutPaste cohort has cross-fold source overlap: {overlap}")
    return cohort


def _build_dataset(
    checkpoint: Mapping[str, object],
    rows: Sequence[CleanTrainRow],
    data_path: Path,
) -> Tuple[MangoYOLOCropDataset, object, Dict[str, object]]:
    semantics = _eval_semantics(checkpoint)
    if int(semantics["temporal_frames"]) != 1:
        raise ValueError("CutPaste A0 supports temporal_frames=1 only")
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
    return (
        DataLoader(
            dataset,
            batch_size=int(batch_size),
            shuffle=False,
            drop_last=False,
            generator=generator,
            **kwargs,
        ),
        summary,
    )


def _rng_seed(sample_index: int, draw: int, condition_id: int) -> int:
    return int(SEED + 1_000_003 * int(sample_index) + 1_009 * int(draw) + int(condition_id))


def _uniform(generator: torch.Generator, low: float, high: float) -> float:
    return float(low + (high - low) * torch.rand((), generator=generator).item())


def _randint(generator: torch.Generator, low: int, high_inclusive: int) -> int:
    return int(torch.randint(low, high_inclusive + 1, (), generator=generator).item())


def _bbox_support(
    rgb: Tensor,
    crop_bbox: Tensor,
    image_mask: Tensor,
    erosion: int = SUPPORT_EROSION_PIXELS,
) -> Tensor:
    if crop_bbox.numel() != 4:
        raise ValueError("crop_bbox must contain normalized xywh")
    mask = image_mask.detach().cpu().to(dtype=torch.bool).squeeze()
    if mask.ndim != 2:
        raise ValueError("image_mask must resolve to [H,W]")
    height, width = (int(mask.shape[0]), int(mask.shape[1]))
    cx, cy, bw, bh = [float(value) for value in crop_bbox.detach().cpu().tolist()]
    x0 = max(0, int(math.floor((cx - bw / 2.0) * width)))
    y0 = max(0, int(math.floor((cy - bh / 2.0) * height)))
    x1 = min(width, int(math.ceil((cx + bw / 2.0) * width)))
    y1 = min(height, int(math.ceil((cy + bh / 2.0) * height)))
    if x1 <= x0 or y1 <= y0:
        raise ValueError(f"Eroded crop_bbox is empty: {(x0, y0, x1, y1)}")
    array = (
        rgb.detach()
        .cpu()
        .float()
        .clamp(0.0, 1.0)
        .permute(1, 2, 0)
        .numpy()
    )
    surface = torch.from_numpy(
        _surface_detail_foreground_mask_array(
            Image.fromarray((array * 255.0).round().astype(np.uint8)),
            margin=SURFACE_MASK_MARGIN,
        )
    ).to(dtype=torch.bool)
    yy, xx = torch.meshgrid(
        torch.arange(height, dtype=torch.float32),
        torch.arange(width, dtype=torch.float32),
        indexing="ij",
    )
    ellipse_cx = (x0 + x1 - 1) * 0.5
    ellipse_cy = (y0 + y1 - 1) * 0.5
    ellipse_rx = max(1.0, (x1 - x0) * 0.5)
    ellipse_ry = max(1.0, (y1 - y0) * 0.5)
    ellipse = (
        ((xx - ellipse_cx) / ellipse_rx).square()
        + ((yy - ellipse_cy) / ellipse_ry).square()
    ) <= SUPPORT_ELLIPSE_RADIUS_FACTOR**2
    rectangle = torch.zeros_like(mask)
    rectangle[y0:y1, x0:x1] = True
    support = mask & rectangle & ellipse & surface
    invalid = (~support).to(dtype=torch.float32).view(1, 1, height, width)
    dilated_invalid = F.max_pool2d(
        invalid,
        kernel_size=2 * int(erosion) + 1,
        stride=1,
        padding=int(erosion),
    ).view(height, width) > 0.0
    support = support & ~dilated_invalid
    if int(support.sum().item()) < 64:
        raise ValueError("Object-interior support is too small for locked CutPaste")
    return support


def _support_bounds(mask: Tensor) -> Tuple[int, int, int, int]:
    points = torch.nonzero(mask, as_tuple=False)
    if points.numel() == 0:
        raise ValueError("Support mask is empty")
    y0 = int(points[:, 0].min().item())
    y1 = int(points[:, 0].max().item()) + 1
    x0 = int(points[:, 1].min().item())
    x1 = int(points[:, 1].max().item()) + 1
    return x0, y0, x1, y1


def _placement_candidates(
    support: Tensor,
    shape_mask: Tensor,
    *,
    forbidden: Optional[Tensor] = None,
    maximum_iou: float = 0.05,
) -> Tuple[Tensor, Tensor]:
    valid_support = support.detach().cpu().to(dtype=torch.bool).squeeze()
    shape = shape_mask.detach().cpu().to(dtype=torch.bool).squeeze()
    if valid_support.ndim != 2 or shape.ndim != 2 or not bool(shape.any()):
        raise ValueError("Placement support and shape masks must be nonempty 2D masks")
    height, width = int(valid_support.shape[0]), int(valid_support.shape[1])
    patch_height, patch_width = int(shape.shape[0]), int(shape.shape[1])
    if patch_height > height or patch_width > width:
        return torch.empty((0, 2), dtype=torch.long), torch.empty(0)
    shape_pixels = int(shape.sum().item())
    if bool(shape.all()):
        integral = F.pad(
            valid_support.to(dtype=torch.int64), (1, 0, 1, 0), value=0
        ).cumsum(dim=0).cumsum(dim=1)
        covered = (
            integral[patch_height:, patch_width:]
            - integral[:-patch_height, patch_width:]
            - integral[patch_height:, :-patch_width]
            + integral[:-patch_height, :-patch_width]
        )
    else:
        covered = F.conv2d(
            valid_support.to(dtype=torch.float32).view(1, 1, height, width),
            shape.to(dtype=torch.float32).view(1, 1, patch_height, patch_width),
        ).view(height - patch_height + 1, width - patch_width + 1)
    valid = covered == shape_pixels
    iou = torch.zeros_like(covered, dtype=torch.float32)
    if forbidden is not None:
        forbidden_mask = forbidden.detach().cpu().to(dtype=torch.bool).squeeze()
        if bool(shape.all()):
            forbidden_integral = F.pad(
                forbidden_mask.to(dtype=torch.int64), (1, 0, 1, 0), value=0
            ).cumsum(dim=0).cumsum(dim=1)
            intersection = (
                forbidden_integral[patch_height:, patch_width:]
                - forbidden_integral[:-patch_height, patch_width:]
                - forbidden_integral[patch_height:, :-patch_width]
                + forbidden_integral[:-patch_height, :-patch_width]
            ).to(dtype=torch.float32)
        else:
            intersection = F.conv2d(
                forbidden_mask.to(dtype=torch.float32).view(1, 1, height, width),
                shape.to(dtype=torch.float32).view(
                    1, 1, patch_height, patch_width
                ),
            ).view(height - patch_height + 1, width - patch_width + 1)
        forbidden_pixels = int(forbidden_mask.sum().item())
        union = float(forbidden_pixels + shape_pixels) - intersection
        iou = intersection / union.clamp_min(1.0)
        valid &= iou <= float(maximum_iou) + 1e-12
    coordinates = torch.nonzero(valid, as_tuple=False)
    return coordinates, iou


def _place_mask(
    support: Tensor,
    shape_mask: Tensor,
    generator: torch.Generator,
    *,
    forbidden: Optional[Tensor] = None,
    maximum_iou: float = 0.05,
    attempts: int = 10,
) -> Tuple[int, int, Tensor, float, int, int]:
    del attempts
    shape = shape_mask.to(dtype=torch.bool).squeeze()
    patch_height, patch_width = int(shape.shape[0]), int(shape.shape[1])
    coordinates, iou_map = _placement_candidates(
        support,
        shape,
        forbidden=forbidden,
        maximum_iou=maximum_iou,
    )
    if int(coordinates.shape[0]) == 0:
        raise RuntimeError("No fully valid CutPaste placement exists")
    selected = _randint(generator, 0, int(coordinates.shape[0]) - 1)
    top = int(coordinates[selected, 0].item())
    left = int(coordinates[selected, 1].item())
    placed = torch.zeros_like(support, dtype=torch.bool)
    placed[top : top + patch_height, left : left + patch_width] = shape
    iou = float(iou_map[top, left].item()) if forbidden is not None else 0.0
    return left, top, placed, iou, 1, int(coordinates.shape[0])


def _jitter_patch(
    patch: Tensor, generator: torch.Generator
) -> Tuple[Tensor, Dict[str, object]]:
    factors = {
        "brightness": _uniform(generator, 0.9, 1.1),
        "contrast": _uniform(generator, 0.9, 1.1),
        "saturation": _uniform(generator, 0.9, 1.1),
        "hue": _uniform(generator, -0.1, 0.1),
    }
    names = ("brightness", "contrast", "saturation", "hue")
    order = [names[index] for index in torch.randperm(4, generator=generator).tolist()]
    result = patch.clone()
    for name in order:
        if name == "brightness":
            result = TVF.adjust_brightness(result, factors[name])
        elif name == "contrast":
            result = TVF.adjust_contrast(result, factors[name])
        elif name == "saturation":
            result = TVF.adjust_saturation(result, factors[name])
        else:
            result = TVF.adjust_hue(result, factors[name])
    return result.clamp(0.0, 1.0), {"order": order, **factors}


def _mask_bbox(mask: Tensor) -> Tuple[int, int, int, int]:
    return _support_bounds(mask)


def make_cutpaste_pair(
    rgb: Tensor,
    crop_bbox: Tensor,
    image_mask: Tensor,
    *,
    sample_index: int,
    draw: int,
    family: str,
    condition_id: int = 0,
    support: Optional[Tensor] = None,
) -> Tuple[Tensor, Tensor, Dict[str, object]]:
    if family not in FAMILIES:
        raise ValueError(f"Unknown CutPaste family: {family}")
    image = rgb.detach().cpu().to(dtype=torch.float32).clamp(0.0, 1.0)
    if image.ndim != 3 or int(image.shape[0]) != 3:
        raise ValueError("CutPaste expects RGB tensor [3,H,W]")
    if support is None:
        support = _bbox_support(image, crop_bbox, image_mask)
    else:
        support = support.detach().cpu().to(dtype=torch.bool).squeeze()
        if support.ndim != 2 or tuple(support.shape) != tuple(image.shape[1:]):
            raise ValueError("Precomputed support must match image spatial shape")
        if int(support.sum().item()) < 64:
            raise ValueError("Precomputed object-interior support is too small")
    seed = _rng_seed(sample_index, draw, condition_id)
    generator = torch.Generator(device="cpu").manual_seed(seed)
    support_x0, support_y0, support_x1, support_y1 = _support_bounds(support)
    support_width = support_x1 - support_x0
    support_height = support_y1 - support_y0
    support_pixels = int(support.sum().item())

    angle = 0.0
    area_ratio = None
    aspect_ratio = None
    scar_short_side = None
    scar_long_side = None
    shape_attempt = 1
    source_candidate_count = 0
    destination_candidate_count = 0
    if family == "regular":
        for shape_attempt in range(1, 11):
            area_ratio = math.exp(
                _uniform(generator, math.log(0.02), math.log(0.15))
            )
            aspect_ratio = math.exp(
                _uniform(generator, math.log(0.3), math.log(3.3))
            )
            target_area = float(support_pixels) * area_ratio
            patch_width = max(1, int(round(math.sqrt(target_area * aspect_ratio))))
            patch_height = max(1, int(round(math.sqrt(target_area / aspect_ratio))))
            actual_area_ratio = float(
                patch_width * patch_height / support_pixels
            )
            if not (
                patch_width <= support_width
                and patch_height <= support_height
                and 0.02 <= actual_area_ratio <= 0.15
            ):
                continue
            source_shape = torch.ones((patch_height, patch_width), dtype=torch.bool)
            try:
                (
                    source_left,
                    source_top,
                    source_mask,
                    _,
                    source_attempt,
                    source_candidate_count,
                ) = _place_mask(support, source_shape, generator)
                (
                    destination_left,
                    destination_top,
                    destination_mask,
                    source_destination_iou,
                    destination_attempt,
                    destination_candidate_count,
                ) = _place_mask(
                    support,
                    source_shape,
                    generator,
                    forbidden=source_mask,
                    maximum_iou=0.05,
                )
                break
            except RuntimeError:
                continue
        else:
            raise RuntimeError(
                "No valid regular CutPaste geometry was found after ten shape attempts"
            )
        source_patch = image[
            :,
            source_top : source_top + patch_height,
            source_left : source_left + patch_width,
        ].clone()
        identity = image.clone()
        identity[
            :,
            source_top : source_top + patch_height,
            source_left : source_left + patch_width,
        ] = source_patch
        identity_error = float((identity - image).abs().max().item())
        jittered, jitter = _jitter_patch(source_patch, generator)
        paste_patch = jittered
        paste_shape = source_shape
    else:
        for shape_attempt in range(1, 11):
            short_side = _randint(generator, 2, 16)
            long_side = _randint(generator, 10, 25)
            scar_short_side = short_side
            scar_long_side = long_side
            if _randint(generator, 0, 1) == 0:
                patch_width, patch_height = long_side, short_side
            else:
                patch_width, patch_height = short_side, long_side
            source_shape = torch.ones((patch_height, patch_width), dtype=torch.bool)
            try:
                (
                    source_left,
                    source_top,
                    source_mask,
                    _,
                    source_attempt,
                    source_candidate_count,
                ) = _place_mask(support, source_shape, generator)
            except RuntimeError:
                continue
            source_patch = image[
                :,
                source_top : source_top + patch_height,
                source_left : source_left + patch_width,
            ].clone()
            identity = image.clone()
            identity[
                :,
                source_top : source_top + patch_height,
                source_left : source_left + patch_width,
            ] = source_patch
            identity_error = float((identity - image).abs().max().item())
            jittered, jitter = _jitter_patch(source_patch, generator)
            angle = _uniform(generator, -45.0, 45.0)
            paste_patch = TVF.rotate(
                jittered,
                angle,
                interpolation=InterpolationMode.BILINEAR,
                expand=True,
                fill=0.0,
            )
            paste_shape = TVF.rotate(
                source_shape.unsqueeze(0).to(dtype=torch.float32),
                angle,
                interpolation=InterpolationMode.NEAREST,
                expand=True,
                fill=0.0,
            ).squeeze(0) > 0.5
            try:
                (
                    destination_left,
                    destination_top,
                    destination_mask,
                    source_destination_iou,
                    destination_attempt,
                    destination_candidate_count,
                ) = _place_mask(
                    support,
                    paste_shape,
                    generator,
                    forbidden=source_mask,
                    maximum_iou=0.05,
                )
                break
            except RuntimeError:
                continue
        else:
            raise RuntimeError(
                "No valid CutPaste-Scar geometry was found after ten shape attempts"
            )

    actual_area_ratio = float(
        int(paste_shape.sum().item()) / support_pixels
    )

    paste_height, paste_width = int(paste_shape.shape[0]), int(paste_shape.shape[1])
    candidate = image.clone()
    destination = candidate[
        :,
        destination_top : destination_top + paste_height,
        destination_left : destination_left + paste_width,
    ]
    destination.copy_(torch.where(paste_shape.unsqueeze(0), paste_patch, destination))
    object_mean = image[:, support].mean(dim=1)
    cutout = image.clone()
    cutout_destination = cutout[
        :,
        destination_top : destination_top + paste_height,
        destination_left : destination_left + paste_width,
    ]
    cutout_destination.copy_(
        torch.where(
            paste_shape.unsqueeze(0),
            object_mean.view(3, 1, 1).expand_as(cutout_destination),
            cutout_destination,
        )
    )
    candidate_changed = (candidate != image).any(dim=0)
    cutout_changed = (cutout != image).any(dim=0)
    destination_bbox = _mask_bbox(destination_mask)
    record: Dict[str, object] = {
        "sample_index": int(sample_index),
        "family": family,
        "draw": int(draw),
        "condition_id": int(condition_id),
        "seed": int(seed),
        "support_x0": support_x0,
        "support_y0": support_y0,
        "support_x1": support_x1,
        "support_y1": support_y1,
        "support_pixels": support_pixels,
        "support_surface_margin": SURFACE_MASK_MARGIN,
        "support_ellipse_radius_factor": SUPPORT_ELLIPSE_RADIUS_FACTOR,
        "support_erosion_pixels": SUPPORT_EROSION_PIXELS,
        "support_algorithm": "surface_mask_cropbbox_imagemask_ellipse_erode",
        "source_left": int(source_left),
        "source_top": int(source_top),
        "source_width": int(patch_width),
        "source_height": int(patch_height),
        "destination_left": int(destination_left),
        "destination_top": int(destination_top),
        "destination_width": int(paste_width),
        "destination_height": int(paste_height),
        "destination_mask_x0": int(destination_bbox[0]),
        "destination_mask_y0": int(destination_bbox[1]),
        "destination_mask_x1": int(destination_bbox[2]),
        "destination_mask_y1": int(destination_bbox[3]),
        "source_attempt": int(source_attempt),
        "destination_attempt": int(destination_attempt),
        "shape_attempt": int(shape_attempt),
        "source_candidate_count": int(source_candidate_count),
        "destination_candidate_count": int(destination_candidate_count),
        "source_destination_iou": float(source_destination_iou),
        "identity_max_abs_error": identity_error,
        "angle_degrees": float(angle),
        "sampled_area_ratio": area_ratio,
        "actual_destination_area_ratio": actual_area_ratio,
        "sampled_aspect_ratio": aspect_ratio,
        "scar_short_side": scar_short_side,
        "scar_long_side": scar_long_side,
        "jitter_order": ">".join(str(value) for value in jitter["order"]),
        "jitter_brightness": float(jitter["brightness"]),
        "jitter_contrast": float(jitter["contrast"]),
        "jitter_saturation": float(jitter["saturation"]),
        "jitter_hue": float(jitter["hue"]),
        "support_mask_sha256": _array_sha256(support.numpy()),
        "source_mask_sha256": _array_sha256(source_mask.numpy()),
        "paired_destination_mask_sha256": _array_sha256(destination_mask.numpy()),
        "candidate_destination_mask_sha256": _array_sha256(destination_mask.numpy()),
        "cutout_destination_mask_sha256": _array_sha256(destination_mask.numpy()),
        "candidate_changed_mask_sha256": _array_sha256(candidate_changed.numpy()),
        "cutout_changed_mask_sha256": _array_sha256(cutout_changed.numpy()),
        "source_fully_valid": bool(support[source_mask].all()),
        "destination_fully_valid": bool(support[destination_mask].all()),
        "candidate_changed_subset_destination": bool(
            (~candidate_changed | destination_mask).all()
        ),
        "cutout_changed_subset_destination": bool(
            (~cutout_changed | destination_mask).all()
        ),
    }
    return candidate, cutout, record


def _global_rng_snapshot() -> Tuple[object, tuple, Tensor]:
    numpy_state = np.random.get_state()
    copied_numpy = (
        numpy_state[0],
        numpy_state[1].copy(),
        numpy_state[2],
        numpy_state[3],
        numpy_state[4],
    )
    return random.getstate(), copied_numpy, torch.get_rng_state().clone()


def _global_rng_equal(
    left: Tuple[object, tuple, Tensor], right: Tuple[object, tuple, Tensor]
) -> bool:
    numpy_equal = (
        left[1][0] == right[1][0]
        and np.array_equal(left[1][1], right[1][1])
        and left[1][2:] == right[1][2:]
    )
    return bool(left[0] == right[0] and numpy_equal and torch.equal(left[2], right[2]))


def _records_sha256(records: Sequence[Mapping[str, object]]) -> str:
    digest = hashlib.sha256()
    for row in records:
        digest.update(
            (json.dumps(to_serializable(row), sort_keys=True, separators=(",", ":")) + "\n").encode(
                "utf-8"
            )
        )
    return digest.hexdigest()


def cutpaste_engineering_checks() -> Dict[str, object]:
    coordinate = torch.linspace(0.0, 1.0, 256, dtype=torch.float32)
    yy, xx = torch.meshgrid(coordinate, coordinate, indexing="ij")
    rgb = torch.stack((xx, yy, (xx + yy) * 0.5), dim=0)
    crop_bbox = torch.tensor([0.5, 0.5, 0.72, 0.64], dtype=torch.float32)
    image_mask = torch.ones((256, 256), dtype=torch.bool)
    image_mask[:18] = False
    before = _global_rng_snapshot()
    first = []
    outputs = []
    for family in FAMILIES:
        candidate, cutout, record = make_cutpaste_pair(
            rgb,
            crop_bbox,
            image_mask,
            sample_index=137,
            draw=2,
            family=family,
            condition_id=0,
        )
        first.append(record)
        outputs.append((candidate, cutout))
    after = _global_rng_snapshot()
    second = [
        make_cutpaste_pair(
            rgb,
            crop_bbox,
            image_mask,
            sample_index=137,
            draw=2,
            family=family,
            condition_id=0,
        )[2]
        for family in FAMILIES
    ]
    changed = [
        float((candidate - rgb).abs().max().item()) > 0.0
        and float((cutout - rgb).abs().max().item()) > 0.0
        for candidate, cutout in outputs
    ]
    checks = {
        "deterministic_geometry_exact": first == second,
        "local_rng_does_not_consume_global_state": _global_rng_equal(before, after),
        "identity_repaste_le_1e_7": max(
            float(row["identity_max_abs_error"]) for row in first
        )
        <= 1e-7,
        "source_destination_iou_le_005": max(
            float(row["source_destination_iou"]) for row in first
        )
        <= 0.05,
        "source_and_destination_fully_valid": all(
            bool(row["source_fully_valid"]) and bool(row["destination_fully_valid"])
            for row in first
        ),
        "changed_support_inside_destination": all(
            bool(row["candidate_changed_subset_destination"])
            and bool(row["cutout_changed_subset_destination"])
            for row in first
        ),
        "candidate_and_cutout_nondegenerate": all(changed),
        "family_seed_shared_for_paired_geometry": all(
            int(row["seed"]) == _rng_seed(137, 2, 0) for row in first
        ),
    }
    return {
        "checks": checks,
        "passed": all(checks.values()),
        "records": first,
        "records_sha256": _records_sha256(first),
    }


def _normalization_tensors(
    semantics: Mapping[str, object], *, dtype: torch.dtype = torch.float32
) -> Tuple[Tensor, Tensor]:
    mean = torch.tensor(semantics["input_mean"], dtype=dtype).view(3, 1, 1)
    std = torch.tensor(semantics["input_std"], dtype=dtype).view(3, 1, 1)
    return mean, std


def _to_rgb(images: Tensor, semantics: Mapping[str, object]) -> Tensor:
    mean, std = _normalization_tensors(semantics, dtype=torch.float32)
    return (images.detach().cpu().float() * std + mean).clamp(0.0, 1.0)


def _to_normalized(images: Tensor, semantics: Mapping[str, object]) -> Tensor:
    mean, std = _normalization_tensors(semantics, dtype=torch.float32)
    return (images.detach().cpu().float() - mean) / std


def _metadata_tensor(
    metadata: Mapping[str, object], key: str, *, device: torch.device, dtype: torch.dtype
) -> Optional[Tensor]:
    value = metadata.get(key)
    if not torch.is_tensor(value):
        return None
    return value.to(device=device, dtype=dtype, non_blocking=True)


def _forward_outputs(
    model: nn.Module,
    images: Tensor,
    *,
    image_mask: Optional[Tensor],
    bbox: Optional[Tensor],
) -> Tuple[Tensor, Tensor, Tensor]:
    features = model.forward_features(  # type: ignore[attr-defined]
        images,
        image_valid_mask=image_mask,
        bbox_token_prior=bbox,
    )
    if bbox is not None:
        features["bbox"] = bbox
    logits = classification_logits_from_features(model, features).float()
    probabilities = logits.softmax(dim=1)
    pooled = features["pooled"].float()
    if logits.ndim != 2 or int(logits.shape[1]) != 5 or pooled.ndim != 2:
        raise ValueError("Keeper forward output shape differs from CutPaste lock")
    if not bool(
        torch.isfinite(logits).all()
        and torch.isfinite(probabilities).all()
        and torch.isfinite(pooled).all()
    ):
        raise ValueError("Keeper forward emitted non-finite CutPaste payload")
    return logits, probabilities, pooled


def response_scalars(
    clean_logits: Tensor,
    clean_probabilities: Tensor,
    clean_pooled: Tensor,
    transformed_logits: Tensor,
    transformed_probabilities: Tensor,
    transformed_pooled: Tensor,
) -> Tensor:
    if clean_logits.shape != transformed_logits.shape:
        raise ValueError("Clean/transformed logit shapes differ")
    rivals = torch.tensor(
        RESTRICTED_NEGATIVE_CLASSES,
        device=clean_logits.device,
        dtype=torch.long,
    )
    clean_margin = clean_logits[:, FOCUS_CLASS] - clean_logits.index_select(
        1, rivals
    ).amax(dim=1)
    transformed_margin = transformed_logits[:, FOCUS_CLASS] - transformed_logits.index_select(
        1, rivals
    ).amax(dim=1)
    clean_p = clean_probabilities.float().clamp_min(LOG_PROBABILITY_CLIP)
    transformed_p = transformed_probabilities.float().clamp_min(
        LOG_PROBABILITY_CLIP
    )
    mixture = (0.5 * (clean_p + transformed_p)).clamp_min(
        LOG_PROBABILITY_CLIP
    )
    js = 0.5 * (
        (clean_p * (clean_p.log() - mixture.log())).sum(dim=1)
        + (
            transformed_p
            * (transformed_p.log() - mixture.log())
        ).sum(dim=1)
    )
    clean_embedding = F.normalize(clean_pooled.float(), dim=1, eps=1e-12)
    transformed_embedding = F.normalize(
        transformed_pooled.float(), dim=1, eps=1e-12
    )
    cosine = (1.0 - (clean_embedding * transformed_embedding).sum(dim=1)).clamp_min(
        0.0
    )
    l2 = torch.linalg.vector_norm(
        clean_embedding - transformed_embedding, ord=2, dim=1
    )
    result = torch.stack(
        (
            transformed_logits[:, FOCUS_CLASS] - clean_logits[:, FOCUS_CLASS],
            transformed_margin - clean_margin,
            js,
            cosine,
            l2,
        ),
        dim=1,
    )
    if tuple(result.shape) != (int(clean_logits.shape[0]), len(RESPONSE_NAMES)):
        raise ValueError("CutPaste response shape differs")
    if not bool(torch.isfinite(result).all()):
        raise ValueError("CutPaste response contains non-finite values")
    return result


def aggregate_responses(raw: np.ndarray) -> np.ndarray:
    value = np.asarray(raw, dtype=np.float64)
    expected_tail = (len(FAMILIES), DRAW_COUNT, len(RESPONSE_NAMES))
    if value.ndim != 4 or tuple(value.shape[1:]) != expected_tail:
        raise ValueError(
            f"Raw CutPaste response must have shape [N,{expected_tail}], got {value.shape}"
        )
    columns = []
    for family_index in range(len(FAMILIES)):
        for response_index in range(len(RESPONSE_NAMES)):
            draws = value[:, family_index, :, response_index]
            columns.extend(
                (
                    draws.mean(axis=1),
                    draws.std(axis=1, ddof=0),
                    draws.min(axis=1),
                    draws.max(axis=1),
                )
            )
    result = np.stack(columns, axis=1)
    if result.shape != (value.shape[0], RESPONSE_DIM):
        raise ValueError(f"Aggregated response shape differs: {result.shape}")
    if not np.isfinite(result).all():
        raise ValueError("Aggregated response contains non-finite values")
    return result


def build_source_derangement(
    responses: np.ndarray,
    *,
    folds: np.ndarray,
    clean_margins: np.ndarray,
    source_stems: Sequence[str],
    sample_indices: np.ndarray,
) -> Tuple[np.ndarray, List[Dict[str, object]]]:
    values = np.asarray(responses, dtype=np.float64)
    fold_values = np.asarray(folds, dtype=np.int64).reshape(-1)
    margins = np.asarray(clean_margins, dtype=np.float64).reshape(-1)
    indices = np.asarray(sample_indices, dtype=np.int64).reshape(-1)
    sources = np.asarray([str(value) for value in source_stems], dtype=str)
    rows = int(values.shape[0])
    if not (
        fold_values.size == margins.size == indices.size == sources.size == rows
    ):
        raise ValueError("Derangement inputs do not align")
    assigned = np.full(rows, -1, dtype=np.int64)
    quartiles = np.full(rows, -1, dtype=np.int64)
    mapping: List[Dict[str, object]] = []
    for fold in FOLDS:
        fold_positions = np.flatnonzero(fold_values == int(fold))
        ranked = fold_positions[
            np.lexsort((indices[fold_positions], margins[fold_positions]))
        ]
        fold_quartiles = np.minimum(
            3, (4 * np.arange(ranked.size, dtype=np.int64)) // ranked.size
        )
        quartiles[ranked] = fold_quartiles
        for quartile in range(4):
            partition = ranked[fold_quartiles == quartile]
            partition = partition[np.argsort(indices[partition], kind="stable")]
            if partition.size < 2:
                raise ValueError("Derangement fold/quartile partition is too small")
            chosen_shift = None
            for shift in range(1, int(partition.size)):
                candidates = np.roll(partition, -shift)
                if np.all(sources[partition] != sources[candidates]):
                    chosen_shift = int(shift)
                    assigned[partition] = candidates
                    break
            if chosen_shift is None:
                raise RuntimeError(
                    f"No source-safe cyclic derangement for fold={fold}, quartile={quartile}"
                )
            for destination, source in zip(partition.tolist(), assigned[partition].tolist()):
                mapping.append(
                    {
                        "destination_sample_index": int(indices[destination]),
                        "source_sample_index": int(indices[source]),
                        "fold": int(fold),
                        "quartile": int(quartile),
                        "cyclic_shift": chosen_shift,
                        "destination_source_stem": str(sources[destination]),
                        "response_source_stem": str(sources[source]),
                    }
                )
    if bool((assigned < 0).any()) or bool((quartiles < 0).any()):
        raise RuntimeError("Source derangement is incomplete")
    if not np.array_equal(fold_values, fold_values[assigned]):
        raise RuntimeError("Source derangement crossed a held fold")
    if np.any(sources == sources[assigned]):
        raise RuntimeError("Source derangement retained an identical source")
    return values[assigned], sorted(mapping, key=lambda row: row["destination_sample_index"])


def build_readout_features(
    *,
    clean_logits: np.ndarray,
    clean_probabilities: np.ndarray,
    cutpaste_responses: np.ndarray,
    cutout_responses: np.ndarray,
    folds: np.ndarray,
    source_stems: Sequence[str],
    sample_indices: np.ndarray,
) -> Tuple[Dict[str, np.ndarray], List[Dict[str, object]], Dict[str, object]]:
    logits = np.asarray(clean_logits, dtype=np.float64)
    probabilities = np.asarray(clean_probabilities, dtype=np.float64)
    if logits.shape != probabilities.shape or logits.ndim != 2 or logits.shape[1] != 5:
        raise ValueError("Clean logits/probabilities must both be [N,5]")
    clipped_logs = np.log(np.clip(probabilities, LOG_PROBABILITY_CLIP, 1.0))
    clean_margin = logits[:, FOCUS_CLASS] - logits[:, RESTRICTED_NEGATIVE_CLASSES].max(
        axis=1
    )
    base = np.concatenate((clipped_logs, clean_margin[:, None]), axis=1)
    candidate = aggregate_responses(cutpaste_responses)
    control = aggregate_responses(cutout_responses)
    contrast = aggregate_responses(
        np.asarray(cutpaste_responses, dtype=np.float64)
        - np.asarray(cutout_responses, dtype=np.float64)
    )
    deranged, mapping = build_source_derangement(
        candidate,
        folds=folds,
        clean_margins=clean_margin,
        source_stems=source_stems,
        sample_indices=sample_indices,
    )
    features = {
        "base_only": base,
        "cutout_control": np.concatenate((base, control), axis=1),
        "cutpaste_candidate": np.concatenate((base, candidate), axis=1),
        "paired_contrast": np.concatenate((base, contrast), axis=1),
        "source_deranged": np.concatenate((base, deranged), axis=1),
    }
    expected = {
        "base_only": BASE_DIM,
        "cutout_control": BASE_DIM + RESPONSE_DIM,
        "cutpaste_candidate": BASE_DIM + RESPONSE_DIM,
        "paired_contrast": BASE_DIM + RESPONSE_DIM,
        "source_deranged": BASE_DIM + RESPONSE_DIM,
    }
    for role, values in features.items():
        if values.shape != (probabilities.shape[0], expected[role]):
            raise ValueError(f"Readout feature shape differs for {role}: {values.shape}")
        if not np.isfinite(values).all():
            raise ValueError(f"Readout feature contains non-finite values: {role}")
    descriptor_hashes = {
        "base": _array_sha256(base.astype(np.float32)),
        "cutpaste": _array_sha256(candidate.astype(np.float32)),
        "cutout": _array_sha256(control.astype(np.float32)),
        "paired_contrast": _array_sha256(contrast.astype(np.float32)),
        "source_deranged": _array_sha256(deranged.astype(np.float32)),
    }
    return features, mapping, descriptor_hashes


def _positive_threshold(scores: np.ndarray, labels: np.ndarray) -> float:
    values = np.asarray(scores, dtype=np.float64).reshape(-1)
    binary = np.asarray(labels, dtype=np.int64).reshape(-1)
    positives = np.sort(values[binary == 1])
    if positives.size == 0:
        raise ValueError("Readout fit fold has no positive rows")
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
    *,
    seed: int = SEED,
) -> Tuple[Dict[str, np.ndarray], Dict[str, np.ndarray], Dict[str, object]]:
    if set(features) != set(ROLE_NAMES):
        raise ValueError("Readout feature roles differ from locked CutPaste bank")
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
            overlap = fit_sources.intersection(held_sources)
            if overlap:
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
                    random_state=int(seed),
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


def descriptor_statistics(descriptors: np.ndarray) -> Dict[str, object]:
    values = np.asarray(descriptors, dtype=np.float64)
    if values.ndim != 2 or int(values.shape[1]) != RESPONSE_DIM:
        raise ValueError(f"Response descriptor must be [N,{RESPONSE_DIM}]")
    centered = values - values.mean(axis=0, keepdims=True)
    singular = np.linalg.svd(centered, full_matrices=False, compute_uv=False)
    energy = np.square(singular)
    distribution = energy / max(float(energy.sum()), 1e-30)
    nonzero = distribution > 0.0
    effective_rank = float(
        np.exp(-np.sum(distribution[nonzero] * np.log(distribution[nonzero])))
    )
    standard_deviation = values.std(axis=0, ddof=0)
    return {
        "shape": [int(value) for value in values.shape],
        "sha256": _array_sha256(values.astype(np.float32)),
        "effective_rank": effective_rank,
        "active_dimensions": int((standard_deviation > 1e-8).sum()),
        "minimum_std": float(standard_deviation.min()),
        "maximum_std": float(standard_deviation.max()),
        "all_values_finite": bool(np.isfinite(values).all()),
    }


def _rival_predictions(probabilities: np.ndarray) -> np.ndarray:
    values = np.asarray(probabilities, dtype=np.float64)
    masked = values.copy()
    masked[:, FOCUS_CLASS] = -np.inf
    return masked.argmax(axis=1).astype(np.int64)


def _role_metrics(
    *,
    labels: np.ndarray,
    targets: np.ndarray,
    folds: np.ndarray,
    scores: np.ndarray,
    actions: np.ndarray,
    clean_probabilities: np.ndarray,
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
        raise ValueError("CutPaste role metric inputs do not align")
    positive = binary == 1
    negative = binary == 0
    rivals = _rival_predictions(clean_probabilities)
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
    semantics_exact = bool(
        np.all(action_predictions[accepted] == FOCUS_CLASS)
        and np.array_equal(action_predictions[~accepted], rivals[~accepted])
    )
    return {
        "auroc": float(roc_auc_score(binary, score_values)),
        "tp_retention": float(accepted[positive].mean()),
        "fp_rejection": float((~accepted[negative]).mean()),
        "tp_breaks": int((~accepted[positive]).sum()),
        "fp_rejects": int((~accepted[negative]).sum()),
        "corrections": int(corrections.sum()),
        "harms": int(harms.sum()),
        "neutral_rejections": int(((~accepted) & ~corrections & ~harms).sum()),
        "action_semantics_exact": semantics_exact,
        "action_prediction_sha256": _array_sha256(action_predictions),
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
    clean_probabilities: np.ndarray,
    candidate_descriptor_stats: Mapping[str, object],
) -> Dict[str, object]:
    if set(scores) != set(ROLE_NAMES) or set(actions) != set(ROLE_NAMES):
        raise ValueError("Analysis role bank differs from locked CutPaste roles")
    metrics = {
        role: _role_metrics(
            labels=labels,
            targets=targets,
            folds=folds,
            scores=scores[role],
            actions=actions[role],
            clean_probabilities=clean_probabilities,
        )
        for role in ROLE_NAMES
    }
    candidate = metrics["cutpaste_candidate"]
    candidate_auc = float(candidate["auroc"])
    gains = {
        role: candidate_auc - float(metrics[role]["auroc"])
        for role in ("base_only", "cutout_control", "source_deranged")
    }
    paired_gain = float(metrics["paired_contrast"]["auroc"]) - float(
        metrics["cutout_control"]["auroc"]
    )
    fold_wins = 0
    candidate_folds = {int(row["fold"]): row for row in candidate["per_fold"]}
    for fold in FOLDS:
        candidate_fold_auc = float(candidate_folds[fold]["auroc"])
        base_fold_auc = float(metrics["base_only"]["per_fold"][fold]["auroc"])
        cutout_fold_auc = float(
            metrics["cutout_control"]["per_fold"][fold]["auroc"]
        )
        fold_wins += int(
            candidate_fold_auc > base_fold_auc
            and candidate_fold_auc > cutout_fold_auc
        )
    base_additional_rejects = int(candidate["fp_rejects"]) - int(
        metrics["base_only"]["fp_rejects"]
    )
    cutout_additional_rejects = int(candidate["fp_rejects"]) - int(
        metrics["cutout_control"]["fp_rejects"]
    )
    mechanism_gates = {
        "candidate_auroc_ge_084": candidate_auc >= 0.84,
        "candidate_gain_vs_base_ge_002": gains["base_only"] >= 0.02,
        "candidate_gain_vs_cutout_ge_0015": gains["cutout_control"] >= 0.015,
        "candidate_gain_vs_source_deranged_ge_002": gains["source_deranged"] >= 0.02,
        "paired_contrast_auroc_ge_065": float(metrics["paired_contrast"]["auroc"])
        >= 0.65,
        "paired_contrast_gain_vs_cutout_ge_0005": paired_gain >= 0.005,
        "candidate_tp_retention_ge_095": float(candidate["tp_retention"]) >= 0.95,
        "candidate_fp_rejection_ge_025": float(candidate["fp_rejection"]) >= 0.25,
        "candidate_fp_rejects_10_above_base": base_additional_rejects >= 10,
        "candidate_fp_rejects_10_above_cutout": cutout_additional_rejects >= 10,
        "candidate_corrections_ge_2x_harms": int(candidate["corrections"])
        >= 2 * int(candidate["harms"]),
        "candidate_wins_both_controls_in_4_of_5_folds": fold_wins >= 4,
        "minimum_fold_tp_retention_ge_090": min(
            float(row["tp_retention"]) for row in candidate["per_fold"]
        )
        >= 0.90,
        "action_semantics_exact": all(
            bool(metrics[role]["action_semantics_exact"]) for role in ROLE_NAMES
        ),
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
        "candidate_response_effective_rank_ge_8": float(
            candidate_descriptor_stats["effective_rank"]
        )
        >= 8.0,
    }
    return {
        "metrics": metrics,
        "candidate_auroc_gains": gains,
        "paired_contrast_gain_vs_cutout": paired_gain,
        "candidate_fold_wins_over_base_and_cutout": int(fold_wins),
        "candidate_additional_fp_rejects": {
            "versus_base": base_additional_rejects,
            "versus_cutout": cutout_additional_rejects,
        },
        "mechanism_gates": mechanism_gates,
        "mechanism_gates_passed": all(mechanism_gates.values()),
    }


def condition_gate(analysis: Mapping[str, object]) -> Dict[str, object]:
    metrics = analysis["metrics"]
    candidate = metrics["cutpaste_candidate"]
    candidate_auc = float(candidate["auroc"])
    gains = {
        role: candidate_auc - float(metrics[role]["auroc"])
        for role in ("base_only", "cutout_control")
    }
    gates = {
        "candidate_auroc_ge_078": candidate_auc >= 0.78,
        "gain_vs_base_ge_001": gains["base_only"] >= 0.01,
        "gain_vs_cutout_ge_001": gains["cutout_control"] >= 0.01,
        "tp_retention_ge_092": float(candidate["tp_retention"]) >= 0.92,
        "fp_rejection_ge_015": float(candidate["fp_rejection"]) >= 0.15,
        "corrections_not_below_harms": int(candidate["corrections"])
        >= int(candidate["harms"]),
        "target0_rejection_ge_008": float(
            candidate["per_target"]["0"]["rejection_rate"]
        )
        >= 0.08,
        "target2_rejection_ge_008": float(
            candidate["per_target"]["2"]["rejection_rate"]
        )
        >= 0.08,
    }
    return {"gains": gains, "gates": gates, "passed": all(gates.values())}


def _review_indices(cohort: Sequence[CleanTrainRow]) -> List[int]:
    categories = (
        lambda row: row.target == FOCUS_CLASS,
        lambda row: row.target == 0,
        lambda row: row.target == 2,
        lambda row: row.target == 4,
    )
    selected: List[int] = []
    for predicate in categories:
        values = sorted(row.sample_index for row in cohort if predicate(row))[:3]
        if len(values) != 3:
            raise ValueError("Fixed CutPaste review cohort lacks three category rows")
        selected.extend(values)
    return selected


def _geometry_summary(records: Sequence[Mapping[str, object]]) -> Dict[str, object]:
    rows = list(records)
    expected_rows = EXPECTED_COHORT_ROWS * len(FAMILIES) * DRAW_COUNT
    unique_keys = {
        (int(row["sample_index"]), str(row["family"]), int(row["draw"]))
        for row in rows
    }
    regular = [row for row in rows if row["family"] == "regular"]
    scar = [row for row in rows if row["family"] == "scar"]
    paired_exact = all(
        row["candidate_destination_mask_sha256"]
        == row["cutout_destination_mask_sha256"]
        == row["paired_destination_mask_sha256"]
        for row in rows
    )
    sample_support_pairs = {
        (int(row["sample_index"]), str(row["support_mask_sha256"])) for row in rows
    }
    sample_indices = {int(row["sample_index"]) for row in rows}
    checks = {
        "record_count_exact": len(rows) == expected_rows,
        "unique_sample_family_draw_keys": len(unique_keys) == expected_rows,
        "paired_candidate_cutout_masks_exact": paired_exact,
        "support_policy_exact": all(
            str(row["support_algorithm"])
            == "surface_mask_cropbbox_imagemask_ellipse_erode"
            and float(row["support_surface_margin"]) == SURFACE_MASK_MARGIN
            and float(row["support_ellipse_radius_factor"])
            == SUPPORT_ELLIPSE_RADIUS_FACTOR
            and int(row["support_erosion_pixels"]) == SUPPORT_EROSION_PIXELS
            for row in rows
        ),
        "support_hash_stable_per_sample": len(sample_indices)
        == EXPECTED_COHORT_ROWS
        and len(sample_support_pairs) == EXPECTED_COHORT_ROWS,
        "enumerated_candidate_counts_positive": all(
            int(row["source_candidate_count"]) > 0
            and int(row["destination_candidate_count"]) > 0
            for row in rows
        ),
        "all_source_fully_valid": all(bool(row["source_fully_valid"]) for row in rows),
        "all_destination_fully_valid": all(
            bool(row["destination_fully_valid"]) for row in rows
        ),
        "all_changed_support_inside_destination": all(
            bool(row["candidate_changed_subset_destination"])
            and bool(row["cutout_changed_subset_destination"])
            for row in rows
        ),
        "all_iou_le_005": max(
            float(row["source_destination_iou"]) for row in rows
        )
        <= 0.05,
        "identity_repaste_le_1e_7": max(
            float(row["identity_max_abs_error"]) for row in rows
        )
        <= 1e-7,
        "attempts_le_10": max(
            max(
                int(row["shape_attempt"]),
                int(row["source_attempt"]),
                int(row["destination_attempt"]),
            )
            for row in rows
        )
        <= 10,
        "regular_area_in_locked_range": all(
            0.02 <= float(row["sampled_area_ratio"]) <= 0.15 for row in regular
        ),
        "regular_aspect_in_locked_range": all(
            0.3 <= float(row["sampled_aspect_ratio"]) <= 3.3 for row in regular
        ),
        "regular_actual_area_in_locked_range": all(
            0.02 <= float(row["actual_destination_area_ratio"]) <= 0.15
            for row in regular
        ),
        "scar_sides_in_locked_range": all(
            2 <= int(row["scar_short_side"]) <= 16
            and 10 <= int(row["scar_long_side"]) <= 25
            for row in scar
        ),
        "scar_angle_in_locked_range": all(
            -45.0 <= float(row["angle_degrees"]) <= 45.0 for row in scar
        ),
        "jitter_in_locked_range": all(
            0.9 <= float(row["jitter_brightness"]) <= 1.1
            and 0.9 <= float(row["jitter_contrast"]) <= 1.1
            and 0.9 <= float(row["jitter_saturation"]) <= 1.1
            and -0.1 <= float(row["jitter_hue"]) <= 0.1
            for row in rows
        ),
    }
    return {
        "rows": len(rows),
        "records_sha256": _records_sha256(rows),
        "checks": checks,
        "passed": all(checks.values()),
        "maximum_source_destination_iou": max(
            float(row["source_destination_iou"]) for row in rows
        ),
        "maximum_identity_error": max(
            float(row["identity_max_abs_error"]) for row in rows
        ),
    }


def extract_condition(
    *,
    model: nn.Module,
    loader: DataLoader,
    loader_summary: Mapping[str, object],
    cohort: Sequence[CleanTrainRow],
    device: torch.device,
    semantics: Mapping[str, object],
    condition: str,
    condition_id: int,
    review_indices: Sequence[int],
) -> Dict[str, object]:
    rows = len(cohort)
    cutpaste_responses = np.empty(
        (rows, len(FAMILIES), DRAW_COUNT, len(RESPONSE_NAMES)), dtype=np.float32
    )
    cutout_responses = np.empty_like(cutpaste_responses)
    clean_logits_bank = np.empty((rows, 5), dtype=np.float32)
    clean_probability_bank = np.empty((rows, 5), dtype=np.float32)
    clean_pooled_bank: Optional[np.ndarray] = None
    targets_bank = np.empty(rows, dtype=np.int64)
    indices_bank = np.empty(rows, dtype=np.int64)
    geometry: List[Dict[str, object]] = []
    visual_cache: Dict[int, Dict[str, object]] = {}
    review_set = {int(value) for value in review_indices}
    model.eval()
    state_before = _model_state_sha256(model)
    if device.type == "cuda":
        torch.cuda.empty_cache()
        torch.cuda.reset_peak_memory_stats(device)
        torch.cuda.synchronize(device)
    started = time.perf_counter()
    position = 0
    rng_local_only = True
    with torch.inference_mode():
        for images, targets, metadata in loader:
            batch = int(targets.numel())
            stop = position + batch
            sample_indices = metadata["sample_index"].detach().cpu().long()
            expected_indices = [
                int(row.sample_index) for row in cohort[position:stop]
            ]
            if sample_indices.tolist() != expected_indices:
                raise ValueError("CutPaste loader changed cohort order")
            if "crop_bbox" not in metadata or "image_mask" not in metadata:
                raise ValueError("CutPaste requires crop_bbox and image_mask metadata")
            crop_bboxes = metadata["crop_bbox"].detach().cpu().float()
            image_masks_cpu = metadata["image_mask"].detach().cpu().bool()
            rgb = _to_rgb(images, semantics)
            support_masks = [
                _bbox_support(
                    rgb[local_index],
                    crop_bboxes[local_index],
                    image_masks_cpu[local_index],
                )
                for local_index in range(batch)
            ]

            images_device = images.to(
                device=device, dtype=torch.float32, non_blocking=True
            )
            image_mask = _metadata_tensor(
                metadata, "image_mask", device=device, dtype=torch.bool
            )
            if image_mask is not None and image_mask.ndim == 3:
                image_mask = image_mask.unsqueeze(1)
            bbox = _metadata_tensor(
                metadata, "bbox", device=device, dtype=torch.float32
            )
            clean_logits, clean_probabilities, clean_pooled = _forward_outputs(
                model,
                images_device,
                image_mask=image_mask,
                bbox=bbox,
            )
            if clean_pooled_bank is None:
                clean_pooled_bank = np.empty(
                    (rows, int(clean_pooled.shape[1])), dtype=np.float32
                )
            clean_logits_bank[position:stop] = clean_logits.detach().cpu().numpy()
            clean_probability_bank[position:stop] = (
                clean_probabilities.detach().cpu().numpy()
            )
            clean_pooled_bank[position:stop] = clean_pooled.detach().cpu().numpy()
            targets_bank[position:stop] = targets.detach().cpu().numpy()
            indices_bank[position:stop] = sample_indices.numpy()

            for family_index, family in enumerate(FAMILIES):
                for draw in range(DRAW_COUNT):
                    before_rng = _global_rng_snapshot()
                    candidate_batch = []
                    cutout_batch = []
                    batch_records = []
                    for local_index, sample_index in enumerate(sample_indices.tolist()):
                        candidate, cutout, record = make_cutpaste_pair(
                            rgb[local_index],
                            crop_bboxes[local_index],
                            image_masks_cpu[local_index],
                            sample_index=int(sample_index),
                            draw=int(draw),
                            family=family,
                            condition_id=int(condition_id),
                            support=support_masks[local_index],
                        )
                        candidate_batch.append(candidate)
                        cutout_batch.append(cutout)
                        batch_records.append(record)
                        if draw == 0 and int(sample_index) in review_set:
                            cache = visual_cache.setdefault(
                                int(sample_index), {"clean": images[local_index].detach().cpu()}
                            )
                            cache[f"{family}_candidate"] = _to_normalized(
                                candidate, semantics
                            )
                            cache[f"{family}_cutout"] = _to_normalized(
                                cutout, semantics
                            )
                            cache[f"{family}_geometry"] = record
                    after_rng = _global_rng_snapshot()
                    rng_local_only = rng_local_only and _global_rng_equal(
                        before_rng, after_rng
                    )
                    geometry.extend(batch_records)
                    candidate_normalized = _to_normalized(
                        torch.stack(candidate_batch, dim=0), semantics
                    )
                    cutout_normalized = _to_normalized(
                        torch.stack(cutout_batch, dim=0), semantics
                    )
                    paired = torch.cat(
                        (candidate_normalized, cutout_normalized), dim=0
                    ).to(device=device, dtype=torch.float32, non_blocking=True)
                    paired_mask = (
                        image_mask.repeat(2, 1, 1, 1)
                        if image_mask is not None
                        else None
                    )
                    paired_bbox = bbox.repeat(2, 1) if bbox is not None else None
                    transformed_logits, transformed_probabilities, transformed_pooled = _forward_outputs(
                        model,
                        paired,
                        image_mask=paired_mask,
                        bbox=paired_bbox,
                    )
                    candidate_response = response_scalars(
                        clean_logits,
                        clean_probabilities,
                        clean_pooled,
                        transformed_logits[:batch],
                        transformed_probabilities[:batch],
                        transformed_pooled[:batch],
                    )
                    cutout_response = response_scalars(
                        clean_logits,
                        clean_probabilities,
                        clean_pooled,
                        transformed_logits[batch:],
                        transformed_probabilities[batch:],
                        transformed_pooled[batch:],
                    )
                    cutpaste_responses[position:stop, family_index, draw] = (
                        candidate_response.detach().cpu().numpy()
                    )
                    cutout_responses[position:stop, family_index, draw] = (
                        cutout_response.detach().cpu().numpy()
                    )
            position = stop
            if position % 128 < batch or position == rows:
                print(
                    json.dumps(
                        {
                            "stage": "cutpaste_response_extraction",
                            "condition": condition,
                            "processed": position,
                            "rows": rows,
                            "forward_views": position * (1 + 2 * len(FAMILIES) * DRAW_COUNT),
                            "elapsed_seconds": time.perf_counter() - started,
                        }
                    ),
                    flush=True,
                )
    if position != rows or clean_pooled_bank is None:
        raise RuntimeError("CutPaste extraction did not cover the full cohort")
    if device.type == "cuda":
        torch.cuda.synchronize(device)
    elapsed = float(time.perf_counter() - started)
    state_after = _model_state_sha256(model)
    geometry_summary = _geometry_summary(geometry)
    return {
        "condition": condition,
        "condition_id": int(condition_id),
        "indices": indices_bank,
        "targets": targets_bank,
        "clean_logits": clean_logits_bank,
        "clean_probabilities": clean_probability_bank,
        "clean_pooled": clean_pooled_bank,
        "cutpaste_responses": cutpaste_responses,
        "cutout_responses": cutout_responses,
        "geometry": geometry,
        "geometry_summary": geometry_summary,
        "visual_cache": visual_cache,
        "runtime": {
            "loader": dict(loader_summary),
            "elapsed_seconds": elapsed,
            "source_images_per_second": float(rows / max(elapsed, 1e-12)),
            "forward_views_per_second": float(
                rows * (1 + 2 * len(FAMILIES) * DRAW_COUNT) / max(elapsed, 1e-12)
            ),
            "peak_cuda_bytes": int(
                torch.cuda.max_memory_allocated(device) if device.type == "cuda" else 0
            ),
            "model_state_before_sha256": state_before,
            "model_state_after_sha256": state_after,
            "geometry_consumed_only_local_rng": bool(rng_local_only),
        },
    }


def _tensor_to_pil(tensor: Tensor, semantics: Mapping[str, object]) -> Image.Image:
    rgb = _to_rgb(tensor, semantics)
    if rgb.ndim == 4:
        if int(rgb.shape[0]) != 1:
            raise ValueError("Contact-sheet tensor batch must contain one image")
        rgb = rgb[0]
    array = (rgb.permute(1, 2, 0).numpy() * 255.0).round().astype(np.uint8)
    return Image.fromarray(array, mode="RGB")


def _draw_geometry(
    image: Image.Image,
    record: Mapping[str, object],
    *,
    draw_source: bool,
) -> Image.Image:
    result = image.copy()
    draw = ImageDraw.Draw(result)
    if draw_source:
        sx0 = int(record["source_left"])
        sy0 = int(record["source_top"])
        sx1 = sx0 + int(record["source_width"]) - 1
        sy1 = sy0 + int(record["source_height"]) - 1
        draw.rectangle((sx0, sy0, sx1, sy1), outline=(0, 220, 80), width=2)
    dx0 = int(record["destination_mask_x0"])
    dy0 = int(record["destination_mask_y0"])
    dx1 = int(record["destination_mask_x1"]) - 1
    dy1 = int(record["destination_mask_y1"]) - 1
    draw.rectangle((dx0, dy0, dx1, dy1), outline=(230, 30, 30), width=2)
    return result


def render_contact_sheet(
    path: Path,
    *,
    visual_cache: Mapping[int, Mapping[str, object]],
    review_indices: Sequence[int],
    cohort: Sequence[CleanTrainRow],
    semantics: Mapping[str, object],
) -> Dict[str, object]:
    tile = int(semantics["image_size"])
    left = 205
    top = 34
    columns = (
        "clean + geometry",
        "regular CutPaste",
        "regular Cutout",
        "scar CutPaste",
        "scar Cutout",
    )
    ordered = [int(value) for value in review_indices]
    if set(visual_cache) != set(ordered):
        raise ValueError("CutPaste contact-sheet cache differs from fixed review rows")
    cohort_by_index = {row.sample_index: row for row in cohort}
    canvas = Image.new(
        "RGB", (left + tile * len(columns), top + tile * len(ordered)), "white"
    )
    draw = ImageDraw.Draw(canvas)
    font = ImageFont.load_default()
    for column_index, label in enumerate(columns):
        draw.text((left + column_index * tile + 6, 9), label, fill="black", font=font)
    rows = []
    for row_index, sample_index in enumerate(ordered):
        row = cohort_by_index[sample_index]
        cache = visual_cache[sample_index]
        regular_geometry = cache["regular_geometry"]
        scar_geometry = cache["scar_geometry"]
        clean = _tensor_to_pil(cache["clean"], semantics)
        regular_candidate = _tensor_to_pil(cache["regular_candidate"], semantics)
        regular_cutout = _tensor_to_pil(cache["regular_cutout"], semantics)
        scar_candidate = _tensor_to_pil(cache["scar_candidate"], semantics)
        scar_cutout = _tensor_to_pil(cache["scar_cutout"], semantics)
        images = (
            _draw_geometry(clean, regular_geometry, draw_source=True),
            _draw_geometry(regular_candidate, regular_geometry, draw_source=False),
            _draw_geometry(regular_cutout, regular_geometry, draw_source=False),
            _draw_geometry(scar_candidate, scar_geometry, draw_source=True),
            _draw_geometry(scar_cutout, scar_geometry, draw_source=False),
        )
        y = top + row_index * tile
        category = "TP" if row.target == FOCUS_CLASS else f"FP target={row.target}"
        draw.text(
            (8, y + 12),
            f"{category}\nidx={sample_index}\nfold={row.fold}\ngreen=source\nred=destination",
            fill="black",
            font=font,
            spacing=4,
        )
        for column_index, image in enumerate(images):
            canvas.paste(image, (left + column_index * tile, y))
        rows.append(
            {
                "sample_index": sample_index,
                "target": int(row.target),
                "fold": int(row.fold),
                "source_stem": row.source_stem,
            }
        )
    canvas.save(path, format="PNG", optimize=True)
    return {
        "path": str(Path(path).resolve()),
        "sha256": _sha256(path),
        "rows": rows,
        "columns": list(columns),
        "legend": {"source": "green", "destination": "red"},
    }


def _save_raw_payload(path: Path, extraction: Mapping[str, object]) -> None:
    np.savez_compressed(
        path,
        indices=np.asarray(extraction["indices"], dtype=np.int64),
        targets=np.asarray(extraction["targets"], dtype=np.int64),
        clean_logits=np.asarray(extraction["clean_logits"], dtype=np.float32),
        clean_probabilities=np.asarray(
            extraction["clean_probabilities"], dtype=np.float32
        ),
        clean_pooled=np.asarray(extraction["clean_pooled"], dtype=np.float32),
        cutpaste_responses=np.asarray(
            extraction["cutpaste_responses"], dtype=np.float32
        ),
        cutout_responses=np.asarray(
            extraction["cutout_responses"], dtype=np.float32
        ),
    )


def _load_raw_payload(path: Path) -> Dict[str, np.ndarray]:
    with np.load(path, allow_pickle=False) as payload:
        required = {
            "indices",
            "targets",
            "clean_logits",
            "clean_probabilities",
            "clean_pooled",
            "cutpaste_responses",
            "cutout_responses",
        }
        if set(payload.files) != required:
            raise ValueError(f"Raw CutPaste payload fields differ: {payload.files}")
        return {key: np.asarray(payload[key]) for key in required}


def _prediction_rows(
    *,
    cohort: Sequence[CleanTrainRow],
    scores: Mapping[str, np.ndarray],
    actions: Mapping[str, np.ndarray],
    clean_probabilities: np.ndarray,
) -> List[Dict[str, object]]:
    rivals = _rival_predictions(clean_probabilities)
    rows = []
    for position, source in enumerate(cohort):
        row: Dict[str, object] = {
            "sample_index": int(source.sample_index),
            "source_stem": source.source_stem,
            "image_path": str(source.image_path),
            "fold": int(source.fold),
            "target": int(source.target),
            "binary_tp_label": int(source.target == FOCUS_CLASS),
            "keeper_prediction": FOCUS_CLASS,
            "nonclass1_rival_prediction": int(rivals[position]),
        }
        for role in ROLE_NAMES:
            accepted = bool(actions[role][position])
            row[f"{role}_score"] = float(scores[role][position])
            row[f"{role}_accepted"] = accepted
            row[f"{role}_action_prediction"] = int(
                FOCUS_CLASS if accepted else rivals[position]
            )
        rows.append(row)
    return rows


def _fold_summary_rows(analysis: Mapping[str, object]) -> List[Dict[str, object]]:
    metrics = analysis["metrics"]
    rows = []
    for fold in FOLDS:
        row: Dict[str, object] = {"fold": int(fold)}
        for role in ROLE_NAMES:
            fold_metrics = metrics[role]["per_fold"][fold]
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
                row[f"{role}_{key}"] = fold_metrics[key]
        rows.append(row)
    return rows


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


def _readout_states_converged(states: Mapping[str, object]) -> bool:
    return all(
        bool(row["converged"])
        for role in ROLE_NAMES
        for row in states[role]["folds"]
    )


def _feature_dimensions(features: Mapping[str, np.ndarray]) -> Dict[str, int]:
    return {role: int(np.asarray(values).shape[1]) for role, values in features.items()}


def replay_summary(summary_path: Path) -> Dict[str, object]:
    resolved = Path(summary_path).expanduser().resolve()
    summary = json.loads(resolved.read_text(encoding="utf-8"))
    if summary.get("mode") != "cutpaste_surface_response_a0_train_information_gate":
        raise ValueError("Summary mode is not the locked CutPaste A0 audit")
    manifest = _verify_manifest(resolved.parent)
    raw = _load_raw_payload(resolved.parent / "raw_responses_clean.npz")
    readout_states = json.loads(
        (resolved.parent / "readout_state_clean.json").read_text(encoding="utf-8")
    )
    cohort_rows = _read_clean_train_rows(
        Path(summary["provenance"]["files"]["cidt_predictions"]["path"])
    )
    cohort = _cohort_from_rows(cohort_rows)
    folds = np.asarray([row.fold for row in cohort], dtype=np.int64)
    sources = [row.source_stem for row in cohort]
    labels = (np.asarray(raw["targets"], dtype=np.int64) == FOCUS_CLASS).astype(
        np.int64
    )
    features, mapping, hashes = build_readout_features(
        clean_logits=raw["clean_logits"],
        clean_probabilities=raw["clean_probabilities"],
        cutpaste_responses=raw["cutpaste_responses"],
        cutout_responses=raw["cutout_responses"],
        folds=folds,
        source_stems=sources,
        sample_indices=raw["indices"],
    )
    candidate_stats = descriptor_statistics(
        aggregate_responses(raw["cutpaste_responses"])
    )
    refit_scores, refit_actions, refit_states = fit_clean_oof_readouts(
        features, labels, folds, sources, seed=SEED
    )
    applied_scores, applied_actions = apply_readout_states(
        features, folds, readout_states
    )
    analysis = build_analysis(
        labels=labels,
        targets=raw["targets"],
        folds=folds,
        scores=refit_scores,
        actions=refit_actions,
        clean_probabilities=raw["clean_probabilities"],
        candidate_descriptor_stats=candidate_stats,
    )
    analysis_difference = _recursive_numeric_difference(
        summary["analysis"], analysis
    )
    state_difference = _recursive_numeric_difference(readout_states, refit_states)
    with (resolved.parent / "oof_predictions_clean.csv").open(
        "r", encoding="utf-8", newline=""
    ) as handle:
        prediction_rows = list(csv.DictReader(handle))
    persisted_scores = {
        role: np.asarray(
            [float(row[f"{role}_score"]) for row in prediction_rows],
            dtype=np.float64,
        )
        for role in ROLE_NAMES
    }
    refit_score_difference = _maximum_role_difference(
        persisted_scores, refit_scores
    )
    applied_score_difference = _maximum_role_difference(
        refit_scores, applied_scores
    )
    applied_actions_exact = all(
        np.array_equal(refit_actions[role], applied_actions[role])
        for role in ROLE_NAMES
    )
    mapping_hash = _records_sha256(mapping)
    passed = bool(
        analysis_difference <= 1e-12
        and state_difference <= 1e-12
        and refit_score_difference <= 1e-12
        and applied_score_difference <= 1e-10
        and applied_actions_exact
        and hashes == summary["descriptor_sha256"]
        and mapping_hash == summary["source_derangement"]["mapping_sha256"]
    )
    if not passed:
        raise ValueError("Independent CutPaste response/readout replay failed")
    return {
        "mode": "cutpaste_surface_response_a0_replay",
        "passed": passed,
        "analysis_maximum_numeric_difference": analysis_difference,
        "readout_state_maximum_numeric_difference": state_difference,
        "refit_score_maximum_abs_difference": refit_score_difference,
        "applied_score_maximum_abs_difference": applied_score_difference,
        "applied_actions_exact": applied_actions_exact,
        "descriptor_hashes_exact": True,
        "derangement_mapping_hash_exact": True,
        **manifest,
    }


def _regenerate_geometry(
    *,
    args: argparse.Namespace,
    checkpoint: Mapping[str, object],
    rows: Sequence[CleanTrainRow],
    cohort: Sequence[CleanTrainRow],
) -> Dict[str, object]:
    base_dataset, transform, dataset_summary = _build_dataset(
        checkpoint, rows, Path(args.data)
    )
    loader, loader_summary = _make_condition_loader(
        base_dataset=base_dataset,
        transform=transform,
        indices=[row.sample_index for row in cohort],
        brightness=1.0,
        contrast=1.0,
        batch_size=BATCH_SIZE,
        num_workers=NUM_WORKERS,
        context="cutpaste_surface_response_a0_geometry_replay",
    )
    records: List[Dict[str, object]] = []
    position = 0
    local_rng_only = True
    started = time.perf_counter()
    for images, targets, metadata in loader:
        batch = int(targets.numel())
        stop = position + batch
        sample_indices = metadata["sample_index"].detach().cpu().long()
        if sample_indices.tolist() != [
            row.sample_index for row in cohort[position:stop]
        ]:
            raise ValueError("Geometry replay loader changed cohort order")
        rgb = _to_rgb(images, dataset_summary["semantics"])
        crop_bboxes = metadata["crop_bbox"].detach().cpu().float()
        image_masks = metadata["image_mask"].detach().cpu().bool()
        support_masks = [
            _bbox_support(
                rgb[local_index],
                crop_bboxes[local_index],
                image_masks[local_index],
            )
            for local_index in range(batch)
        ]
        before = _global_rng_snapshot()
        for family in FAMILIES:
            for draw in range(DRAW_COUNT):
                for local_index, sample_index in enumerate(sample_indices.tolist()):
                    _, _, record = make_cutpaste_pair(
                        rgb[local_index],
                        crop_bboxes[local_index],
                        image_masks[local_index],
                        sample_index=int(sample_index),
                        draw=int(draw),
                        family=family,
                        condition_id=0,
                        support=support_masks[local_index],
                    )
                    records.append(record)
        after = _global_rng_snapshot()
        local_rng_only = local_rng_only and _global_rng_equal(before, after)
        position = stop
        if position % 256 < batch or position == len(cohort):
            print(
                json.dumps(
                    {
                        "stage": "cutpaste_geometry_replay",
                        "processed": position,
                        "rows": len(cohort),
                        "elapsed_seconds": time.perf_counter() - started,
                    }
                ),
                flush=True,
            )
    if position != len(cohort):
        raise RuntimeError("Geometry replay did not cover the full cohort")
    summary = _geometry_summary(records)
    summary["local_rng_only"] = bool(local_rng_only)
    summary["loader"] = dict(loader_summary)
    summary["elapsed_seconds"] = float(time.perf_counter() - started)
    return summary


def geometry_replay_summary(
    summary_path: Path, args: argparse.Namespace
) -> Dict[str, object]:
    resolved = Path(summary_path).expanduser().resolve()
    summary = json.loads(resolved.read_text(encoding="utf-8"))
    if summary.get("mode") != "cutpaste_surface_response_a0_train_information_gate":
        raise ValueError("Summary mode is not the locked CutPaste A0 audit")
    manifest_before = _verify_manifest(resolved.parent)
    provenance = verify_locked_inputs(args)
    repo_state = _repo_state()
    if not bool(repo_state["tracked_worktree_clean"]) or not bool(
        repo_state["head_matches_upstream"]
    ):
        raise ValueError("Geometry replay requires the same clean pushed repository")
    all_rows = _read_clean_train_rows(
        Path(provenance["files"]["cidt_predictions"]["path"])
    )
    cohort = _cohort_from_rows(all_rows)
    checkpoint = load_checkpoint(Path(provenance["files"]["keeper"]["path"]), map_location="cpu")
    observed = _regenerate_geometry(
        args=args, checkpoint=checkpoint, rows=all_rows, cohort=cohort
    )
    expected = summary["geometry"]
    passed = bool(
        observed["passed"]
        and observed["local_rng_only"]
        and int(observed["rows"]) == int(expected["rows"])
        and observed["records_sha256"] == expected["records_sha256"]
    )
    result = {
        "second_process": True,
        "passed": passed,
        "expected_records_sha256": expected["records_sha256"],
        "observed_records_sha256": observed["records_sha256"],
        "rows": observed["rows"],
        "local_rng_only": observed["local_rng_only"],
        "elapsed_seconds": observed["elapsed_seconds"],
        "manifest_sha256_before_replay": manifest_before["manifest_sha256"],
    }
    summary["external_geometry_replay"] = result
    summary["structural_gates"]["second_process_geometry_replay_exact"] = passed
    structural_passed = all(bool(value) for value in summary["structural_gates"].values())
    automated_passed = structural_passed and bool(
        summary["analysis"]["mechanism_gates_passed"]
    )
    summary["structural_gates_passed"] = structural_passed
    summary["automated_gate_passed"] = automated_passed
    summary["status"] = (
        "awaiting_visual_review" if automated_passed else "rejected_automated_gate"
    )
    _write_json(resolved, summary)
    manifest_after = _write_manifest(resolved.parent)
    result["summary_sha256"] = _sha256(resolved)
    result["manifest_sha256"] = _sha256(manifest_after)
    return result


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
            "Visual review summary SHA differs: "
            f"expected={expected_summary_sha256}, observed={observed_summary_sha}"
        )
    manifest_before = _verify_manifest(resolved.parent)
    summary = json.loads(resolved.read_text(encoding="utf-8"))
    if summary.get("mode") != "cutpaste_surface_response_a0_train_information_gate":
        raise ValueError("Visual review target is not CutPaste A0")
    if not bool(summary.get("external_geometry_replay", {}).get("passed", False)):
        raise ValueError("Visual review requires a passing second-process geometry replay")
    contact_path = resolved.parent / "cutpaste_contact_sheet_clean.png"
    contact_sha = _sha256(contact_path)
    if contact_sha != summary["contact_sheet"]["sha256"]:
        raise ValueError("Visual review contact sheet differs from the locked summary")
    geometry_passed = str(result) == "pass"
    automated_passed = bool(summary["automated_gate_passed"])
    clean_passed = bool(automated_passed and geometry_passed)
    summary["visual_review"] = {
        "required": True,
        "completed": True,
        "geometry_passed": geometry_passed,
        "decision": str(result),
        "reviewed_summary_sha256": observed_summary_sha,
        "contact_sheet_sha256": contact_sha,
        "manifest_sha256_before_review": manifest_before["manifest_sha256"],
    }
    summary["clean_a0_gates_passed"] = clean_passed
    summary["stage_b_conditions_authorized"] = clean_passed
    summary["trainer_integration_authorized"] = False
    summary["short_pair_authorized"] = False
    summary["full_train_authorized"] = False
    summary["current_command_update_authorized"] = False
    summary["status"] = (
        "passed_clean_a0_gates_stage_b_authorized"
        if clean_passed
        else (
            "rejected_visual_gate"
            if automated_passed and not geometry_passed
            else "rejected_automated_gate"
        )
    )
    _write_json(resolved, summary)
    manifest_after = _write_manifest(resolved.parent)
    return {
        "mode": "cutpaste_surface_response_a0_visual_review",
        "decision": str(result),
        "automated_gate_passed": automated_passed,
        "clean_a0_gates_passed": clean_passed,
        "stage_b_conditions_authorized": clean_passed,
        "full_train_authorized": False,
        "current_command_update_authorized": False,
        "reviewed_summary_sha256": observed_summary_sha,
        "summary_sha256": _sha256(resolved),
        "manifest_sha256": _sha256(manifest_after),
    }


def preflight(args: argparse.Namespace) -> Dict[str, object]:
    _validate_locked_args(args)
    output_dir = Path(args.output_dir).expanduser().resolve()
    if output_dir.exists():
        raise FileExistsError(f"Preflight output must not exist: {output_dir}")
    provenance = verify_locked_inputs(args)
    engineering = cutpaste_engineering_checks()
    if not bool(engineering["passed"]):
        raise RuntimeError(f"CutPaste engineering checks failed: {engineering}")
    return {
        "mode": "cutpaste_surface_response_a0_preflight",
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


def run_audit(args: argparse.Namespace) -> Dict[str, object]:
    _validate_locked_args(args)
    if os.environ.get("TRKH_CUTPASTE_A0_PREFLIGHT") != "passed":
        raise RuntimeError(
            "Formal CutPaste A0 must be launched through the locked PowerShell preflight"
        )
    if not torch.cuda.is_available():
        raise RuntimeError("CUDA is unavailable for formal CutPaste A0")
    provenance = verify_locked_inputs(args)
    repo_state = _repo_state()
    if not bool(repo_state["tracked_worktree_clean"]) or not bool(
        repo_state["head_matches_upstream"]
    ):
        raise ValueError(
            f"Formal CutPaste A0 requires clean pushed tracked state: {repo_state}"
        )
    output_dir = _prepare_output_dir(args.output_dir)
    set_seed(SEED, deterministic=True)
    torch.backends.cuda.matmul.allow_tf32 = False
    torch.backends.cudnn.allow_tf32 = True
    device = torch.device("cuda")
    engineering = cutpaste_engineering_checks()
    if not bool(engineering["passed"]):
        raise RuntimeError("CutPaste engineering preflight changed before formal run")

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
    review_indices = _review_indices(cohort)
    loader, loader_summary = _make_condition_loader(
        base_dataset=base_dataset,
        transform=transform,
        indices=[row.sample_index for row in cohort],
        brightness=1.0,
        contrast=1.0,
        batch_size=BATCH_SIZE,
        num_workers=NUM_WORKERS,
        context="cutpaste_surface_response_a0_clean",
    )
    extraction = extract_condition(
        model=model,
        loader=loader,
        loader_summary=loader_summary,
        cohort=cohort,
        device=device,
        semantics=dataset_summary["semantics"],
        condition="clean",
        condition_id=0,
        review_indices=review_indices,
    )
    targets = np.asarray(extraction["targets"], dtype=np.int64)
    indices = np.asarray(extraction["indices"], dtype=np.int64)
    folds = np.asarray([row.fold for row in cohort], dtype=np.int64)
    source_stems = [row.source_stem for row in cohort]
    labels = (targets == FOCUS_CLASS).astype(np.int64)
    expected_probabilities = np.asarray(
        [row.keeper_probabilities for row in cohort], dtype=np.float64
    )
    clean_probabilities = np.asarray(
        extraction["clean_probabilities"], dtype=np.float64
    )
    cidt_probability_error = float(
        np.max(np.abs(clean_probabilities - expected_probabilities))
    )
    cidt_argmax_exact = bool(
        np.array_equal(
            clean_probabilities.argmax(axis=1),
            np.asarray([row.keeper_prediction for row in cohort], dtype=np.int64),
        )
    )
    cohort_indices = np.asarray(
        [row.sample_index for row in cohort], dtype=np.int64
    )
    cohort_targets = np.asarray([row.target for row in cohort], dtype=np.int64)
    if not np.array_equal(indices, cohort_indices) or not np.array_equal(
        targets, cohort_targets
    ):
        raise ValueError("Formal CutPaste extraction changed cohort identity")

    features, derangement_mapping, descriptor_hashes = build_readout_features(
        clean_logits=extraction["clean_logits"],
        clean_probabilities=clean_probabilities,
        cutpaste_responses=extraction["cutpaste_responses"],
        cutout_responses=extraction["cutout_responses"],
        folds=folds,
        source_stems=source_stems,
        sample_indices=indices,
    )
    candidate_descriptor_stats = descriptor_statistics(
        aggregate_responses(extraction["cutpaste_responses"])
    )
    cutout_descriptor_stats = descriptor_statistics(
        aggregate_responses(extraction["cutout_responses"])
    )
    paired_descriptor_stats = descriptor_statistics(
        aggregate_responses(
            np.asarray(extraction["cutpaste_responses"], dtype=np.float64)
            - np.asarray(extraction["cutout_responses"], dtype=np.float64)
        )
    )
    scores, actions, readout_states = fit_clean_oof_readouts(
        features, labels, folds, source_stems, seed=SEED
    )
    analysis = build_analysis(
        labels=labels,
        targets=targets,
        folds=folds,
        scores=scores,
        actions=actions,
        clean_probabilities=clean_probabilities,
        candidate_descriptor_stats=candidate_descriptor_stats,
    )

    raw_path = output_dir / "raw_responses_clean.npz"
    geometry_path = output_dir / "geometry_clean.csv"
    derangement_path = output_dir / "source_derangement_clean.csv"
    readout_path = output_dir / "readout_state_clean.json"
    prediction_path = output_dir / "oof_predictions_clean.csv"
    fold_path = output_dir / "fold_summary_clean.csv"
    contact_path = output_dir / "cutpaste_contact_sheet_clean.png"
    descriptor_path = output_dir / "descriptor_statistics_clean.json"
    _save_raw_payload(raw_path, extraction)
    _write_csv(geometry_path, extraction["geometry"])
    _write_csv(derangement_path, derangement_mapping)
    _write_json(readout_path, readout_states)
    _write_csv(
        prediction_path,
        _prediction_rows(
            cohort=cohort,
            scores=scores,
            actions=actions,
            clean_probabilities=clean_probabilities,
        ),
    )
    _write_csv(fold_path, _fold_summary_rows(analysis))
    descriptor_statistics_payload = {
        "cutpaste_candidate": candidate_descriptor_stats,
        "cutout_control": cutout_descriptor_stats,
        "paired_contrast": paired_descriptor_stats,
    }
    _write_json(descriptor_path, descriptor_statistics_payload)
    contact_sheet = render_contact_sheet(
        contact_path,
        visual_cache=extraction["visual_cache"],
        review_indices=review_indices,
        cohort=cohort,
        semantics=dataset_summary["semantics"],
    )

    persisted = _load_raw_payload(raw_path)
    replay_features, replay_mapping, replay_hashes = build_readout_features(
        clean_logits=persisted["clean_logits"],
        clean_probabilities=persisted["clean_probabilities"],
        cutpaste_responses=persisted["cutpaste_responses"],
        cutout_responses=persisted["cutout_responses"],
        folds=folds,
        source_stems=source_stems,
        sample_indices=persisted["indices"],
    )
    replay_scores, replay_actions, replay_states = fit_clean_oof_readouts(
        replay_features, labels, folds, source_stems, seed=SEED
    )
    applied_scores, applied_actions = apply_readout_states(
        replay_features, folds, readout_states
    )
    replay_candidate_stats = descriptor_statistics(
        aggregate_responses(persisted["cutpaste_responses"])
    )
    replay_analysis = build_analysis(
        labels=labels,
        targets=targets,
        folds=folds,
        scores=replay_scores,
        actions=replay_actions,
        clean_probabilities=persisted["clean_probabilities"],
        candidate_descriptor_stats=replay_candidate_stats,
    )
    replay = {
        "analysis_maximum_numeric_difference": _recursive_numeric_difference(
            analysis, replay_analysis
        ),
        "readout_state_maximum_numeric_difference": _recursive_numeric_difference(
            readout_states, replay_states
        ),
        "refit_score_maximum_abs_difference": _maximum_role_difference(
            scores, replay_scores
        ),
        "applied_score_maximum_abs_difference": _maximum_role_difference(
            scores, applied_scores
        ),
        "refit_actions_exact": all(
            np.array_equal(actions[role], replay_actions[role])
            for role in ROLE_NAMES
        ),
        "applied_actions_exact": all(
            np.array_equal(actions[role], applied_actions[role])
            for role in ROLE_NAMES
        ),
        "descriptor_hashes_exact": descriptor_hashes == replay_hashes,
        "derangement_mapping_exact": derangement_mapping == replay_mapping,
    }
    replay["passed"] = bool(
        float(replay["analysis_maximum_numeric_difference"]) <= 1e-12
        and float(replay["readout_state_maximum_numeric_difference"]) <= 1e-12
        and float(replay["refit_score_maximum_abs_difference"]) <= 1e-12
        and float(replay["applied_score_maximum_abs_difference"]) <= 1e-10
        and bool(replay["refit_actions_exact"])
        and bool(replay["applied_actions_exact"])
        and bool(replay["descriptor_hashes_exact"])
        and bool(replay["derangement_mapping_exact"])
    )

    geometry_summary = extraction["geometry_summary"]
    loader_effective_workers = int(
        extraction["runtime"]["loader"].get("effective_num_workers", -1)
    )
    source_fold_overlap = 0
    for held_fold in FOLDS:
        held_sources = {
            source_stems[index]
            for index in np.flatnonzero(folds == held_fold).tolist()
        }
        fit_sources = {
            source_stems[index]
            for index in np.flatnonzero(folds != held_fold).tolist()
        }
        source_fold_overlap += len(held_sources.intersection(fit_sources))
    structural_gates = {
        "locked_hashes_verified": True,
        "official_torchvision_exact_and_clean": bool(
            provenance["official_repository"]["worktree_clean"]
        ),
        "protected_untracked_identity_exact": bool(
            provenance["protected_untracked"]["passed"]
        ),
        "repo_clean_and_pushed": bool(repo_state["tracked_worktree_clean"])
        and bool(repo_state["head_matches_upstream"]),
        "locked_launcher_compile_pyflakes_focused_full_tests_ps_parse_attested": os.environ.get(
            "TRKH_CUTPASTE_A0_PREFLIGHT"
        )
        == "passed",
        "train_split_only": bool(dataset_summary["train_paths_only"]),
        "dataset_mapping_exact": bool(dataset_summary["paths_exact"]),
        "cohort_rows_order_support_exact": len(cohort) == EXPECTED_COHORT_ROWS
        and int(labels.sum()) == EXPECTED_POSITIVES
        and int((labels == 0).sum()) == EXPECTED_NEGATIVES
        and _cohort_index_sha256(indices.tolist())
        == EXPECTED_ORDERED_INDEX_SHA256,
        "five_source_folds_and_zero_overlap": set(folds.tolist()) == set(FOLDS)
        and source_fold_overlap == 0,
        "cidt_probability_error_le_2e_5": cidt_probability_error
        <= MAX_CIDT_PROBABILITY_ERROR,
        "cidt_argmax_exact": cidt_argmax_exact,
        "geometry_checks_passed": bool(geometry_summary["passed"]),
        "geometry_uses_only_local_rng": bool(
            extraction["runtime"]["geometry_consumed_only_local_rng"]
        ),
        "second_process_geometry_replay_exact": False,
        "all_responses_features_scores_finite": bool(
            np.isfinite(extraction["cutpaste_responses"]).all()
            and np.isfinite(extraction["cutout_responses"]).all()
            and all(np.isfinite(value).all() for value in features.values())
            and all(np.isfinite(value).all() for value in scores.values())
        ),
        "feature_dimensions_exact": _feature_dimensions(features)
        == {
            "base_only": BASE_DIM,
            "cutout_control": BASE_DIM + RESPONSE_DIM,
            "cutpaste_candidate": BASE_DIM + RESPONSE_DIM,
            "paired_contrast": BASE_DIM + RESPONSE_DIM,
            "source_deranged": BASE_DIM + RESPONSE_DIM,
        },
        "all_25_readouts_converged_without_retry": _readout_states_converged(
            readout_states
        ),
        "candidate_response_effective_rank_ge_8": float(
            candidate_descriptor_stats["effective_rank"]
        )
        >= 8.0,
        "model_state_bit_exact": extraction["runtime"][
            "model_state_before_sha256"
        ]
        == extraction["runtime"]["model_state_after_sha256"],
        "requested_workers_4_effective_workers_4": NUM_WORKERS == 4
        and loader_effective_workers == 4,
        "peak_cuda_allocation_le_3_5_gib": int(
            extraction["runtime"]["peak_cuda_bytes"]
        )
        <= int(MAX_PEAK_CUDA_GIB * 1024**3),
        "raw_response_readout_replay_exact": bool(replay["passed"]),
        "contact_sheet_12_rows_5_columns": len(contact_sheet["rows"]) == 12
        and len(contact_sheet["columns"]) == 5,
    }
    structural_without_external = all(
        bool(value)
        for key, value in structural_gates.items()
        if key != "second_process_geometry_replay_exact"
    )
    artifacts = {
        "raw_responses": _artifact_record(raw_path, root=output_dir),
        "geometry": _artifact_record(geometry_path, root=output_dir),
        "source_derangement": _artifact_record(derangement_path, root=output_dir),
        "readout_state": _artifact_record(readout_path, root=output_dir),
        "oof_predictions": _artifact_record(prediction_path, root=output_dir),
        "fold_summary": _artifact_record(fold_path, root=output_dir),
        "descriptor_statistics": _artifact_record(descriptor_path, root=output_dir),
        "contact_sheet": _artifact_record(contact_path, root=output_dir),
    }
    summary = {
        "mode": "cutpaste_surface_response_a0_train_information_gate",
        "status": "awaiting_second_process_geometry_replay",
        "split": "train",
        "rows": len(cohort),
        "validation_data_used": False,
        "test_data_used": False,
        "class_names": list(class_names),
        "focus_class": FOCUS_CLASS,
        "restricted_negative_classes": list(RESTRICTED_NEGATIVE_CLASSES),
        "families": list(FAMILIES),
        "draw_count": DRAW_COUNT,
        "response_names": list(RESPONSE_NAMES),
        "aggregations": list(AGGREGATIONS),
        "role_names": list(ROLE_NAMES),
        "feature_dimensions": _feature_dimensions(features),
        "provenance": provenance,
        "repo_state": repo_state,
        "dataset": dataset_summary,
        "cohort": {
            "rows": len(cohort),
            "positives": int(labels.sum()),
            "negatives": int((labels == 0).sum()),
            "fold_counts": EXPECTED_FOLD_COUNTS,
            "ordered_index_sha256": _cohort_index_sha256(indices.tolist()),
            "source_fold_overlap": source_fold_overlap,
        },
        "cidt_replay": {
            "probability_max_abs_difference": cidt_probability_error,
            "probability_tolerance": MAX_CIDT_PROBABILITY_ERROR,
            "argmax_exact": cidt_argmax_exact,
        },
        "engineering": engineering,
        "geometry": geometry_summary,
        "external_geometry_replay": {
            "second_process": True,
            "passed": False,
            "pending": True,
        },
        "descriptor_sha256": descriptor_hashes,
        "descriptor_statistics": descriptor_statistics_payload,
        "source_derangement": {
            "rows": len(derangement_mapping),
            "mapping_sha256": _records_sha256(derangement_mapping),
            "same_fold": True,
            "same_margin_quartile": True,
            "different_source": True,
            "labels_used_to_build_mapping": False,
        },
        "readout": {
            "C": READOUT_C,
            "class_weight": "balanced",
            "solver": "lbfgs",
            "max_iter": READOUT_MAX_ITER,
            "tol": READOUT_TOLERANCE,
            "minimum_fit_tp_retention": MIN_FIT_TP_RETENTION,
            "states": readout_states,
        },
        "analysis": analysis,
        "runtime": {
            "device": str(device),
            "torch_version": torch.__version__,
            "cuda_version": torch.version.cuda,
            "gpu_name": torch.cuda.get_device_name(device),
            "batch_size": BATCH_SIZE,
            "effective_paired_gpu_batch": 2 * BATCH_SIZE,
            "requested_num_workers": NUM_WORKERS,
            **extraction["runtime"],
            "peak_cuda_gib": float(
                int(extraction["runtime"]["peak_cuda_bytes"]) / 1024**3
            ),
        },
        "internal_replay": replay,
        "contact_sheet": contact_sheet,
        "visual_review": {
            "required": True,
            "completed": False,
            "geometry_passed": None,
        },
        "structural_gates": structural_gates,
        "structural_gates_passed_without_external_geometry": structural_without_external,
        "structural_gates_passed": False,
        "automated_gate_passed": False,
        "clean_a0_gates_passed": False,
        "stage_b_conditions_authorized": False,
        "trainer_integration_authorized": False,
        "short_pair_authorized": False,
        "full_train_authorized": False,
        "current_command_update_authorized": False,
        "artifacts": artifacts,
        "guardrail": (
            "A second-process geometry replay and fixed-sheet visual review are "
            "required. Validation/test, trainer integration, smoke/probe/full train, "
            "and current-best command changes remain forbidden."
        ),
    }
    summary_path = output_dir / "summary.json"
    _write_json(summary_path, summary)
    _write_manifest(output_dir)
    return summary


def main(argv: Optional[Sequence[str]] = None) -> int:
    args = parse_args(argv)
    selected_modes = sum(
        int(value)
        for value in (
            bool(args.preflight_only),
            args.replay_summary is not None,
            args.geometry_replay_summary is not None,
            args.finalize_visual_review is not None,
        )
    )
    if selected_modes > 1 and not (
        args.finalize_visual_review is not None and args.replay_summary is not None
    ):
        raise ValueError("CutPaste A0 execution modes are mutually exclusive")
    if args.expected_summary_sha256 and args.finalize_visual_review is None:
        raise ValueError(
            "expected-summary-sha256 is valid only with finalize-visual-review"
        )
    if args.preflight_only:
        print(json.dumps(to_serializable(preflight(args)), indent=2), flush=True)
        return 0
    if args.geometry_replay_summary is not None:
        print(
            json.dumps(
                to_serializable(
                    geometry_replay_summary(args.geometry_replay_summary, args)
                ),
                indent=2,
            ),
            flush=True,
        )
        return 0
    if args.finalize_visual_review is not None:
        if args.replay_summary is None or not args.expected_summary_sha256:
            raise ValueError(
                "Visual finalization requires --replay-summary and expected SHA"
            )
        print(
            json.dumps(
                to_serializable(
                    finalize_visual_review(
                        args.replay_summary,
                        result=args.finalize_visual_review,
                        expected_summary_sha256=args.expected_summary_sha256,
                    )
                ),
                indent=2,
            ),
            flush=True,
        )
        return 0
    if args.replay_summary is not None:
        print(
            json.dumps(to_serializable(replay_summary(args.replay_summary)), indent=2),
            flush=True,
        )
        return 0
    print(json.dumps(to_serializable(run_audit(args)), indent=2), flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
