from __future__ import annotations

import argparse
import csv
import hashlib
import json
import subprocess
import time
import warnings
from pathlib import Path
from typing import Dict, List, Mapping, Optional, Sequence, Tuple

import numpy as np
import torch
import torch.nn.functional as F
from PIL import Image, ImageDraw, ImageFont
from sklearn.exceptions import ConvergenceWarning
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import roc_auc_score
from sklearn.preprocessing import StandardScaler
from torch import Tensor, nn
from torch.utils.data import DataLoader, Dataset

from trkh.core.config import load_data_spec, to_serializable
from trkh.core.utils import build_safe_dataloader_kwargs, set_seed
from trkh.data.dataset import MangoYOLOCropDataset
from trkh.evaluation.evaluate import resolve_crop_to_primary_object
from trkh.evaluation.robustness_eval import IdentityCorruption
from trkh.inference.inference import load_model
from trkh.models.model import classification_logits_from_features
from trkh.tools.audit_counterfactual_illumination_disagreement_readiness import (
    _SelectedConditionDataset,
    _assign_source_folds,
    _build_eval_transform,
    _classification_metrics,
    _dataset_identity,
    _split_source_stems,
)
from trkh.tools.build_precision_ensemble_checkpoint import _eval_semantics


EXPECTED_TRAIN_ROWS = 9215
FOCUS_CLASS = 1
RESTRICTED_NEGATIVE_CLASSES = (0, 2, 4)
FOLDS = (0, 1, 2, 3, 4)
STAGE_LAYERS = (2, 5, 8)
JIGSAW_GRIDS = (8, 4, 2)
ROLE_NAMES = (
    "clean_control",
    "pmg_aligned",
    "reverse_placebo",
    "deepest_placebo",
)
FEATURE_DIM = 256
ROLE_DIM = 4 * FEATURE_DIM
READOUT_C = 0.05
READOUT_TOL = 1e-6
READOUT_MAX_ITER = 1000
PROTOCOL_SHA256 = "7af3630a0639019e68c295497216d9cc6fad55f43f9ae81c4f20ea64d9f0a8c1"
PMG_COMMIT = "db7a7d7ab5fd91c2e322e1fcf200cbfbe192b51a"
PMG_TREE = "85949974008e279c885510bf564396c5f9dbe99e"
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

REPO_ROOT = Path(__file__).resolve().parents[2]
PMG_REPO = Path(
    r"D:\DataAI\external_sources\repositories\PMG-Progressive-Multi-Granularity-Training"
)
LOCKED_FILES: Tuple[Tuple[str, Path, str], ...] = (
    (
        "protocol",
        REPO_ROOT
        / "docs"
        / "TRKH_5CLASS_PMG_PROGRESSIVE_JIGSAW_SIGNAL_A0_PROTOCOL_20260720.md",
        PROTOCOL_SHA256,
    ),
    (
        "keeper",
        REPO_ROOT
        / "runs"
        / "probe_v8_yolof_pairroute_teacherfocusbinary015_boundarydrop_bboxprior_120b_2e_20260701"
        / "checkpoints"
        / "best.pt",
        "1f49d577240c69dc63c30af70db52ec2aa9da65a17aef1c4b1c09ece6c482677",
    ),
    (
        "keeper_launcher_args",
        REPO_ROOT
        / "runs"
        / "probe_v8_yolof_pairroute_teacherfocusbinary015_boundarydrop_bboxprior_120b_2e_20260701"
        / "launcher_args.json",
        "908a05cf66b2a01162cae62e4ff2251eaae1297d31e70510144e4954159b7eff",
    ),
    (
        "keeper_resolved_config",
        REPO_ROOT
        / "runs"
        / "probe_v8_yolof_pairroute_teacherfocusbinary015_boundarydrop_bboxprior_120b_2e_20260701"
        / "resolved_config.json",
        "e9c4f48917e333d2f34f61806bb54041b35f2217ebb23afb3bd0ced969854674",
    ),
    (
        "data_yaml",
        Path(r"D:\DataAI\AIEx\newdataset\yolo_f\data.yaml"),
        "716e33df24c63a9e9920f97b685199707fb84ab4c7154544f5dd9a3e00d884ef",
    ),
    (
        "cidt_predictions",
        REPO_ROOT
        / "runs"
        / "audit_cidt_readiness_full_train_20260714"
        / "predictions_all_conditions.csv",
        "2e0993752d58d99ea429bfefe1e2bfe6fa949e45aea1a26cc4bdfee97d4db21c",
    ),
    (
        "cidt_summary",
        REPO_ROOT
        / "runs"
        / "audit_cidt_readiness_full_train_20260714"
        / "summary.json",
        "d4891edf2963ab12385b7ce5bdc812ec3e19c5c098acd25c66eb557af541d7ad",
    ),
    (
        "current_best_commands",
        REPO_ROOT / "docs" / "TRKH_CURRENT_BEST_FULL_TRAIN_COMMANDS_20260706.txt",
        "36b9aa1a21b765829acf4c8321be147bd76297de4ccdb8a40e6dee8e37940faf",
    ),
    (
        "current_best_history",
        REPO_ROOT / "docs" / "TRKH_CURRENT_BEST_COMMAND_UPDATE_HISTORY.txt",
        "39bd2879ce66fddf36a953021ea1e40f8d9de6cb4334b9b825011b2b8dc98f53",
    ),
    (
        "pmg_paper",
        Path(r"D:\DataAI\external_sources\papers\PMG_ECCV_2020_accepted.pdf"),
        "ae70129965ea54459d6d59f1a6257bac7361d4972c63234740225e7be0f2da85",
    ),
    (
        "pmg_license",
        PMG_REPO / "LICENSE",
        "d50df6b22917cf3f93a9d9987339e88b972342edd83aedb49897c91b034f5270",
    ),
    (
        "pmg_train_source",
        PMG_REPO / "train.py",
        "8331cb58e5e37c77e7e8755b619c69e8b9bfb11dfe0ca532be64a019ecc9fed0",
    ),
    (
        "pmg_model_source",
        PMG_REPO / "model.py",
        "fc0ad1b6718e38d84699d7e19f096c5347fbb3f949c021bcc47edbdad4d421c9",
    ),
    (
        "pmg_utils_source",
        PMG_REPO / "utils.py",
        "44706f38646ac8c2bb2346ee89d6b4d0d9836e028f5cceadaa3f9f8c9fc06d28",
    ),
)


def parse_args(argv: Optional[Sequence[str]] = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Locked train-only PMG progressive-jigsaw information gate."
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=REPO_ROOT / "runs" / "audit_pmg_progressive_jigsaw_signal_a0_20260720",
    )
    parser.add_argument("--batch-size", type=int, default=64)
    parser.add_argument("--num-workers", type=int, default=4)
    parser.add_argument("--folds", type=int, default=5)
    parser.add_argument("--seed", type=int, default=20260714)
    parser.add_argument("--device", choices=("auto", "cpu", "cuda"), default="cuda")
    parser.add_argument("--preflight-only", action="store_true")
    parser.add_argument("--replay-summary", type=Path)
    parser.add_argument("--finalize-visual-review", choices=("pass", "fail"))
    parser.add_argument("--expected-summary-sha256")
    return parser.parse_args(argv)


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
    branch = _git_value(REPO_ROOT, "branch", "--show-current")
    tracked_status = _git_value(
        REPO_ROOT, "status", "--short", "--untracked-files=no"
    )
    full_status = _git_value(REPO_ROOT, "status", "--short")
    upstream = _git_value(REPO_ROOT, "rev-parse", "@{upstream}")
    counts = _git_value(
        REPO_ROOT, "rev-list", "--left-right", "--count", "HEAD...@{upstream}"
    ).split()
    return {
        "branch": branch,
        "head": head,
        "upstream": upstream,
        "ahead": int(counts[0]),
        "behind": int(counts[1]),
        "tracked_worktree_clean": tracked_status == "",
        "head_matches_upstream": head == upstream,
        "full_status": full_status.splitlines() if full_status else [],
    }


def _protected_untracked_state() -> Dict[str, object]:
    expected = {relative: sha256 for relative, sha256 in PROTECTED_UNTRACKED_FILES}
    observed_files = {
        path.relative_to(REPO_ROOT).as_posix()
        for path in (REPO_ROOT / "BaoCao").rglob("*")
        if path.is_file()
    }
    observed_files.update(
        relative
        for relative in (
            "deep-research-report (9).md",
            "deep-research-report (10).md",
        )
        if (REPO_ROOT / relative).is_file()
    )
    untracked_output = subprocess.check_output(
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
        value.decode("utf-8")
        for value in untracked_output.split(b"\0")
        if value
    }
    hashes = {
        relative: _sha256(REPO_ROOT / relative)
        for relative in sorted(expected)
        if (REPO_ROOT / relative).is_file()
    }
    checks = {
        "file_set_exact": observed_files == set(expected),
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
            for relative in sorted(expected)
            if relative in hashes
        ],
        "missing_or_extra": sorted(set(expected).symmetric_difference(observed_files)),
    }


def verify_locked_inputs() -> Dict[str, object]:
    files = {
        label: _verify_hash(path, expected, label)
        for label, path, expected in LOCKED_FILES
    }
    if not PMG_REPO.is_dir():
        raise FileNotFoundError(f"Official PMG repository is missing: {PMG_REPO}")
    commit = _git_value(PMG_REPO, "rev-parse", "HEAD")
    tree = _git_value(PMG_REPO, "rev-parse", "HEAD^{tree}")
    status = _git_value(PMG_REPO, "status", "--porcelain")
    remote = _git_value(PMG_REPO, "remote", "get-url", "origin")
    if commit != PMG_COMMIT or tree != PMG_TREE or status:
        raise ValueError(
            "Official PMG repository lock differs: "
            f"commit={commit}, tree={tree}, status={status!r}"
        )
    protected_untracked = _protected_untracked_state()
    if not bool(protected_untracked["passed"]):
        raise ValueError(
            "Protected untracked user payloads differ from the prospective lock: "
            f"{protected_untracked}"
        )
    return {
        "files": files,
        "official_repository": {
            "path": str(PMG_REPO.resolve()),
            "remote": remote,
            "commit": commit,
            "tree": tree,
            "worktree_clean": status == "",
        },
        "protected_untracked": protected_untracked,
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
    fieldnames = list(rows[0].keys())
    with Path(path).open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        for row in rows:
            if list(row.keys()) != fieldnames:
                raise ValueError("CSV row fields are not stable")
            writer.writerow(row)


def _derived_permutation_seed(seed: int, sample_index: int, grid: int) -> int:
    payload = f"pmg-a0:{int(seed)}:{int(sample_index)}:{int(grid)}".encode("ascii")
    value = int.from_bytes(hashlib.sha256(payload).digest()[:8], "little")
    return int(value & ((1 << 63) - 1))


def build_permutation_bank(
    sample_indices: Sequence[int], *, grid: int, seed: int
) -> Tensor:
    n = int(grid)
    if n not in JIGSAW_GRIDS:
        raise ValueError(f"PMG A0 grid must be one of {JIGSAW_GRIDS}; got {grid}")
    rows: List[Tensor] = []
    for sample_index in sample_indices:
        generator = torch.Generator(device="cpu")
        generator.manual_seed(
            _derived_permutation_seed(int(seed), int(sample_index), n)
        )
        rows.append(torch.randperm(n * n, generator=generator, dtype=torch.long))
    return torch.stack(rows, dim=0)


def apply_patch_permutation(
    tensor: Tensor, permutations: Tensor, *, grid: int
) -> Tensor:
    if tensor.ndim != 4:
        raise ValueError("Jigsaw tensor must have shape [B,C,H,W]")
    batch, channels, height, width = (int(value) for value in tensor.shape)
    n = int(grid)
    if height % n != 0 or width % n != 0:
        raise ValueError(f"Tensor shape {(height, width)} is not divisible by grid {n}")
    if tuple(permutations.shape) != (batch, n * n):
        raise ValueError(
            f"Permutation shape {tuple(permutations.shape)} != {(batch, n * n)}"
        )
    permutations = permutations.to(device=tensor.device, dtype=torch.long)
    expected = torch.arange(n * n, device=tensor.device, dtype=torch.long)
    if not torch.equal(torch.sort(permutations, dim=1).values, expected.expand(batch, -1)):
        raise ValueError("Every jigsaw row must be a permutation without replacement")
    patch_height = height // n
    patch_width = width // n
    patches = (
        tensor.reshape(batch, channels, n, patch_height, n, patch_width)
        .permute(0, 2, 4, 1, 3, 5)
        .reshape(batch, n * n, channels, patch_height, patch_width)
    )
    gather_index = permutations[:, :, None, None, None].expand_as(patches)
    shuffled = torch.gather(patches, dim=1, index=gather_index)
    return (
        shuffled.reshape(batch, n, n, channels, patch_height, patch_width)
        .permute(0, 3, 1, 4, 2, 5)
        .reshape(batch, channels, height, width)
    )


def jigsaw_engineering_checks(seed: int = 20260714) -> Dict[str, object]:
    image = torch.arange(2 * 3 * 256 * 256, dtype=torch.int64).reshape(2, 3, 256, 256)
    mask = ((image[:, :1] % 7) != 0).to(torch.bool)
    sample_indices = [3, 801]
    checks: Dict[str, object] = {}
    all_passed = True
    for grid in JIGSAW_GRIDS:
        permutations = build_permutation_bank(sample_indices, grid=grid, seed=seed)
        repeated = build_permutation_bank(sample_indices, grid=grid, seed=seed)
        inverse = torch.argsort(permutations, dim=1)
        shuffled_image = apply_patch_permutation(image, permutations, grid=grid)
        shuffled_mask = apply_patch_permutation(mask, permutations, grid=grid)
        restored_image = apply_patch_permutation(shuffled_image, inverse, grid=grid)
        restored_mask = apply_patch_permutation(shuffled_mask, inverse, grid=grid)
        row = {
            "deterministic": bool(torch.equal(permutations, repeated)),
            "bijective": bool(
                torch.equal(
                    torch.sort(permutations, dim=1).values,
                    torch.arange(grid * grid).expand(2, -1),
                )
            ),
            "image_roundtrip_bit_exact": bool(torch.equal(image, restored_image)),
            "mask_roundtrip_bit_exact": bool(torch.equal(mask, restored_mask)),
            "non_identity_rows": int(
                (permutations != torch.arange(grid * grid)).any(dim=1).sum().item()
            ),
        }
        row["passed"] = bool(
            row["deterministic"]
            and row["bijective"]
            and row["image_roundtrip_bit_exact"]
            and row["mask_roundtrip_bit_exact"]
            and int(row["non_identity_rows"]) == 2
        )
        checks[str(grid)] = row
        all_passed = all_passed and bool(row["passed"])
    checks["all_passed"] = all_passed
    return checks


def stage_local_descriptor(tokens: Tensor, norm: nn.Module, *, prefix_count: int) -> Tensor:
    if tokens.ndim != 3:
        raise ValueError("Stage token tensor must have shape [B,N,D]")
    if int(tokens.size(1)) <= int(prefix_count):
        raise ValueError("Stage token tensor has no patch tokens")
    normalized = norm(tokens.float())
    patches = normalized[:, int(prefix_count) :]
    descriptor = 0.5 * patches.mean(dim=1) + 0.5 * patches.amax(dim=1)
    descriptor = F.normalize(descriptor, dim=1, eps=1e-6)
    if not bool(torch.isfinite(descriptor).all()):
        raise ValueError("Stage descriptor contains non-finite values")
    return descriptor


class FrozenStageCapture:
    def __init__(self, model: nn.Module, layers: Sequence[int] = STAGE_LAYERS) -> None:
        self.model = model
        self.layers = tuple(int(value) for value in layers)
        self.outputs: Dict[int, Tensor] = {}
        blocks = getattr(model, "blocks", None)
        if not isinstance(blocks, nn.ModuleList):
            raise TypeError("PMG A0 requires a TRKH model with a ModuleList named blocks")
        if max(self.layers) > len(blocks):
            raise ValueError("Requested PMG stage is outside the keeper block stack")
        self.handles = [
            blocks[layer - 1].register_forward_hook(self._hook(layer))
            for layer in self.layers
        ]

    def _hook(self, layer: int):
        def capture(_module, _inputs, output) -> None:
            value = output[0] if isinstance(output, tuple) else output
            if not torch.is_tensor(value):
                raise TypeError(f"Block {layer} did not return a token tensor")
            self.outputs[int(layer)] = value

        return capture

    def clear(self) -> None:
        self.outputs.clear()

    def descriptors(self) -> Tuple[Dict[int, Tensor], Dict[int, int]]:
        if set(self.outputs) != set(self.layers):
            raise RuntimeError(
                f"Captured PMG layers differ: {sorted(self.outputs)} != {list(self.layers)}"
            )
        norm = getattr(self.model, "norm", None)
        prefix_count = int(getattr(self.model, "num_prefix_tokens", -1))
        if not isinstance(norm, nn.Module) or prefix_count < 1:
            raise TypeError("Keeper norm/prefix layout is unavailable")
        descriptors = {
            layer: stage_local_descriptor(
                self.outputs[layer], norm, prefix_count=prefix_count
            )
            for layer in self.layers
        }
        patch_counts = {
            layer: int(self.outputs[layer].size(1) - prefix_count)
            for layer in self.layers
        }
        return descriptors, patch_counts

    def close(self) -> None:
        for handle in self.handles:
            handle.remove()
        self.handles.clear()

    def __enter__(self) -> "FrozenStageCapture":
        return self

    def __exit__(self, _exc_type, _exc, _traceback) -> None:
        self.close()


def assemble_role_descriptors(
    clean: Mapping[int, Tensor],
    jigsaw: Mapping[int, Mapping[int, Tensor]],
    clean_global: Tensor,
) -> Dict[str, Tensor]:
    required_layers = set(STAGE_LAYERS)
    if set(clean) != required_layers or set(jigsaw) != set(JIGSAW_GRIDS):
        raise ValueError("Incomplete PMG clean/jigsaw stage bank")
    for grid in JIGSAW_GRIDS:
        if set(jigsaw[grid]) != required_layers:
            raise ValueError(f"Incomplete PMG stage bank for grid {grid}")
    roles = {
        "clean_control": torch.cat(
            (clean[2], clean[5], clean[8], clean_global), dim=1
        ),
        "pmg_aligned": torch.cat(
            (jigsaw[8][2], jigsaw[4][5], jigsaw[2][8], clean_global), dim=1
        ),
        "reverse_placebo": torch.cat(
            (jigsaw[2][2], jigsaw[4][5], jigsaw[8][8], clean_global), dim=1
        ),
        "deepest_placebo": torch.cat(
            (jigsaw[8][8], jigsaw[4][8], jigsaw[2][8], clean_global), dim=1
        ),
    }
    batch = int(clean_global.size(0))
    for role, value in roles.items():
        if tuple(value.shape) != (batch, ROLE_DIM):
            raise ValueError(f"Role {role} has unexpected shape {tuple(value.shape)}")
        if not bool(torch.isfinite(value).all()):
            raise ValueError(f"Role {role} contains non-finite values")
    return roles


def _model_state_sha256(model: nn.Module) -> str:
    digest = hashlib.sha256()
    for name, tensor in sorted(model.state_dict().items()):
        value = tensor.detach().cpu().contiguous()
        digest.update(name.encode("utf-8"))
        digest.update(str(value.dtype).encode("ascii"))
        digest.update(np.asarray(value.shape, dtype=np.int64).tobytes())
        digest.update(value.reshape(-1).view(torch.uint8).numpy().tobytes())
    return digest.hexdigest()


def _read_cidt_clean(path: Path, *, num_classes: int) -> Dict[str, object]:
    rows: List[Mapping[str, str]] = []
    with Path(path).open("r", newline="", encoding="utf-8") as handle:
        for row in csv.DictReader(handle):
            if str(row.get("condition", "")) == "clean":
                rows.append(row)
    rows.sort(key=lambda row: int(row["sample_index"]))
    if len(rows) != EXPECTED_TRAIN_ROWS:
        raise ValueError(f"CIDT clean support mismatch: {len(rows)}")
    indices = np.asarray([int(row["sample_index"]) for row in rows], dtype=np.int64)
    if not np.array_equal(indices, np.arange(EXPECTED_TRAIN_ROWS, dtype=np.int64)):
        raise ValueError("CIDT clean sample indices are not complete and ordered")
    probabilities = np.asarray(
        [
            [float(row[f"keeper_prob_{class_index}"]) for class_index in range(num_classes)]
            for row in rows
        ],
        dtype=np.float64,
    )
    return {
        "indices": indices,
        "targets": np.asarray([int(row["target_index"]) for row in rows], dtype=np.int64),
        "predictions": np.asarray(
            [int(row["keeper_prediction"]) for row in rows], dtype=np.int64
        ),
        "probabilities": probabilities,
        "folds": np.asarray([int(row["fold"]) for row in rows], dtype=np.int64),
        "paths": [str(row["image_path"]) for row in rows],
        "source_stems": [str(row["source_stem"]) for row in rows],
    }


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


def _forward_features_with_capture(
    *,
    model: nn.Module,
    capture: FrozenStageCapture,
    images: Tensor,
    image_mask: Optional[Tensor],
    bbox: Optional[Tensor],
) -> Tuple[Dict[str, Tensor], Dict[int, Tensor], Dict[int, int]]:
    capture.clear()
    features = model.forward_features(  # type: ignore[attr-defined]
        images,
        image_valid_mask=image_mask,
        bbox_token_prior=bbox,
    )
    stage_descriptors, patch_counts = capture.descriptors()
    return features, stage_descriptors, patch_counts


def _extract_feature_bank(
    *,
    model: nn.Module,
    dataset: Dataset,
    labels: np.ndarray,
    device: torch.device,
    batch_size: int,
    num_workers: int,
    seed: int,
) -> Dict[str, object]:
    loader_kwargs, loader_summary = build_safe_dataloader_kwargs(
        requested_num_workers=int(num_workers),
        requested_pin_memory=device.type == "cuda",
        context="pmg_progressive_jigsaw_signal_a0",
        prefetch_factor=2,
        persistent_workers=True,
    )
    loader = DataLoader(
        dataset,
        batch_size=int(batch_size),
        shuffle=False,
        **loader_kwargs,
    )
    permutation_banks = {
        grid: build_permutation_bank(
            range(len(dataset)), grid=grid, seed=int(seed)
        )
        for grid in JIGSAW_GRIDS
    }
    visual_indices = {
        int(np.flatnonzero(labels == class_index)[0])
        for class_index in range(int(labels.max()) + 1)
    }
    visual_cache: Dict[int, Dict[str, Tensor]] = {}
    descriptor_chunks: Dict[str, List[Tensor]] = {role: [] for role in ROLE_NAMES}
    all_indices: List[Tensor] = []
    all_targets: List[Tensor] = []
    all_clean_probabilities: List[Tensor] = []
    patch_count_sets: Dict[str, Dict[str, set[int]]] = {
        "clean": {str(layer): set() for layer in STAGE_LAYERS},
        **{
            f"p{grid}": {str(layer): set() for layer in STAGE_LAYERS}
            for grid in JIGSAW_GRIDS
        },
    }
    model.eval()
    state_before = _model_state_sha256(model)
    processed = 0
    started = time.perf_counter()
    if device.type == "cuda":
        torch.cuda.empty_cache()
        torch.cuda.reset_peak_memory_stats(device)
        torch.cuda.synchronize(device)
    with FrozenStageCapture(model) as capture, torch.inference_mode():
        for images, targets, metadata in loader:
            images = images.to(device=device, dtype=torch.float32, non_blocking=True)
            targets = targets.to(dtype=torch.long)
            sample_indices = metadata["sample_index"].to(dtype=torch.long)
            sample_indices_cpu = sample_indices.detach().cpu()
            image_mask = _metadata_tensor(
                metadata, "image_mask", device=device, dtype=torch.bool
            )
            if image_mask is not None and image_mask.ndim == 3:
                image_mask = image_mask.unsqueeze(1)
            bbox = _metadata_tensor(metadata, "bbox", device=device, dtype=torch.float32)

            clean_features, clean_stages, clean_counts = _forward_features_with_capture(
                model=model,
                capture=capture,
                images=images,
                image_mask=image_mask,
                bbox=bbox,
            )
            if bbox is not None:
                clean_features["bbox"] = bbox
            clean_logits = classification_logits_from_features(model, clean_features)
            clean_probabilities = F.softmax(clean_logits.float(), dim=1)
            clean_global = F.normalize(
                clean_features["pooled"].float(), dim=1, eps=1e-6
            )
            for layer, count in clean_counts.items():
                patch_count_sets["clean"][str(layer)].add(int(count))

            jigsaw_stages: Dict[int, Dict[int, Tensor]] = {}
            jigsaw_images: Dict[int, Tensor] = {}
            for grid in JIGSAW_GRIDS:
                permutations = permutation_banks[grid].index_select(
                    0, sample_indices_cpu
                ).to(device=device, non_blocking=True)
                transformed_images = apply_patch_permutation(
                    images, permutations, grid=grid
                )
                transformed_mask = (
                    apply_patch_permutation(image_mask, permutations, grid=grid)
                    if image_mask is not None
                    else None
                )
                _, stage_values, stage_counts = _forward_features_with_capture(
                    model=model,
                    capture=capture,
                    images=transformed_images,
                    image_mask=transformed_mask,
                    bbox=None,
                )
                jigsaw_stages[grid] = stage_values
                jigsaw_images[grid] = transformed_images
                for layer, count in stage_counts.items():
                    patch_count_sets[f"p{grid}"][str(layer)].add(int(count))

            roles = assemble_role_descriptors(
                clean_stages, jigsaw_stages, clean_global
            )
            for role, value in roles.items():
                descriptor_chunks[role].append(value.detach().cpu())
            all_indices.append(sample_indices_cpu)
            all_targets.append(targets.detach().cpu())
            all_clean_probabilities.append(clean_probabilities.detach().cpu())

            for local_index, sample_index in enumerate(sample_indices_cpu.tolist()):
                if int(sample_index) in visual_indices:
                    visual_cache[int(sample_index)] = {
                        "clean": images[local_index].detach().cpu(),
                        **{
                            f"p{grid}": jigsaw_images[grid][local_index]
                            .detach()
                            .cpu()
                            for grid in JIGSAW_GRIDS
                        },
                    }

            processed += int(targets.numel())
            if processed % 1024 < int(targets.numel()) or processed == len(dataset):
                print(
                    json.dumps(
                        {
                            "stage": "pmg_feature_extraction",
                            "processed": processed,
                            "rows": len(dataset),
                            "elapsed_seconds": time.perf_counter() - started,
                        }
                    ),
                    flush=True,
                )
    if device.type == "cuda":
        torch.cuda.synchronize(device)
    elapsed = float(time.perf_counter() - started)
    state_after = _model_state_sha256(model)
    descriptors = {
        role: torch.cat(chunks, dim=0).numpy().astype(np.float32, copy=False)
        for role, chunks in descriptor_chunks.items()
    }
    return {
        "indices": torch.cat(all_indices).numpy(),
        "targets": torch.cat(all_targets).numpy(),
        "clean_probabilities": torch.cat(all_clean_probabilities).numpy(),
        "descriptors": descriptors,
        "descriptor_sha256": {
            role: _array_sha256(value) for role, value in descriptors.items()
        },
        "visual_cache": visual_cache,
        "loader": loader_summary,
        "patch_counts": {
            view: {
                layer: sorted(int(value) for value in values)
                for layer, values in layers.items()
            }
            for view, layers in patch_count_sets.items()
        },
        "state_before_sha256": state_before,
        "state_after_sha256": state_after,
        "elapsed_seconds": elapsed,
        "throughput_source_images_per_second": float(len(dataset) / max(elapsed, 1e-12)),
        "throughput_forward_views_per_second": float(
            len(dataset) * 4 / max(elapsed, 1e-12)
        ),
        "peak_cuda_bytes": int(
            torch.cuda.max_memory_allocated(device) if device.type == "cuda" else 0
        ),
    }


def fit_oof_readout(
    descriptors: np.ndarray,
    targets: np.ndarray,
    folds: np.ndarray,
    *,
    seed: int,
) -> Tuple[np.ndarray, Dict[str, object]]:
    x = np.asarray(descriptors, dtype=np.float64)
    y = np.asarray(targets, dtype=np.int64).reshape(-1)
    fold_values = np.asarray(folds, dtype=np.int64).reshape(-1)
    if x.ndim != 2 or x.shape != (y.size, ROLE_DIM) or y.shape != fold_values.shape:
        raise ValueError("OOF descriptor/target/fold shape mismatch")
    class_count = int(y.max()) + 1
    expected_classes = np.arange(class_count, dtype=np.int64)
    probabilities = np.full((y.size, class_count), np.nan, dtype=np.float64)
    prediction_count = np.zeros(y.size, dtype=np.int64)
    fold_rows: List[Dict[str, object]] = []
    started = time.perf_counter()
    for fold in FOLDS:
        fit_mask = fold_values != int(fold)
        holdout_mask = fold_values == int(fold)
        scaler = StandardScaler(copy=True, with_mean=True, with_std=True)
        fit_x = scaler.fit_transform(x[fit_mask])
        holdout_x = scaler.transform(x[holdout_mask])
        classifier = LogisticRegression(
            C=READOUT_C,
            penalty="l2",
            solver="lbfgs",
            tol=READOUT_TOL,
            max_iter=READOUT_MAX_ITER,
            class_weight=None,
            random_state=int(seed),
        )
        with warnings.catch_warnings(record=True) as caught:
            warnings.simplefilter("always", ConvergenceWarning)
            classifier.fit(fit_x, y[fit_mask])
        convergence_warnings = [
            str(item.message)
            for item in caught
            if issubclass(item.category, ConvergenceWarning)
        ]
        if not np.array_equal(classifier.classes_, expected_classes):
            raise ValueError(f"Fold {fold} readout class order differs")
        probabilities[holdout_mask] = classifier.predict_proba(holdout_x)
        prediction_count[holdout_mask] += 1
        max_iterations = int(np.max(classifier.n_iter_))
        fold_rows.append(
            {
                "fold": int(fold),
                "fit_rows": int(fit_mask.sum()),
                "holdout_rows": int(holdout_mask.sum()),
                "n_iter_max": max_iterations,
                "convergence_warning_count": len(convergence_warnings),
                "converged": len(convergence_warnings) == 0
                and max_iterations < READOUT_MAX_ITER,
                "coefficient_l2": float(np.linalg.norm(classifier.coef_)),
                "intercept_l2": float(np.linalg.norm(classifier.intercept_)),
                "scaler_active_dimensions": int((scaler.scale_ > 1e-12).sum()),
            }
        )
    if not np.all(prediction_count == 1):
        raise RuntimeError("Every OOF row must be predicted exactly once")
    if not np.isfinite(probabilities).all():
        raise RuntimeError("OOF readout emitted non-finite probabilities")
    if not np.allclose(probabilities.sum(axis=1), 1.0, atol=1e-10, rtol=0.0):
        raise RuntimeError("OOF readout probabilities are not normalized")
    return probabilities, {
        "C": READOUT_C,
        "tol": READOUT_TOL,
        "max_iter": READOUT_MAX_ITER,
        "solver": "lbfgs",
        "penalty": "l2",
        "class_weight": None,
        "natural_prior": True,
        "elapsed_seconds": float(time.perf_counter() - started),
        "folds": fold_rows,
        "all_converged": all(bool(row["converged"]) for row in fold_rows),
        "every_row_predicted_once": bool(np.all(prediction_count == 1)),
    }


def _ece(targets: np.ndarray, probabilities: np.ndarray, bins: int = 15) -> float:
    target = np.asarray(targets, dtype=np.int64).reshape(-1)
    probs = np.asarray(probabilities, dtype=np.float64)
    predictions = probs.argmax(axis=1)
    confidence = probs.max(axis=1)
    correctness = predictions == target
    result = 0.0
    edges = np.linspace(0.0, 1.0, int(bins) + 1)
    for index in range(int(bins)):
        selected = (confidence > edges[index]) & (confidence <= edges[index + 1])
        if index == 0:
            selected |= confidence == 0.0
        if not bool(selected.any()):
            continue
        result += float(selected.mean()) * abs(
            float(correctness[selected].mean()) - float(confidence[selected].mean())
        )
    return float(result)


def classification_metrics(
    targets: np.ndarray, probabilities: np.ndarray
) -> Dict[str, object]:
    probs = np.asarray(probabilities, dtype=np.float64)
    predictions = probs.argmax(axis=1)
    metrics = _classification_metrics(
        np.asarray(targets, dtype=np.int64), predictions, num_classes=probs.shape[1]
    )
    metrics["ece_15"] = _ece(targets, probs, bins=15)
    metrics["mean_confidence"] = float(probs.max(axis=1).mean())
    return metrics


def _safe_auc(labels: np.ndarray, scores: np.ndarray) -> Optional[float]:
    label = np.asarray(labels, dtype=np.int64).reshape(-1)
    score = np.asarray(scores, dtype=np.float64).reshape(-1)
    if label.size == 0 or np.unique(label).size != 2:
        return None
    return float(roc_auc_score(label, score))


def _transition_summary(
    targets: np.ndarray,
    baseline_probabilities: np.ndarray,
    candidate_probabilities: np.ndarray,
) -> Dict[str, int]:
    target = np.asarray(targets, dtype=np.int64)
    baseline = np.asarray(baseline_probabilities).argmax(axis=1)
    candidate = np.asarray(candidate_probabilities).argmax(axis=1)
    restricted = np.isin(target, RESTRICTED_NEGATIVE_CLASSES)
    return {
        "changed": int((baseline != candidate).sum()),
        "corrections": int(((baseline != target) & (candidate == target)).sum()),
        "harms": int(((baseline == target) & (candidate != target)).sum()),
        "focus_fn_rescue": int(
            ((target == FOCUS_CLASS) & (baseline != FOCUS_CLASS) & (candidate == FOCUS_CLASS)).sum()
        ),
        "focus_tp_break": int(
            ((target == FOCUS_CLASS) & (baseline == FOCUS_CLASS) & (candidate != FOCUS_CLASS)).sum()
        ),
        "restricted_fp_remove": int(
            (restricted & (baseline == FOCUS_CLASS) & (candidate != FOCUS_CLASS)).sum()
        ),
        "restricted_fp_create": int(
            (restricted & (baseline != FOCUS_CLASS) & (candidate == FOCUS_CLASS)).sum()
        ),
        "restricted_fp_net_removal": int(
            (restricted & (baseline == FOCUS_CLASS) & (candidate != FOCUS_CLASS)).sum()
            - (restricted & (baseline != FOCUS_CLASS) & (candidate == FOCUS_CLASS)).sum()
        ),
    }


def descriptor_statistics(descriptors: np.ndarray) -> Dict[str, object]:
    value = np.asarray(descriptors, dtype=np.float64)
    if value.ndim != 2 or value.shape[1] != ROLE_DIM:
        raise ValueError("Descriptor statistics require [N,1024]")
    positions = np.linspace(0, value.shape[0] - 1, num=min(512, value.shape[0]))
    positions = np.unique(np.rint(positions).astype(np.int64))
    sketch = value[positions]
    sketch = sketch - sketch.mean(axis=0, keepdims=True)
    singular = np.linalg.svd(sketch, full_matrices=False, compute_uv=False)
    energy = np.square(singular)
    probability = energy / max(float(energy.sum()), 1e-30)
    nonzero = probability > 0.0
    effective_rank = float(
        np.exp(-np.sum(probability[nonzero] * np.log(probability[nonzero])))
    )
    blocks: List[Dict[str, object]] = []
    for block_index in range(4):
        block = value[:, block_index * FEATURE_DIM : (block_index + 1) * FEATURE_DIM]
        standard_deviation = block.std(axis=0)
        blocks.append(
            {
                "block": int(block_index),
                "active_dimensions": int((standard_deviation > 1e-8).sum()),
                "minimum_std": float(standard_deviation.min()),
                "maximum_std": float(standard_deviation.max()),
                "constant": bool(np.all(standard_deviation <= 1e-8)),
            }
        )
    return {
        "shape": [int(value.shape[0]), int(value.shape[1])],
        "sha256": _array_sha256(value.astype(np.float32)),
        "effective_rank_512_row_sketch": effective_rank,
        "sketch_rows": int(positions.size),
        "blocks": blocks,
        "all_values_finite": bool(np.isfinite(value).all()),
    }


def build_analysis(
    *,
    targets: np.ndarray,
    folds: np.ndarray,
    keeper_probabilities: np.ndarray,
    role_probabilities: Mapping[str, np.ndarray],
    descriptor_stats: Mapping[str, Mapping[str, object]],
) -> Dict[str, object]:
    target = np.asarray(targets, dtype=np.int64).reshape(-1)
    fold_values = np.asarray(folds, dtype=np.int64).reshape(-1)
    keeper_probs = np.asarray(keeper_probabilities, dtype=np.float64)
    if set(role_probabilities) != set(ROLE_NAMES):
        raise ValueError("Role probability bank differs from locked PMG roles")
    metrics = {
        role: classification_metrics(target, probabilities)
        for role, probabilities in role_probabilities.items()
    }
    control = np.asarray(role_probabilities["clean_control"], dtype=np.float64)
    candidate = np.asarray(role_probabilities["pmg_aligned"], dtype=np.float64)
    transition = _transition_summary(target, control, candidate)
    deltas = {
        "macro_f1": float(metrics["pmg_aligned"]["macro_f1"])
        - float(metrics["clean_control"]["macro_f1"]),
        "class1_f1": float(metrics["pmg_aligned"]["per_class_f1"][FOCUS_CLASS])
        - float(metrics["clean_control"]["per_class_f1"][FOCUS_CLASS]),
        "class1_precision": float(
            metrics["pmg_aligned"]["per_class_precision"][FOCUS_CLASS]
        )
        - float(metrics["clean_control"]["per_class_precision"][FOCUS_CLASS]),
        "class1_recall": float(
            metrics["pmg_aligned"]["per_class_recall"][FOCUS_CLASS]
        )
        - float(metrics["clean_control"]["per_class_recall"][FOCUS_CLASS]),
    }
    fold_rows: List[Dict[str, object]] = []
    for fold in FOLDS:
        selected = fold_values == int(fold)
        control_metrics = classification_metrics(target[selected], control[selected])
        candidate_metrics = classification_metrics(target[selected], candidate[selected])
        fold_transition = _transition_summary(
            target[selected], control[selected], candidate[selected]
        )
        fold_rows.append(
            {
                "fold": int(fold),
                "rows": int(selected.sum()),
                "control_macro_f1": float(control_metrics["macro_f1"]),
                "candidate_macro_f1": float(candidate_metrics["macro_f1"]),
                "macro_f1_delta": float(candidate_metrics["macro_f1"])
                - float(control_metrics["macro_f1"]),
                "class1_f1_delta": float(
                    candidate_metrics["per_class_f1"][FOCUS_CLASS]
                )
                - float(control_metrics["per_class_f1"][FOCUS_CLASS]),
                "class1_precision_delta": float(
                    candidate_metrics["per_class_precision"][FOCUS_CLASS]
                )
                - float(control_metrics["per_class_precision"][FOCUS_CLASS]),
                "class1_recall_delta": float(
                    candidate_metrics["per_class_recall"][FOCUS_CLASS]
                )
                - float(control_metrics["per_class_recall"][FOCUS_CLASS]),
                **fold_transition,
            }
        )

    keeper_prediction = keeper_probs.argmax(axis=1)
    keeper_tp = (target == FOCUS_CLASS) & (keeper_prediction == FOCUS_CLASS)
    keeper_fp = (
        np.isin(target, RESTRICTED_NEGATIVE_CLASSES)
        & (keeper_prediction == FOCUS_CLASS)
    )
    direction_selected = keeper_tp | keeper_fp
    direction_labels = keeper_tp[direction_selected].astype(np.int64)
    directional: Dict[str, object] = {
        "class1_tp_support": int(keeper_tp.sum()),
        "restricted_fp_support": int(keeper_fp.sum()),
    }
    for role in ("pmg_aligned", "reverse_placebo", "deepest_placebo"):
        delta_p1 = (
            np.asarray(role_probabilities[role], dtype=np.float64)[:, FOCUS_CLASS]
            - control[:, FOCUS_CLASS]
        )
        directional[role] = {
            "auroc_tp_vs_restricted_fp": _safe_auc(
                direction_labels, delta_p1[direction_selected]
            ),
            "class1_tp_delta_mean": float(delta_p1[keeper_tp].mean()),
            "restricted_fp_delta_mean": float(delta_p1[keeper_fp].mean()),
        }

    candidate_f1 = float(metrics["pmg_aligned"]["per_class_f1"][FOCUS_CLASS])
    reverse_f1 = float(metrics["reverse_placebo"]["per_class_f1"][FOCUS_CLASS])
    deepest_f1 = float(metrics["deepest_placebo"]["per_class_f1"][FOCUS_CLASS])
    candidate_auc = float(
        directional["pmg_aligned"]["auroc_tp_vs_restricted_fp"]
    )
    placebo_auc = max(
        float(directional["reverse_placebo"]["auroc_tp_vs_restricted_fp"]),
        float(directional["deepest_placebo"]["auroc_tp_vs_restricted_fp"]),
    )
    candidate_descriptor = descriptor_stats["pmg_aligned"]
    mechanism_gates = {
        "macro_f1_delta_ge_0003": deltas["macro_f1"] >= 0.003,
        "class1_f1_delta_ge_0010": deltas["class1_f1"] >= 0.010,
        "class1_precision_delta_ge_0010": deltas["class1_precision"] >= 0.010,
        "class1_recall_delta_ge_minus0005": deltas["class1_recall"] >= -0.005,
        "corrections_exceed_harms": transition["corrections"] > transition["harms"],
        "restricted_fp_net_removal_ge_10": transition[
            "restricted_fp_net_removal"
        ]
        >= 10,
        "tp_break_le_fn_rescue_plus_1": transition["focus_tp_break"]
        <= transition["focus_fn_rescue"] + 1,
        "precision_nonworse_folds_ge_4": sum(
            float(row["class1_precision_delta"]) >= 0.0 for row in fold_rows
        )
        >= 4,
        "positive_class1_f1_folds_ge_3": sum(
            float(row["class1_f1_delta"]) > 0.0 for row in fold_rows
        )
        >= 3,
        "worst_fold_class1_recall_delta_ge_minus003": min(
            float(row["class1_recall_delta"]) for row in fold_rows
        )
        >= -0.03,
        "class1_f1_beats_reverse_by_0005": candidate_f1 - reverse_f1 >= 0.005,
        "class1_f1_beats_deepest_by_0005": candidate_f1 - deepest_f1 >= 0.005,
        "direction_auroc_ge_060": candidate_auc >= 0.60,
        "direction_auroc_beats_placebos_by_003": candidate_auc - placebo_auc >= 0.03,
        "candidate_effective_rank_ge_16": float(
            candidate_descriptor["effective_rank_512_row_sketch"]
        )
        >= 16.0,
        "candidate_no_constant_block": not any(
            bool(row["constant"]) for row in candidate_descriptor["blocks"]
        ),
    }
    return {
        "role_metrics": metrics,
        "candidate_vs_control_deltas": deltas,
        "candidate_vs_control_transitions": transition,
        "folds": fold_rows,
        "directional": directional,
        "placebo_comparisons": {
            "candidate_minus_reverse_class1_f1": candidate_f1 - reverse_f1,
            "candidate_minus_deepest_class1_f1": candidate_f1 - deepest_f1,
            "candidate_minus_best_placebo_direction_auroc": candidate_auc
            - placebo_auc,
        },
        "mechanism_gates": mechanism_gates,
        "mechanism_gates_passed": all(mechanism_gates.values()),
    }


def _tensor_to_pil(tensor: Tensor, semantics: Mapping[str, object]) -> Image.Image:
    mean = torch.tensor(semantics["input_mean"], dtype=torch.float32).view(3, 1, 1)
    std = torch.tensor(semantics["input_std"], dtype=torch.float32).view(3, 1, 1)
    rgb = (tensor.detach().float().cpu() * std + mean).clamp(0.0, 1.0)
    array = (rgb.permute(1, 2, 0).numpy() * 255.0).round().astype(np.uint8)
    return Image.fromarray(array, mode="RGB")


def _render_contact_sheet(
    path: Path,
    visual_cache: Mapping[int, Mapping[str, Tensor]],
    labels: np.ndarray,
    class_names: Sequence[str],
    semantics: Mapping[str, object],
) -> Dict[str, object]:
    selected = sorted(visual_cache)
    if len(selected) != len(class_names):
        raise ValueError("PMG contact sheet requires one fixed row per class")
    tile = int(semantics["image_size"])
    left = 150
    top = 30
    columns = ("clean", "p8", "p4", "p2")
    canvas = Image.new("RGB", (left + tile * len(columns), top + tile * len(selected)), "white")
    draw = ImageDraw.Draw(canvas)
    font = ImageFont.load_default()
    for column_index, column in enumerate(columns):
        draw.text((left + column_index * tile + 8, 8), column, fill="black", font=font)
    rows: List[Dict[str, object]] = []
    for row_index, sample_index in enumerate(selected):
        target = int(labels[sample_index])
        draw.text(
            (6, top + row_index * tile + 8),
            f"sample={sample_index}\nclass={target}:{class_names[target]}",
            fill="black",
            font=font,
        )
        for column_index, column in enumerate(columns):
            image = _tensor_to_pil(visual_cache[sample_index][column], semantics)
            canvas.paste(image, (left + column_index * tile, top + row_index * tile))
        rows.append({"sample_index": int(sample_index), "target": target})
    canvas.save(path)
    return {
        "path": path.name,
        "rows": rows,
        "columns": list(columns),
        "width": int(canvas.width),
        "height": int(canvas.height),
        "manual_review_required": True,
    }


def _prediction_rows(
    *,
    targets: np.ndarray,
    folds: np.ndarray,
    paths: Sequence[Path],
    source_stems: Sequence[str],
    keeper_probabilities: np.ndarray,
    role_probabilities: Mapping[str, np.ndarray],
    class_names: Sequence[str],
) -> List[Dict[str, object]]:
    control_prediction = np.asarray(role_probabilities["clean_control"]).argmax(axis=1)
    candidate_prediction = np.asarray(role_probabilities["pmg_aligned"]).argmax(axis=1)
    keeper_prediction = np.asarray(keeper_probabilities).argmax(axis=1)
    rows: List[Dict[str, object]] = []
    for index in range(len(targets)):
        target = int(targets[index])
        restricted = target in RESTRICTED_NEGATIVE_CLASSES
        row: Dict[str, object] = {
            "sample_index": int(index),
            "image_path": str(Path(paths[index]).resolve()),
            "source_stem": str(source_stems[index]),
            "fold": int(folds[index]),
            "target_index": target,
            "target_name": str(class_names[target]),
            "keeper_prediction": int(keeper_prediction[index]),
            "control_prediction": int(control_prediction[index]),
            "candidate_prediction": int(candidate_prediction[index]),
            "candidate_correction": bool(
                control_prediction[index] != target and candidate_prediction[index] == target
            ),
            "candidate_harm": bool(
                control_prediction[index] == target and candidate_prediction[index] != target
            ),
            "focus_fn_rescue": bool(
                target == FOCUS_CLASS
                and control_prediction[index] != FOCUS_CLASS
                and candidate_prediction[index] == FOCUS_CLASS
            ),
            "focus_tp_break": bool(
                target == FOCUS_CLASS
                and control_prediction[index] == FOCUS_CLASS
                and candidate_prediction[index] != FOCUS_CLASS
            ),
            "restricted_fp_remove": bool(
                restricted
                and control_prediction[index] == FOCUS_CLASS
                and candidate_prediction[index] != FOCUS_CLASS
            ),
            "restricted_fp_create": bool(
                restricted
                and control_prediction[index] != FOCUS_CLASS
                and candidate_prediction[index] == FOCUS_CLASS
            ),
        }
        for class_index in range(len(class_names)):
            row[f"keeper_prob_{class_index}"] = float(
                keeper_probabilities[index, class_index]
            )
            for role in ROLE_NAMES:
                row[f"{role}_prob_{class_index}"] = float(
                    role_probabilities[role][index, class_index]
                )
        rows.append(row)
    return rows


def _artifact_record(path: Path, *, root: Path) -> Dict[str, object]:
    resolved = Path(path).resolve()
    return {
        "path": resolved.relative_to(Path(root).resolve()).as_posix(),
        "bytes": int(resolved.stat().st_size),
        "sha256": _sha256(resolved),
    }


def _write_manifest(output_dir: Path) -> Path:
    manifest_path = Path(output_dir) / "artifact_manifest.json"
    payloads = [
        _artifact_record(path, root=output_dir)
        for path in sorted(Path(output_dir).rglob("*"))
        if path.is_file() and path != manifest_path
    ]
    _write_json(
        manifest_path,
        {
            "mode": "pmg_progressive_jigsaw_signal_a0_artifact_manifest",
            "payload_count": len(payloads),
            "payload_bytes": int(sum(int(row["bytes"]) for row in payloads)),
            "payloads": payloads,
        },
    )
    return manifest_path


def _verify_manifest(output_dir: Path) -> Dict[str, object]:
    manifest_path = Path(output_dir) / "artifact_manifest.json"
    payload = json.loads(manifest_path.read_text(encoding="utf-8"))
    rows = payload.get("payloads")
    if not isinstance(rows, list):
        raise ValueError("PMG artifact manifest has no payload list")
    for row in rows:
        path = Path(output_dir) / str(row["path"])
        if not path.is_file():
            raise FileNotFoundError(f"PMG manifest payload is missing: {path}")
        if int(path.stat().st_size) != int(row["bytes"]) or _sha256(path) != row["sha256"]:
            raise ValueError(f"PMG manifest payload differs: {path}")
    listed = {str(row["path"]) for row in rows}
    observed = {
        path.relative_to(output_dir).as_posix()
        for path in Path(output_dir).rglob("*")
        if path.is_file() and path != manifest_path
    }
    if listed != observed:
        raise ValueError("PMG manifest file set differs from output directory")
    return {
        "payload_count": len(rows),
        "payload_bytes": int(sum(int(row["bytes"]) for row in rows)),
        "manifest_sha256": _sha256(manifest_path),
    }


def _recursive_numeric_difference(left: object, right: object) -> float:
    if isinstance(left, Mapping) and isinstance(right, Mapping):
        if set(left) != set(right):
            return float("inf")
        return max(
            (_recursive_numeric_difference(left[key], right[key]) for key in left),
            default=0.0,
        )
    if isinstance(left, list) and isinstance(right, list):
        if len(left) != len(right):
            return float("inf")
        return max(
            (_recursive_numeric_difference(a, b) for a, b in zip(left, right)),
            default=0.0,
        )
    if isinstance(left, (int, float, bool)) and isinstance(right, (int, float, bool)):
        return abs(float(left) - float(right))
    return 0.0 if left == right else float("inf")


def _load_probability_cache(path: Path) -> Dict[str, object]:
    with np.load(path, allow_pickle=False) as cache:
        role_probabilities = {
            role: cache[f"probabilities_{role}"].astype(np.float64)
            for role in ROLE_NAMES
        }
        return {
            "targets": cache["targets"].astype(np.int64),
            "folds": cache["folds"].astype(np.int64),
            "keeper_probabilities": cache["keeper_probabilities"].astype(np.float64),
            "role_probabilities": role_probabilities,
        }


def replay_summary(summary_path: Path) -> Dict[str, object]:
    resolved = Path(summary_path).expanduser().resolve()
    summary = json.loads(resolved.read_text(encoding="utf-8"))
    output_dir = resolved.parent
    manifest = _verify_manifest(output_dir)
    cache = _load_probability_cache(output_dir / "oof_probabilities.npz")
    descriptor_stats = json.loads(
        (output_dir / "descriptor_statistics.json").read_text(encoding="utf-8")
    )
    replayed = build_analysis(
        targets=cache["targets"],
        folds=cache["folds"],
        keeper_probabilities=cache["keeper_probabilities"],
        role_probabilities=cache["role_probabilities"],
        descriptor_stats=descriptor_stats,
    )
    maximum_difference = _recursive_numeric_difference(summary["analysis"], replayed)
    result = {
        "mode": "pmg_progressive_jigsaw_signal_a0_replay",
        "rows": int(cache["targets"].size),
        "maximum_numeric_difference": float(maximum_difference),
        "replay_passed": bool(maximum_difference <= 1e-12),
        "mechanism_gates_passed": bool(replayed["mechanism_gates_passed"]),
        **manifest,
    }
    if not bool(result["replay_passed"]):
        raise ValueError(f"PMG A0 replay failed: {result}")
    return result


def finalize_visual_review(
    summary_path: Path,
    *,
    result: str,
    expected_summary_sha256: str,
) -> Dict[str, object]:
    resolved = Path(summary_path).expanduser().resolve()
    if resolved.name != "summary.json" or not resolved.is_file():
        raise FileNotFoundError(f"PMG A0 summary is missing: {resolved}")
    expected = str(expected_summary_sha256).strip().casefold()
    if len(expected) != 64 or any(
        character not in "0123456789abcdef" for character in expected
    ):
        raise ValueError("Expected summary SHA-256 must contain exactly 64 hex characters")
    observed = _sha256(resolved)
    if observed != expected:
        raise ValueError(
            "PMG visual review summary hash differs: "
            f"expected={expected}, observed={observed}"
        )
    if result not in {"pass", "fail"}:
        raise ValueError(f"Visual review result must be pass or fail; got {result!r}")

    output_dir = resolved.parent
    manifest_before = _verify_manifest(output_dir)
    summary = json.loads(resolved.read_text(encoding="utf-8"))
    if summary.get("mode") != "pmg_progressive_jigsaw_signal_a0_train_information_gate":
        raise ValueError("Visual review summary mode differs from PMG A0")
    review = summary.get("visual_review")
    if not isinstance(review, Mapping) or not bool(review.get("required")):
        raise ValueError("PMG A0 summary does not require a visual review")
    if bool(review.get("completed")):
        raise ValueError("PMG A0 visual review was already finalized")

    artifacts = summary.get("artifacts")
    if not isinstance(artifacts, Mapping):
        raise ValueError("PMG A0 summary artifacts are missing")
    contact = artifacts.get("jigsaw_contact_sheet")
    if not isinstance(contact, Mapping):
        raise ValueError("PMG A0 contact-sheet artifact is missing")
    contact_path = output_dir / str(contact.get("path", ""))
    contact_sha256 = _sha256(contact_path)
    if contact_sha256 != str(contact.get("sha256", "")):
        raise ValueError("PMG A0 contact sheet differs from its locked summary record")

    geometry_passed = result == "pass"
    automated_passed = bool(summary.get("automated_gate_passed"))
    all_gates_passed = automated_passed and geometry_passed
    summary["visual_review"] = {
        "required": True,
        "completed": True,
        "geometry_passed": geometry_passed,
        "decision": result,
        "reviewed_summary_sha256": observed,
        "reviewed_contact_sheet_sha256": contact_sha256,
        "manifest_sha256_before_review": manifest_before["manifest_sha256"],
    }
    summary["status"] = (
        "passed_all_a0_gates"
        if all_gates_passed
        else (
            "rejected_visual_gate"
            if automated_passed and not geometry_passed
            else "rejected_automated_gate"
        )
    )
    summary["all_a0_gates_passed"] = all_gates_passed
    summary["trainer_integration_authorized"] = all_gates_passed
    summary["short_pair_authorized"] = all_gates_passed
    summary["full_train_authorized"] = False
    summary["current_command_update_authorized"] = False
    _write_json(resolved, summary)
    manifest_path = _write_manifest(output_dir)
    return {
        "mode": "pmg_progressive_jigsaw_signal_a0_visual_review",
        "decision": result,
        "geometry_passed": geometry_passed,
        "automated_gate_passed": automated_passed,
        "all_a0_gates_passed": all_gates_passed,
        "status": summary["status"],
        "reviewed_summary_sha256": observed,
        "summary_sha256": _sha256(resolved),
        "manifest_sha256": _sha256(manifest_path),
        "short_pair_authorized": all_gates_passed,
        "full_train_authorized": False,
        "current_command_update_authorized": False,
    }


def _validate_locked_args(args: argparse.Namespace) -> None:
    if int(args.batch_size) != 64 or int(args.num_workers) != 4:
        raise ValueError("The PMG A0 protocol locks batch-size=64 and num-workers=4")
    if int(args.folds) != 5 or int(args.seed) != 20260714:
        raise ValueError("The PMG A0 protocol locks folds=5 and seed=20260714")


def run_audit(args: argparse.Namespace) -> Dict[str, object]:
    _validate_locked_args(args)
    provenance = verify_locked_inputs()
    repo_state = _repo_state()
    if not bool(repo_state["tracked_worktree_clean"]) or not bool(
        repo_state["head_matches_upstream"]
    ):
        raise ValueError(f"Formal PMG A0 requires clean pushed tracked state: {repo_state}")
    output_dir = _prepare_output_dir(args.output_dir)
    device_name = str(args.device)
    if device_name == "cuda" and not torch.cuda.is_available():
        raise RuntimeError("CUDA was requested but is unavailable")
    device = torch.device(
        "cuda"
        if device_name == "cuda" or (device_name == "auto" and torch.cuda.is_available())
        else "cpu"
    )
    set_seed(int(args.seed), deterministic=True)
    torch.backends.cuda.matmul.allow_tf32 = False
    if device.type == "cuda":
        torch.backends.cudnn.allow_tf32 = True

    files = provenance["files"]
    model, checkpoint, class_names = load_model(Path(files["keeper"]["path"]), device)
    semantics = _eval_semantics(checkpoint)
    if int(semantics["temporal_frames"]) != 1:
        raise ValueError("PMG A0 supports temporal_frames=1 only")
    model_config = checkpoint.get("model_config", {})
    if not isinstance(model_config, Mapping):
        raise ValueError("Keeper model_config is missing")
    if int(model_config.get("embed_dim", -1)) != FEATURE_DIM:
        raise ValueError("PMG A0 locks keeper embed_dim=256")
    if bool(model_config.get("deep_class_prompt", False)):
        raise ValueError("PMG A0 excludes dynamic deep prompt prefixes")

    data_path = Path(files["data_yaml"]["path"])
    data_spec = load_data_spec(
        data_path, class_name_mode="raw", expected_num_classes=len(class_names)
    )
    if list(data_spec.class_names) != list(class_names):
        raise ValueError("Dataset and keeper class order differ")
    base_dataset = MangoYOLOCropDataset.from_data_spec(
        data_spec=data_spec,
        split="train",
        transform=None,
        crop_margin_ratio=float(semantics["crop_margin_ratio"]),
        crop_to_primary_object=resolve_crop_to_primary_object(checkpoint),
        classification_target=True,
        classification_object_crops=True,
        classification_bbox_metadata=True,
    )
    if len(base_dataset) != EXPECTED_TRAIN_ROWS:
        raise ValueError(f"Full train support mismatch: {len(base_dataset)}")
    labels = np.asarray(base_dataset.labels(), dtype=np.int64)
    paths = [Path(path).resolve() for path in base_dataset.sample_paths()]
    source_stems = np.asarray([path.stem.casefold() for path in paths], dtype=str)
    folds, fold_assignment = _assign_source_folds(
        labels, source_stems, folds=int(args.folds), seed=int(args.seed)
    )
    dataset_identity = _dataset_identity(base_dataset, range(len(base_dataset)))
    train_sources = set(source_stems.tolist())
    val_sources = _split_source_stems(data_spec.val_images)
    test_sources = (
        _split_source_stems(data_spec.test_images)
        if data_spec.test_images is not None
        else set()
    )
    train_val_overlap = sorted(train_sources.intersection(val_sources))
    train_test_overlap = sorted(train_sources.intersection(test_sources))

    transform = _build_eval_transform(semantics)
    inference_dataset = _SelectedConditionDataset(
        base_dataset,
        list(range(len(base_dataset))),
        corruption=IdentityCorruption(),
        transform=transform,
    )
    extraction = _extract_feature_bank(
        model=model,
        dataset=inference_dataset,
        labels=labels,
        device=device,
        batch_size=int(args.batch_size),
        num_workers=int(args.num_workers),
        seed=int(args.seed),
    )
    indices = np.asarray(extraction["indices"], dtype=np.int64)
    targets = np.asarray(extraction["targets"], dtype=np.int64)
    clean_probabilities = np.asarray(extraction["clean_probabilities"], dtype=np.float64)
    descriptors = extraction["descriptors"]

    cidt = _read_cidt_clean(
        Path(files["cidt_predictions"]["path"]), num_classes=len(class_names)
    )
    cidt_probability_error = float(
        np.max(np.abs(clean_probabilities - cidt["probabilities"]))
    )
    cidt_checks = {
        "sample_indices_exact": bool(np.array_equal(indices, cidt["indices"])),
        "targets_exact": bool(np.array_equal(targets, cidt["targets"])),
        "predictions_exact": bool(
            np.array_equal(clean_probabilities.argmax(axis=1), cidt["predictions"])
        ),
        "folds_exact": bool(np.array_equal(folds, cidt["folds"])),
        "paths_exact": [str(path) for path in paths] == list(cidt["paths"]),
        "source_stems_exact": source_stems.tolist() == list(cidt["source_stems"]),
        "probability_max_abs_le_2e_5": cidt_probability_error <= 2e-5,
    }

    descriptor_stats = {
        role: descriptor_statistics(descriptors[role]) for role in ROLE_NAMES
    }
    readout_probabilities: Dict[str, np.ndarray] = {}
    readout_telemetry: Dict[str, object] = {}
    for role in ROLE_NAMES:
        print(json.dumps({"stage": "oof_readout", "role": role}), flush=True)
        probabilities, telemetry = fit_oof_readout(
            descriptors[role], targets, folds, seed=int(args.seed)
        )
        readout_probabilities[role] = probabilities
        readout_telemetry[role] = telemetry

    analysis = build_analysis(
        targets=targets,
        folds=folds,
        keeper_probabilities=clean_probabilities,
        role_probabilities=readout_probabilities,
        descriptor_stats=descriptor_stats,
    )
    engineering = jigsaw_engineering_checks(int(args.seed))
    contact_path = output_dir / "jigsaw_contact_sheet.png"
    contact_sheet = _render_contact_sheet(
        contact_path,
        extraction["visual_cache"],
        labels,
        class_names,
        semantics,
    )
    descriptor_stats_path = output_dir / "descriptor_statistics.json"
    readout_path = output_dir / "readout_telemetry.json"
    prediction_path = output_dir / "predictions.csv"
    fold_path = output_dir / "fold_summary.csv"
    cache_path = output_dir / "oof_probabilities.npz"
    _write_json(descriptor_stats_path, descriptor_stats)
    _write_json(readout_path, readout_telemetry)
    _write_csv(
        prediction_path,
        _prediction_rows(
            targets=targets,
            folds=folds,
            paths=paths,
            source_stems=source_stems.tolist(),
            keeper_probabilities=clean_probabilities,
            role_probabilities=readout_probabilities,
            class_names=class_names,
        ),
    )
    _write_csv(fold_path, analysis["folds"])
    np.savez_compressed(
        cache_path,
        targets=targets,
        folds=folds,
        keeper_probabilities=clean_probabilities,
        **{
            f"probabilities_{role}": readout_probabilities[role]
            for role in ROLE_NAMES
        },
    )
    cache_replay = _load_probability_cache(cache_path)
    replayed_analysis = build_analysis(
        targets=cache_replay["targets"],
        folds=cache_replay["folds"],
        keeper_probabilities=cache_replay["keeper_probabilities"],
        role_probabilities=cache_replay["role_probabilities"],
        descriptor_stats=descriptor_stats,
    )
    replay_difference = _recursive_numeric_difference(analysis, replayed_analysis)

    role_shapes = {
        role: [int(value) for value in descriptors[role].shape]
        for role in ROLE_NAMES
    }
    clean_global_reference = descriptors["clean_control"][:, -FEATURE_DIM:]
    clean_global_identical = all(
        np.array_equal(clean_global_reference, descriptors[role][:, -FEATURE_DIM:])
        for role in ROLE_NAMES
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
        "train_split_only": True,
        "full_train_support_9215": len(base_dataset) == EXPECTED_TRAIN_ROWS,
        "ordered_indices_complete": bool(
            np.array_equal(indices, np.arange(EXPECTED_TRAIN_ROWS, dtype=np.int64))
        ),
        "targets_match_dataset": bool(np.array_equal(targets, labels)),
        "source_fold_assignment_complete": bool(fold_assignment["assignment_complete"]),
        "source_fold_overlap_zero": int(fold_assignment["source_overlap"]) == 0,
        "train_val_source_overlap_zero": len(train_val_overlap) == 0,
        "train_test_source_overlap_zero": len(train_test_overlap) == 0,
        "cidt_alignment_complete": all(cidt_checks.values()),
        "all_role_shapes_9215x1024": all(
            shape == [EXPECTED_TRAIN_ROWS, ROLE_DIM] for shape in role_shapes.values()
        ),
        "all_descriptors_finite": all(
            bool(descriptor_stats[role]["all_values_finite"]) for role in ROLE_NAMES
        ),
        "all_roles_nonzero_effective_rank": all(
            float(descriptor_stats[role]["effective_rank_512_row_sketch"]) > 1.0
            for role in ROLE_NAMES
        ),
        "clean_global_blocks_bit_exact": clean_global_identical,
        "jigsaw_engineering_checks_passed": bool(engineering["all_passed"]),
        "all_20_oof_fits_converged": all(
            bool(readout_telemetry[role]["all_converged"]) for role in ROLE_NAMES
        ),
        "every_oof_row_predicted_once": all(
            bool(readout_telemetry[role]["every_row_predicted_once"])
            for role in ROLE_NAMES
        ),
        "oof_probabilities_finite_normalized": all(
            np.isfinite(readout_probabilities[role]).all()
            and np.allclose(
                readout_probabilities[role].sum(axis=1), 1.0, atol=1e-10, rtol=0.0
            )
            for role in ROLE_NAMES
        ),
        "model_state_bit_exact": extraction["state_before_sha256"]
        == extraction["state_after_sha256"],
        "requested_workers_4": int(args.num_workers) == 4,
        "safe_loader_no_worker_fallback": int(extraction["loader"]["effective_num_workers"])
        == 4,
        "peak_cuda_allocation_le_75_gib": int(extraction["peak_cuda_bytes"])
        <= int(7.5 * 1024**3),
        "independent_probability_replay_le_1e_12": replay_difference <= 1e-12,
        "contact_sheet_complete": len(contact_sheet["rows"]) == len(class_names),
    }
    structural_passed = all(structural_gates.values())
    automated_gate_passed = structural_passed and bool(
        analysis["mechanism_gates_passed"]
    )
    artifacts = {
        "probability_cache": _artifact_record(cache_path, root=output_dir),
        "predictions": _artifact_record(prediction_path, root=output_dir),
        "fold_summary": _artifact_record(fold_path, root=output_dir),
        "descriptor_statistics": _artifact_record(
            descriptor_stats_path, root=output_dir
        ),
        "readout_telemetry": _artifact_record(readout_path, root=output_dir),
        "jigsaw_contact_sheet": _artifact_record(contact_path, root=output_dir),
    }
    summary = {
        "mode": "pmg_progressive_jigsaw_signal_a0_train_information_gate",
        "status": "awaiting_visual_review"
        if automated_gate_passed
        else "rejected_automated_gate",
        "split": "train",
        "rows": len(base_dataset),
        "validation_data_used": False,
        "test_data_used": False,
        "validation_test_filename_scan_only": True,
        "class_names": list(class_names),
        "focus_class": FOCUS_CLASS,
        "restricted_negative_classes": list(RESTRICTED_NEGATIVE_CLASSES),
        "stage_layers": list(STAGE_LAYERS),
        "jigsaw_grids": list(JIGSAW_GRIDS),
        "role_names": list(ROLE_NAMES),
        "role_shapes": role_shapes,
        "dataset_identity_sha256": dataset_identity,
        "eval_semantics": semantics,
        "provenance": provenance,
        "repo_state": repo_state,
        "source_audit": {
            "train_sources": len(train_sources),
            "val_sources": len(val_sources),
            "test_sources": len(test_sources),
            "train_val_overlap": len(train_val_overlap),
            "train_test_overlap": len(train_test_overlap),
            "overlap_examples": (train_val_overlap + train_test_overlap)[:20],
            "fold_assignment": fold_assignment,
            "note": (
                "Validation/test filenames only; pixels, labels, predictions, and "
                "metrics were not opened."
            ),
        },
        "cidt_replay": {
            "checks": cidt_checks,
            "probability_max_abs_difference": cidt_probability_error,
        },
        "jigsaw_engineering": engineering,
        "descriptor_sha256": extraction["descriptor_sha256"],
        "descriptor_statistics": descriptor_stats,
        "readout_telemetry": readout_telemetry,
        "analysis": analysis,
        "runtime": {
            "device": str(device),
            "torch_version": torch.__version__,
            "cuda_version": torch.version.cuda,
            "gpu_name": torch.cuda.get_device_name(device)
            if device.type == "cuda"
            else None,
            "batch_size": int(args.batch_size),
            "requested_num_workers": int(args.num_workers),
            "loader": extraction["loader"],
            "patch_counts": extraction["patch_counts"],
            "feature_extraction_elapsed_seconds": extraction["elapsed_seconds"],
            "throughput_source_images_per_second": extraction[
                "throughput_source_images_per_second"
            ],
            "throughput_forward_views_per_second": extraction[
                "throughput_forward_views_per_second"
            ],
            "peak_cuda_bytes": extraction["peak_cuda_bytes"],
            "peak_cuda_gib": float(extraction["peak_cuda_bytes"] / 1024**3),
            "model_state_before_sha256": extraction["state_before_sha256"],
            "model_state_after_sha256": extraction["state_after_sha256"],
        },
        "probability_replay": {
            "maximum_numeric_difference": float(replay_difference),
            "passed": bool(replay_difference <= 1e-12),
        },
        "contact_sheet": contact_sheet,
        "visual_review": {
            "required": True,
            "completed": False,
            "geometry_passed": None,
        },
        "structural_gates": structural_gates,
        "structural_gates_passed": structural_passed,
        "automated_gate_passed": automated_gate_passed,
        "trainer_integration_authorized": False,
        "short_pair_authorized": False,
        "full_train_authorized": False,
        "current_command_update_authorized": False,
        "artifacts": artifacts,
        "guardrail": (
            "Only an automated pass plus explicit fixed-sheet geometry review may "
            "authorize one prospectively locked matched short pair. Full train and "
            "current-best command changes remain forbidden."
        ),
    }
    summary_path = output_dir / "summary.json"
    _write_json(summary_path, summary)
    _write_manifest(output_dir)
    return summary


def preflight(args: argparse.Namespace) -> Dict[str, object]:
    _validate_locked_args(args)
    if Path(args.output_dir).expanduser().resolve().exists():
        raise FileExistsError(f"Preflight output must not exist: {args.output_dir}")
    provenance = verify_locked_inputs()
    engineering = jigsaw_engineering_checks(int(args.seed))
    if not bool(engineering["all_passed"]):
        raise RuntimeError(f"PMG jigsaw engineering check failed: {engineering}")
    return {
        "mode": "pmg_progressive_jigsaw_signal_a0_preflight",
        "locked_inputs_verified": True,
        "official_repository_verified": True,
        "jigsaw_engineering_checks": engineering,
        "repo_state_observed_not_gated_until_formal": _repo_state(),
        "output_created": False,
        "model_loaded": False,
        "dataset_loaded": False,
        "provenance": provenance,
    }


def main(argv: Optional[Sequence[str]] = None) -> int:
    args = parse_args(argv)
    if args.preflight_only and (
        args.replay_summary is not None or args.finalize_visual_review is not None
    ):
        raise ValueError(
            "preflight-only cannot be combined with replay or visual finalization"
        )
    if args.expected_summary_sha256 and args.finalize_visual_review is None:
        raise ValueError(
            "expected-summary-sha256 is valid only with finalize-visual-review"
        )
    if args.preflight_only:
        print(json.dumps(to_serializable(preflight(args)), indent=2), flush=True)
        return 0
    if args.finalize_visual_review is not None:
        if args.replay_summary is None:
            raise ValueError(
                "finalize-visual-review requires --replay-summary pointing to summary.json"
            )
        if not args.expected_summary_sha256:
            raise ValueError(
                "finalize-visual-review requires --expected-summary-sha256"
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
    summary = run_audit(args)
    print(
        json.dumps(
            to_serializable(
                {
                    "status": summary["status"],
                    "rows": summary["rows"],
                    "candidate_vs_control_deltas": summary["analysis"][
                        "candidate_vs_control_deltas"
                    ],
                    "candidate_vs_control_transitions": summary["analysis"][
                        "candidate_vs_control_transitions"
                    ],
                    "failed_mechanism_gates": [
                        name
                        for name, passed in summary["analysis"][
                            "mechanism_gates"
                        ].items()
                        if not passed
                    ],
                    "structural_gates_passed": summary[
                        "structural_gates_passed"
                    ],
                    "automated_gate_passed": summary["automated_gate_passed"],
                    "summary_sha256": _sha256(
                        Path(args.output_dir).resolve() / "summary.json"
                    ),
                    "manifest_sha256": _sha256(
                        Path(args.output_dir).resolve() / "artifact_manifest.json"
                    ),
                }
            ),
            indent=2,
        ),
        flush=True,
    )
    if not bool(summary["structural_gates_passed"]):
        return 2
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
