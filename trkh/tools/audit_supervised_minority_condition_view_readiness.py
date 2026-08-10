from __future__ import annotations

import argparse
import copy
import csv
import gc
import hashlib
import json
import math
import os
from pathlib import Path
import statistics
import subprocess
import time
from typing import Dict, Iterable, Mapping, Optional, Sequence

import numpy as np

os.environ.setdefault("CUBLAS_WORKSPACE_CONFIG", ":4096:8")

import torch
from PIL import Image, ImageDraw, ImageFilter
from sklearn.metrics import roc_auc_score
from torch import Tensor, nn
import torch.nn.functional as F

from trkh.core.utils import set_seed
from trkh.data.dataset import MangoYOLOCropDataset
from trkh.evaluation.input_normalization import checkpoint_input_normalization
from trkh.evaluation.robustness_eval import _forward_classification_with_metadata
from trkh.inference.inference import load_checkpoint
from trkh.models.model import (
    classification_logits_from_features,
    create_model,
    load_model_state,
)
from trkh.tools.audit_ceconv_stem_residual_readiness import (
    _build_dataset,
    _failed_export,
    _full_worktree_clean,
    _heat_overlay,
    _onnx_compare,
    _rgb_from_tensor,
    _tracked_worktree_clean,
)
from trkh.tools.audit_class1_reference_agem_readiness import (
    FOCUS_CLASS,
    RESTRICTED_NEGATIVE_CLASSES,
    _cohort_serializable,
    _locked_cohort_summary,
)
from trkh.tools.audit_deep_class_prompt_readiness import LIGHTING_CONDITIONS
from trkh.tools.audit_more_model_rebalancing_readiness import (
    CleanTrainRow,
    _ordered_index_sha256,
    _read_clean_train_rows,
)
from trkh.tools.audit_support_constrained_capsule_alm_readiness import (
    _condition_loader,
)
from trkh.tools.audit_visual_contrast_attention_readiness import _state_sha256
from trkh.tools.audit_xca_dual_axis_readiness import (
    _amp_dtype,
    _comparison,
    _git_commit,
    _prepare_output_dir,
    _sha256,
    _verify_sha256,
)


A0_CONTRACT_ID = "a0_fp16"
A1_CONTRACT_ID = "a1_fp32_cidt"
NUM_CLASSES = 5
EMBED_DIM = 256
HIDDEN_DIM = 64
PROJECTION_DIM = 128
SEED = 42
FOLD = 0
CONTRASTIVE_EPOCHS = 20
PROBE_EPOCHS = 10
CONTRASTIVE_BATCH_SIZE = 256
PROBE_BATCH_SIZE = 1024
A0_EXTRACT_BATCH_SIZE = 32
A1_EXTRACT_BATCH_SIZE = 64
XAI_BATCH_SIZE = 4
TEMPERATURE = 0.07
BASE_TEMPERATURE = 0.07
ADAPTER_RESIDUAL_SCALE = 0.10
PROBE_RESIDUAL_SCALE = 0.10
STAGE1_LEARNING_RATE = 0.0625
STAGE1_WARMUP_START = 0.00625
STAGE1_ETA_MIN = 0.0000625
STAGE1_WARMUP_EPOCHS = 2
STAGE1_MOMENTUM = 0.9
STAGE1_WEIGHT_DECAY = 1e-4
STAGE2_LEARNING_RATE = 3e-4
STAGE2_WEIGHT_DECAY = 1e-4
EXPECTED_KEEPER_PARAMETERS = 7_245_590
EXPECTED_REPRESENTATION_PARAMETERS = 132_288
EXPECTED_PROBE_PARAMETERS = 1_285
EXPECTED_TOTAL_PARAMETERS = 133_573
EXPECTED_TRAIN_ROWS = 9_215
EXPECTED_FIT_ROWS = 7_372
EXPECTED_HOLDOUT_ROWS = 1_843
EXPECTED_FIT_CLASS_COUNTS = [1_561, 432, 1_527, 2_017, 1_835]
EXPECTED_CONTRASTIVE_OCCURRENCES = 147_440
EXPECTED_PROBE_OCCURRENCES = 73_720
KNN_NEIGHBORS = 369
CONDITIONS = ("clean", *(name for name, _brightness, _contrast in LIGHTING_CONDITIONS))
SECONDARY_CONDITIONS = tuple(name for name, _brightness, _contrast in LIGHTING_CONDITIONS)

LOCKED_KEEPER_SHA256 = "1f49d577240c69dc63c30af70db52ec2aa9da65a17aef1c4b1c09ece6c482677"
LOCKED_LAUNCHER_ARGS_SHA256 = "908a05cf66b2a01162cae62e4ff2251eaae1297d31e70510144e4954159b7eff"
LOCKED_DATA_SHA256 = "716e33df24c63a9e9920f97b685199707fb84ab4c7154544f5dd9a3e00d884ef"
LOCKED_CIDT_SUMMARY_SHA256 = "d4891edf2963ab12385b7ce5bdc812ec3e19c5c098acd25c66eb557af541d7ad"
LOCKED_CIDT_PREDICTIONS_SHA256 = "2e0993752d58d99ea429bfefe1e2bfe6fa949e45aea1a26cc4bdfee97d4db21c"
LOCKED_A0_PROTOCOL_SHA256 = "d4fb528cc45dbcfbbdfd0218971cde05e7ba203b3ce0ad29f6080c3f561b4ce9"
LOCKED_A1_PROTOCOL_SHA256 = "e17f5cb56631b1f24206b6183fdef7b92e17a050097b84c600adb0a0744d9aa2"
LOCKED_CURRENT_COMMAND_SHA256 = "36b9aa1a21b765829acf4c8321be147bd76297de4ccdb8a40e6dee8e37940faf"
LOCKED_COMMAND_HISTORY_SHA256 = "39bd2879ce66fddf36a953021ea1e40f8d9de6cb4334b9b825011b2b8dc98f53"
LOCKED_TTC_COMMIT = "0b1e6974254993b074ad27a226c7ce864da7f95c"
LOCKED_TTC_TREE = "707c9b34ece30ac4a23f1de28be9d0ee803710b7"
LOCKED_TTC_PAPER_SHA256 = "9919077444fcd45005f1114128524bc593fb1852823e7e849f31b696584c95d8"
LOCKED_TTC_LOSS_SHA256 = "200e763967904f1c498d2e028485b8e7026f4c469c480897ffcf6a97fb934c71"
LOCKED_TTC_MODEL_SHA256 = "9a73df402cd160cf2b4188efcd19739157543a88c71a6fba54af5fcfd1d491f1"
LOCKED_TTC_README_SHA256 = "b48559693415782936966c88a4dd3a067e53abedad770729f6d53443cf7e2491"
LOCKED_TTC_LICENSE_SHA256 = "2c25d4b5ace923d0e3d2f2105f1639e2604335de4cb0a3d4b2bd16e5cb74087f"
LOCKED_FIT_INDEX_SHA256 = "22edca99022fe2dcd0287a08b5f6b7c0d699603b904f65c82b51fdea7f31ce5d"
LOCKED_HOLDOUT_INDEX_SHA256 = "a628686b491c8b8f10bbf1782e6c84a923f21c8b53617cf0b0325260e97e97ae"
LOCKED_CONTRASTIVE_SCHEDULE_SHA256 = "d73ba25938afb8ba69061ea7c1ae79ac58b254d8a7c9dbb952566b130da1b52a"
LOCKED_PROBE_SCHEDULE_SHA256 = "bf5568009a9e0480aec7f6f8d7f3cce7f08409ac74325930747f5d022a4f10f0"

PROTOCOL_CONTRACTS: Dict[str, Dict[str, object]] = {
    A0_CONTRACT_ID: {
        "contract_id": A0_CONTRACT_ID,
        "method": "supervised_minority_condition_view_a0",
        "extract_batch_size": A0_EXTRACT_BATCH_SIZE,
        "cache_autocast_enabled": True,
        "cache_precision": "cuda_fp16",
        "replay_rows": 32,
        "cidt_maximum_probability_error": 2e-3,
        "protocol_sha256": LOCKED_A0_PROTOCOL_SHA256,
        "output_relative_path": (
            "runs/audit_supervised_minority_condition_view_readiness_20260716"
        ),
    },
    A1_CONTRACT_ID: {
        "contract_id": A1_CONTRACT_ID,
        "method": "supervised_minority_condition_view_a1_fp32_cidt",
        "extract_batch_size": A1_EXTRACT_BATCH_SIZE,
        "cache_autocast_enabled": False,
        "cache_precision": "fp32_autocast_disabled",
        "replay_rows": 64,
        "cidt_maximum_probability_error": 1e-6,
        "protocol_sha256": LOCKED_A1_PROTOCOL_SHA256,
        "output_relative_path": (
            "runs/audit_supervised_minority_condition_view_a1_fp32_20260716"
        ),
    },
}


def _protocol_contract(args: argparse.Namespace) -> Dict[str, object]:
    contract_id = str(args.contract)
    if contract_id not in PROTOCOL_CONTRACTS:
        raise ValueError(f"Unknown TTC-SupMin protocol contract: {contract_id}")
    return dict(PROTOCOL_CONTRACTS[contract_id])


def parse_args(argv: Optional[Sequence[str]] = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Locked train-only CVPR-2025 Supervised-Minority condition-view "
            "readiness audit. Validation and test access are forbidden."
        )
    )
    parser.add_argument(
        "--contract",
        choices=tuple(PROTOCOL_CONTRACTS),
        default=A0_CONTRACT_ID,
    )
    parser.add_argument(
        "--checkpoint",
        type=Path,
        default=Path(
            "runs/probe_v8_yolof_pairroute_teacherfocusbinary015_boundarydrop_bboxprior_120b_2e_20260701/checkpoints/best.pt"
        ),
    )
    parser.add_argument(
        "--launcher-args",
        type=Path,
        default=Path(
            "runs/probe_v8_yolof_pairroute_teacherfocusbinary015_boundarydrop_bboxprior_120b_2e_20260701/launcher_args.json"
        ),
    )
    parser.add_argument(
        "--data",
        type=Path,
        default=Path(r"D:\DataAI\AIEx\newdataset\yolo_f\data.yaml"),
    )
    parser.add_argument(
        "--cidt-summary",
        type=Path,
        default=Path("runs/audit_cidt_readiness_full_train_20260714/summary.json"),
    )
    parser.add_argument(
        "--cidt-predictions",
        type=Path,
        default=Path(
            "runs/audit_cidt_readiness_full_train_20260714/predictions_all_conditions.csv"
        ),
    )
    parser.add_argument(
        "--protocol",
        type=Path,
        default=Path(
            "docs/TRKH_5CLASS_SUPERVISED_MINORITY_CONDITION_VIEW_READINESS_PROTOCOL_20260716.md"
        ),
    )
    parser.add_argument(
        "--official-ttc-root",
        type=Path,
        default=Path(r"D:\DataAI\external_sources\official\ttc-cvpr2025"),
    )
    parser.add_argument(
        "--ttc-paper",
        type=Path,
        default=Path(
            r"D:\DataAI\external_sources\papers\mildenberger2025_tale_two_classes.pdf"
        ),
    )
    parser.add_argument(
        "--current-commands",
        type=Path,
        default=Path("docs/TRKH_CURRENT_BEST_FULL_TRAIN_COMMANDS_20260706.txt"),
    )
    parser.add_argument(
        "--command-history",
        type=Path,
        default=Path("docs/TRKH_CURRENT_BEST_COMMAND_UPDATE_HISTORY.txt"),
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=Path("runs/audit_supervised_minority_condition_view_readiness_20260716"),
    )
    parser.add_argument("--preflight-only", action="store_true", default=False)
    parser.add_argument("--device", choices=("cuda",), default="cuda")
    parser.add_argument("--batch-size", type=int, default=A0_EXTRACT_BATCH_SIZE)
    parser.add_argument("--num-workers", type=int, default=4)
    parser.add_argument("--xai-batch-size", type=int, default=XAI_BATCH_SIZE)
    parser.add_argument("--seed", type=int, default=SEED)
    parser.add_argument("--fold", type=int, default=FOLD)
    return parser.parse_args(argv)


def _locked_args_exact(args: argparse.Namespace) -> bool:
    contract = _protocol_contract(args)
    return bool(
        str(args.device) == "cuda"
        and int(args.batch_size) == int(contract["extract_batch_size"])
        and int(args.num_workers) == 4
        and int(args.xai_batch_size) == XAI_BATCH_SIZE
        and int(args.seed) == SEED
        and int(args.fold) == FOLD
        and Path(args.output_dir).resolve()
        == Path(str(contract["output_relative_path"])).resolve()
    )


def _source_paths(args: argparse.Namespace) -> Dict[str, Path]:
    ttc = Path(args.official_ttc_root).resolve()
    return {
        "checkpoint": Path(args.checkpoint).resolve(),
        "launcher_args": Path(args.launcher_args).resolve(),
        "data": Path(args.data).resolve(),
        "cidt_summary": Path(args.cidt_summary).resolve(),
        "cidt_predictions": Path(args.cidt_predictions).resolve(),
        "protocol": Path(args.protocol).resolve(),
        "current_commands": Path(args.current_commands).resolve(),
        "command_history": Path(args.command_history).resolve(),
        "ttc_root": ttc,
        "ttc_paper": Path(args.ttc_paper).resolve(),
        "ttc_loss": ttc / "loss.py",
        "ttc_model": ttc / "models" / "sup_cont.py",
        "ttc_readme": ttc / "README.md",
        "ttc_license": ttc / "LICENSE",
    }


def _canonical_record_sha256(records: Iterable[Mapping[str, object]]) -> str:
    digest = hashlib.sha256()
    for record in records:
        digest.update(
            json.dumps(record, sort_keys=True, separators=(",", ":")).encode("utf-8")
        )
        digest.update(b"\n")
    return digest.hexdigest()


def locked_contrastive_epoch_order(
    fit_indices: Sequence[int], *, epoch: int
) -> np.ndarray:
    values = np.asarray([int(value) for value in fit_indices], dtype=np.int64)
    rng = np.random.default_rng(SEED + int(epoch))
    return values[rng.permutation(values.size)]


def locked_secondary_condition(*, epoch: int, sample_index: int) -> str:
    return SECONDARY_CONDITIONS[(int(epoch) + int(sample_index)) % len(SECONDARY_CONDITIONS)]


def contrastive_schedule_sha256(fit_indices: Sequence[int]) -> str:
    def records():
        for epoch in range(CONTRASTIVE_EPOCHS):
            for position, sample_index in enumerate(
                locked_contrastive_epoch_order(fit_indices, epoch=epoch).tolist()
            ):
                yield {
                    "epoch": epoch + 1,
                    "position": position,
                    "sample_index": int(sample_index),
                    "secondary_condition": locked_secondary_condition(
                        epoch=epoch, sample_index=int(sample_index)
                    ),
                }

    return _canonical_record_sha256(records())


def locked_probe_epoch_order(fit_indices: Sequence[int], *, epoch: int) -> np.ndarray:
    values = np.asarray([int(value) for value in fit_indices], dtype=np.int64)
    rng = np.random.default_rng(1042 + int(epoch))
    return values[rng.permutation(values.size)]


def probe_schedule_sha256(fit_indices: Sequence[int]) -> str:
    def records():
        for epoch in range(PROBE_EPOCHS):
            for position, sample_index in enumerate(
                locked_probe_epoch_order(fit_indices, epoch=epoch).tolist()
            ):
                yield {
                    "epoch": epoch + 1,
                    "position": position,
                    "sample_index": int(sample_index),
                }

    return _canonical_record_sha256(records())


class ConditionViewVariant(nn.Module):
    def __init__(self) -> None:
        super().__init__()
        self.adapter_norm = nn.LayerNorm(EMBED_DIM)
        self.adapter_down = nn.Linear(EMBED_DIM, HIDDEN_DIM)
        self.adapter_up = nn.Linear(HIDDEN_DIM, EMBED_DIM)
        self.projector = nn.Sequential(
            nn.Linear(EMBED_DIM, EMBED_DIM),
            nn.ReLU(inplace=False),
            nn.Linear(EMBED_DIM, PROJECTION_DIM),
        )
        self.probe = nn.Linear(EMBED_DIM, NUM_CLASSES)
        nn.init.zeros_(self.adapter_up.weight)
        nn.init.zeros_(self.adapter_up.bias)
        nn.init.zeros_(self.probe.weight)
        nn.init.zeros_(self.probe.bias)

    def adapted_feature(self, pooled: Tensor) -> Tensor:
        residual = self.adapter_up(F.gelu(self.adapter_down(self.adapter_norm(pooled))))
        return pooled + ADAPTER_RESIDUAL_SCALE * residual

    def projected_feature(self, pooled: Tensor) -> Tensor:
        return F.normalize(self.projector(self.adapted_feature(pooled)), dim=-1)

    def deployed_logits(self, pooled: Tensor, raw_logits: Tensor) -> Tensor:
        return raw_logits + PROBE_RESIDUAL_SCALE * self.probe(
            self.adapted_feature(pooled)
        )

    def representation_parameters(self) -> Iterable[nn.Parameter]:
        yield from self.adapter_norm.parameters()
        yield from self.adapter_down.parameters()
        yield from self.adapter_up.parameters()
        yield from self.projector.parameters()


def _positive_mask(labels: Tensor, *, supervised_minority: bool) -> Tensor:
    if labels.ndim != 1:
        raise ValueError("Binary labels must be one-dimensional.")
    labels = labels.to(dtype=torch.long)
    if supervised_minority:
        mask = torch.eye(labels.numel(), device=labels.device, dtype=torch.float32)
        minority = labels.eq(FOCUS_CLASS)
        mask = torch.where(
            minority[:, None] & minority[None, :],
            torch.ones_like(mask),
            mask,
        )
        return mask
    return labels[:, None].eq(labels[None, :]).to(dtype=torch.float32)


def supervised_contrastive_loss(
    features: Tensor,
    labels: Tensor,
    *,
    supervised_minority: bool,
) -> tuple[Tensor, Dict[str, object]]:
    if features.ndim != 3 or features.size(1) != 2:
        raise ValueError("Contrastive features must have shape [batch,2,dim].")
    if labels.shape != features.shape[:1]:
        raise ValueError("Contrastive labels do not align with features.")
    batch = int(features.size(0))
    base_mask = _positive_mask(labels, supervised_minority=supervised_minority)
    contrast = torch.cat(torch.unbind(F.normalize(features.float(), dim=-1), dim=1), dim=0)
    logits = contrast @ contrast.T / TEMPERATURE
    logits = logits - logits.max(dim=1, keepdim=True).values.detach()
    mask = base_mask.repeat(2, 2)
    self_mask = torch.eye(2 * batch, device=features.device, dtype=torch.bool)
    logits_mask = ~self_mask
    mask = mask * logits_mask.to(dtype=mask.dtype)
    exp_logits = torch.exp(logits) * logits_mask.to(dtype=logits.dtype)
    log_prob = logits - torch.log(exp_logits.sum(dim=1, keepdim=True).clamp_min(1e-12))
    positive_counts = mask.sum(dim=1)
    per_anchor = -(mask * log_prob).sum(dim=1) / positive_counts.clamp_min(1e-12)
    loss = (TEMPERATURE / BASE_TEMPERATURE) * per_anchor.mean()
    class1_anchor = labels.repeat(2).eq(FOCUS_CLASS)
    return loss, {
        "positive_count_min": float(positive_counts.min().item()),
        "positive_count_max": float(positive_counts.max().item()),
        "class1_positive_count_min": float(positive_counts[class1_anchor].min().item()),
        "nonclass1_positive_count_min": float(positive_counts[~class1_anchor].min().item()),
        "nonclass1_positive_count_max": float(positive_counts[~class1_anchor].max().item()),
        "finite": bool(torch.isfinite(loss).item()),
    }


def independent_contrastive_loss(
    features: Tensor,
    labels: Tensor,
    *,
    supervised_minority: bool,
) -> Tensor:
    values = F.normalize(features.double(), dim=-1)
    batch = int(values.size(0))
    flat = torch.cat([values[:, 0], values[:, 1]], dim=0)
    losses = []
    for anchor in range(2 * batch):
        sample = anchor % batch
        anchor_label = int(labels[sample].item())
        positives = []
        denominators = []
        for other in range(2 * batch):
            if other == anchor:
                continue
            other_sample = other % batch
            denominators.append(other)
            if supervised_minority:
                is_positive = (
                    anchor_label == FOCUS_CLASS
                    and int(labels[other_sample].item()) == FOCUS_CLASS
                ) or other_sample == sample
            else:
                is_positive = int(labels[other_sample].item()) == anchor_label
            if is_positive:
                positives.append(other)
        scores = flat[anchor] @ flat.T / TEMPERATURE
        log_denom = torch.logsumexp(scores[denominators], dim=0)
        losses.append(torch.stack([-(scores[index] - log_denom) for index in positives]).mean())
    return torch.stack(losses).mean().to(dtype=features.dtype)


def _stage1_learning_rate(epoch: int) -> float:
    epoch = int(epoch)
    if epoch < STAGE1_WARMUP_EPOCHS:
        if STAGE1_WARMUP_EPOCHS == 1:
            return STAGE1_LEARNING_RATE
        fraction = epoch / float(STAGE1_WARMUP_EPOCHS - 1)
        return STAGE1_WARMUP_START + fraction * (
            STAGE1_LEARNING_RATE - STAGE1_WARMUP_START
        )
    denominator = max(1, CONTRASTIVE_EPOCHS - STAGE1_WARMUP_EPOCHS - 1)
    progress = (epoch - STAGE1_WARMUP_EPOCHS) / float(denominator)
    return STAGE1_ETA_MIN + 0.5 * (STAGE1_LEARNING_RATE - STAGE1_ETA_MIN) * (
        1.0 + math.cos(math.pi * progress)
    )


def _stream_tensor_sha256(value: Tensor) -> str:
    tensor = value.detach().cpu().contiguous()
    digest = hashlib.sha256()
    digest.update(str(tensor.dtype).encode("ascii"))
    digest.update(json.dumps(list(tensor.shape)).encode("ascii"))
    digest.update(tensor.numpy().tobytes())
    return digest.hexdigest()


def _load_locked_inputs(
    args: argparse.Namespace,
) -> tuple[Dict[str, object], list[CleanTrainRow], Dict[str, object]]:
    contract = _protocol_contract(args)
    if not _locked_args_exact(args):
        raise ValueError(
            f"Arguments differ from precommitted TTC-SupMin contract "
            f"{contract['contract_id']}."
        )
    paths = _source_paths(args)
    hashes = {
        "checkpoint": _verify_sha256(paths["checkpoint"], LOCKED_KEEPER_SHA256, "keeper checkpoint"),
        "launcher_args": _verify_sha256(paths["launcher_args"], LOCKED_LAUNCHER_ARGS_SHA256, "launcher args"),
        "data": _verify_sha256(paths["data"], LOCKED_DATA_SHA256, "data YAML"),
        "cidt_summary": _verify_sha256(paths["cidt_summary"], LOCKED_CIDT_SUMMARY_SHA256, "CIDT summary"),
        "cidt_predictions": _verify_sha256(paths["cidt_predictions"], LOCKED_CIDT_PREDICTIONS_SHA256, "CIDT predictions"),
        "protocol": _verify_sha256(
            paths["protocol"],
            str(contract["protocol_sha256"]),
            "TTC-SupMin protocol",
        ),
        "current_commands": _verify_sha256(paths["current_commands"], LOCKED_CURRENT_COMMAND_SHA256, "current-best commands"),
        "command_history": _verify_sha256(paths["command_history"], LOCKED_COMMAND_HISTORY_SHA256, "command history"),
        "ttc_paper": _verify_sha256(paths["ttc_paper"], LOCKED_TTC_PAPER_SHA256, "TTC paper"),
        "ttc_loss": _verify_sha256(paths["ttc_loss"], LOCKED_TTC_LOSS_SHA256, "official TTC loss"),
        "ttc_model": _verify_sha256(paths["ttc_model"], LOCKED_TTC_MODEL_SHA256, "official TTC model"),
        "ttc_readme": _verify_sha256(paths["ttc_readme"], LOCKED_TTC_README_SHA256, "official TTC README"),
        "ttc_license": _verify_sha256(paths["ttc_license"], LOCKED_TTC_LICENSE_SHA256, "official TTC license"),
    }
    if _git_commit(paths["ttc_root"]) != LOCKED_TTC_COMMIT:
        raise ValueError("Official TTC commit differs from protocol.")
    tree = subprocess.run(
        ["git", "-C", str(paths["ttc_root"]), "rev-parse", "HEAD^{tree}"],
        check=True,
        capture_output=True,
        text=True,
    ).stdout.strip()
    if tree != LOCKED_TTC_TREE or not _full_worktree_clean(paths["ttc_root"]):
        raise ValueError("Official TTC tree or cleanliness differs from protocol.")
    cidt_summary = json.loads(paths["cidt_summary"].read_text(encoding="utf-8"))
    if bool(cidt_summary.get("test_data_used", True)) or bool(
        cidt_summary.get("validation_predictions_used", cidt_summary.get("validation_data_used", True))
    ):
        raise ValueError("CIDT comparator provenance is not train-only.")
    rows = _read_clean_train_rows(paths["cidt_predictions"])
    cohorts = _locked_cohort_summary(rows, fold=int(args.fold))
    if cohorts["fit_index_sha256"] != LOCKED_FIT_INDEX_SHA256:
        raise ValueError("Fit-index hash differs from protocol.")
    if cohorts["holdout_index_sha256"] != LOCKED_HOLDOUT_INDEX_SHA256:
        raise ValueError("Holdout-index hash differs from protocol.")
    fit_counts = np.bincount(
        np.asarray([rows[index].target for index in cohorts["fit_indices"]], dtype=np.int64),
        minlength=NUM_CLASSES,
    ).tolist()
    if fit_counts != EXPECTED_FIT_CLASS_COUNTS:
        raise ValueError(f"Fit class counts differ: {fit_counts}")
    contrastive_hash = contrastive_schedule_sha256(cohorts["fit_indices"])
    probe_hash = probe_schedule_sha256(cohorts["fit_indices"])
    if contrastive_hash != LOCKED_CONTRASTIVE_SCHEDULE_SHA256:
        raise ValueError("Contrastive schedule hash differs from protocol.")
    if probe_hash != LOCKED_PROBE_SCHEDULE_SHA256:
        raise ValueError("Probe schedule hash differs from protocol.")
    repository_root = Path.cwd().resolve()
    if not _tracked_worktree_clean(repository_root):
        raise ValueError(
            f"Tracked TRKH worktree must be clean for formal TTC-SupMin "
            f"{contract['contract_id']}."
        )
    return (
        {
            "paths": {key: str(value) for key, value in paths.items()},
            "sha256": hashes,
            "ttc_commit": LOCKED_TTC_COMMIT,
            "ttc_tree": tree,
            "ttc_worktree_clean": True,
            "repository_commit": _git_commit(repository_root),
            "tracked_worktree_clean": True,
            "protocol_contract": contract,
            "contrastive_schedule_sha256": contrastive_hash,
            "probe_schedule_sha256": probe_hash,
            "validation_predictions_used": False,
            "test_data_used": False,
        },
        rows,
        cohorts,
    )


def _construct_keeper_and_variants(
    checkpoint: Mapping[str, object],
) -> tuple[nn.Module, Dict[str, ConditionViewVariant], Dict[str, object]]:
    model_config = checkpoint.get("model_config")
    model_state = checkpoint.get("model_state")
    if not isinstance(model_config, Mapping) or not isinstance(model_state, Mapping):
        raise ValueError("Keeper checkpoint has invalid model config/state.")
    set_seed(SEED, deterministic=True)
    keeper = create_model(num_classes=NUM_CLASSES, model_config=model_config).eval()
    load_model_state(keeper, dict(model_state), strict=True)
    for parameter in keeper.parameters():
        parameter.requires_grad_(False)
    rng_before = torch.get_rng_state().clone()
    with torch.random.fork_rng(devices=[]):
        torch.manual_seed(SEED)
        prototype = ConditionViewVariant()
    rng_after = torch.get_rng_state().clone()
    variants = {
        name: copy.deepcopy(prototype)
        for name in ("ce_identity", "standard_supcon", "supervised_minority")
    }
    representation_count = sum(
        parameter.numel() for parameter in prototype.representation_parameters()
    )
    probe_count = sum(parameter.numel() for parameter in prototype.probe.parameters())
    return keeper, variants, {
        "keeper_parameters": sum(parameter.numel() for parameter in keeper.parameters()),
        "representation_parameters": representation_count,
        "probe_parameters": probe_count,
        "total_parameters": representation_count + probe_count,
        "parameter_schema": {
            name: [int(value) for value in parameter.shape]
            for name, parameter in prototype.named_parameters()
        },
        "rng_restored": bool(torch.equal(rng_before, rng_after)),
        "initial_state_sha256": {
            name: _state_sha256(variant) for name, variant in variants.items()
        },
        "initial_states_bit_exact": len(
            {_state_sha256(variant) for variant in variants.values()}
        )
        == 1,
        "adapter_up_zero": bool(
            torch.count_nonzero(prototype.adapter_up.weight).item() == 0
            and torch.count_nonzero(prototype.adapter_up.bias).item() == 0
        ),
        "probe_zero": bool(
            torch.count_nonzero(prototype.probe.weight).item() == 0
            and torch.count_nonzero(prototype.probe.bias).item() == 0
        ),
    }


def _cache_positions(cache: Mapping[str, Tensor]) -> Tensor:
    sample_indices = cache["sample_indices"].to(dtype=torch.long)
    if sample_indices.ndim != 1 or sample_indices.numel() == 0:
        raise ValueError("Feature cache sample indices are invalid.")
    positions = torch.full(
        (int(sample_indices.max().item()) + 1,), -1, dtype=torch.long
    )
    positions[sample_indices] = torch.arange(sample_indices.numel(), dtype=torch.long)
    if int(torch.count_nonzero(positions >= 0).item()) != int(sample_indices.numel()):
        raise ValueError("Feature cache contains duplicate sample indices.")
    return positions


def _subset_cache(
    cache: Mapping[str, Tensor], indices: Sequence[int]
) -> Dict[str, Tensor]:
    requested = torch.tensor([int(value) for value in indices], dtype=torch.long)
    positions = _cache_positions(cache)
    if requested.numel() and int(requested.max().item()) >= int(positions.numel()):
        raise ValueError("Feature-cache subset index is out of range.")
    rows = positions[requested]
    if bool((rows < 0).any().item()):
        raise ValueError("Feature-cache subset contains a missing sample.")
    output = {key: value[rows].clone() for key, value in cache.items()}
    if output["sample_indices"].tolist() != requested.tolist():
        raise ValueError("Feature-cache subset changed sample order.")
    return output


def _extract_condition_cache(
    *,
    keeper: nn.Module,
    dataset: MangoYOLOCropDataset,
    transform,
    indices: Sequence[int],
    condition: str,
    args: argparse.Namespace,
    device: torch.device,
) -> tuple[Dict[str, Tensor], Dict[str, object]]:
    contract = _protocol_contract(args)
    loader, loader_summary = _condition_loader(
        name=condition,
        dataset=dataset,
        transform=transform,
        indices=indices,
        args=args,
        context=f"ttc_supmin_pooled_cache_{condition}",
    )
    pooled_rows: list[Tensor] = []
    logit_rows: list[Tensor] = []
    target_rows: list[Tensor] = []
    index_rows: list[Tensor] = []
    started = time.perf_counter()
    processed = 0
    keeper = keeper.to(device).eval()
    with torch.inference_mode():
        for images_cpu, targets_cpu, metadata_cpu in loader:
            images = images_cpu.to(device=device, non_blocking=True)
            with torch.autocast(
                device_type="cuda",
                dtype=torch.float16,
                enabled=bool(contract["cache_autocast_enabled"]),
            ):
                raw_logits, features = _forward_classification_with_metadata(
                    keeper, images, metadata_cpu, device=device
                )
            if not isinstance(features, Mapping):
                raise ValueError("Keeper pooled-cache forward returned no features.")
            pooled = features.get("pooled")
            sample_indices = metadata_cpu.get("sample_index")
            if not torch.is_tensor(pooled) or tuple(pooled.shape[1:]) != (EMBED_DIM,):
                raise ValueError(
                    f"Keeper pooled feature shape differs from [B,{EMBED_DIM}]."
                )
            if not torch.is_tensor(sample_indices):
                raise ValueError("Pooled-cache batch is missing sample_index.")
            pooled_rows.append(pooled.detach().float().cpu())
            logit_rows.append(raw_logits.detach().float().cpu())
            target_rows.append(targets_cpu.detach().long().cpu())
            index_rows.append(sample_indices.detach().long().cpu())
            processed += int(targets_cpu.numel())
            if processed % 1024 < int(targets_cpu.numel()) or processed == len(indices):
                print(
                    json.dumps(
                        {
                            "phase": "pooled_cache",
                            "condition": condition,
                            "processed": processed,
                            "rows": len(indices),
                            "elapsed_seconds": time.perf_counter() - started,
                        }
                    ),
                    flush=True,
                )
    cache = {
        "pooled": torch.cat(pooled_rows),
        "raw_logits": torch.cat(logit_rows),
        "targets": torch.cat(target_rows),
        "sample_indices": torch.cat(index_rows),
    }
    expected = [int(value) for value in indices]
    if cache["sample_indices"].tolist() != expected:
        raise ValueError(f"Condition {condition} changed ordered sample identity.")
    if cache["pooled"].shape != (len(indices), EMBED_DIM):
        raise ValueError(f"Condition {condition} pooled cache has an invalid shape.")
    if cache["raw_logits"].shape != (len(indices), NUM_CLASSES):
        raise ValueError(f"Condition {condition} logit cache has an invalid shape.")
    finite = all(
        bool(torch.isfinite(value).all().item())
        for key, value in cache.items()
        if key in {"pooled", "raw_logits"}
    )
    if not finite:
        raise ValueError(f"Condition {condition} cache contains non-finite values.")
    return cache, {
        "condition": condition,
        "rows": len(indices),
        "pooled_shape": list(cache["pooled"].shape),
        "logit_shape": list(cache["raw_logits"].shape),
        "pooled_sha256": _stream_tensor_sha256(cache["pooled"]),
        "logit_sha256": _stream_tensor_sha256(cache["raw_logits"]),
        "target_sha256": _stream_tensor_sha256(cache["targets"]),
        "sample_index_sha256": _stream_tensor_sha256(cache["sample_indices"]),
        "finite": finite,
        "autocast": str(contract["cache_precision"]),
        "loader": loader_summary,
        "elapsed_seconds": float(time.perf_counter() - started),
    }


def _extract_all_condition_caches(
    *,
    keeper: nn.Module,
    dataset: MangoYOLOCropDataset,
    transform,
    rows: Sequence[CleanTrainRow],
    args: argparse.Namespace,
    device: torch.device,
) -> tuple[Dict[str, Dict[str, Tensor]], Dict[str, object]]:
    contract = _protocol_contract(args)
    replay_rows = int(contract["replay_rows"])
    cidt_error_limit = float(contract["cidt_maximum_probability_error"])
    indices = list(range(EXPECTED_TRAIN_ROWS))
    caches: Dict[str, Dict[str, Tensor]] = {}
    summaries: Dict[str, object] = {}
    clean, clean_summary = _extract_condition_cache(
        keeper=keeper,
        dataset=dataset,
        transform=transform,
        indices=indices,
        condition="clean",
        args=args,
        device=device,
    )
    caches["clean"] = clean
    summaries["clean"] = clean_summary
    replay, replay_summary = _extract_condition_cache(
        keeper=keeper,
        dataset=dataset,
        transform=transform,
        indices=indices[:replay_rows],
        condition="clean",
        args=args,
        device=device,
    )
    replay_feature_error = float(
        (clean["pooled"][:replay_rows] - replay["pooled"]).abs().amax().item()
    )
    replay_logit_error = float(
        (clean["raw_logits"][:replay_rows] - replay["raw_logits"]).abs().amax().item()
    )
    clean_probabilities = F.softmax(clean["raw_logits"], dim=1)
    cidt_probabilities = torch.tensor(
        [row.keeper_probabilities for row in rows], dtype=torch.float32
    )
    cidt_probability_error = float(
        (clean_probabilities - cidt_probabilities).abs().amax().item()
    )
    cidt_argmax_mismatches = int(
        torch.count_nonzero(
            clean_probabilities.argmax(dim=1)
            != torch.tensor([row.keeper_prediction for row in rows], dtype=torch.long)
        ).item()
    )
    replay_contract = {
        "rows": replay_rows,
        "pooled_maximum_absolute_error": replay_feature_error,
        "logit_maximum_absolute_error": replay_logit_error,
        "deterministic_within_1e_6": max(replay_feature_error, replay_logit_error)
        <= 1e-6,
        "sample_indices_exact": torch.equal(
            clean["sample_indices"][:replay_rows], replay["sample_indices"]
        ),
        "targets_exact": torch.equal(
            clean["targets"][:replay_rows], replay["targets"]
        ),
        "cidt_maximum_probability_error": cidt_probability_error,
        "cidt_argmax_mismatches": cidt_argmax_mismatches,
        "cidt_maximum_probability_error_limit": cidt_error_limit,
        "cidt_within_2e_3": cidt_probability_error <= 2e-3
        and cidt_argmax_mismatches == 0,
        "cidt_within_contract": cidt_probability_error <= cidt_error_limit
        and cidt_argmax_mismatches == 0,
        "protocol_contract": str(contract["contract_id"]),
        "replay_loader": replay_summary["loader"],
    }
    if not all(
        bool(replay_contract[key])
        for key in (
            "deterministic_within_1e_6",
            "sample_indices_exact",
            "targets_exact",
            "cidt_within_contract",
        )
    ):
        raise ValueError(f"Pooled-cache replay contract failed: {replay_contract}")
    del replay
    for condition in SECONDARY_CONDITIONS:
        caches[condition], summaries[condition] = _extract_condition_cache(
            keeper=keeper,
            dataset=dataset,
            transform=transform,
            indices=indices,
            condition=condition,
            args=args,
            device=device,
        )
        if not torch.equal(caches[condition]["targets"], clean["targets"]):
            raise ValueError(f"Condition {condition} changed targets.")
        if not torch.equal(
            caches[condition]["sample_indices"], clean["sample_indices"]
        ):
            raise ValueError(f"Condition {condition} changed sample order.")
    keeper.cpu()
    gc.collect()
    torch.cuda.empty_cache()
    return caches, {"conditions": summaries, "replay": replay_contract}


def _adapt_features(
    pooled: Tensor, variant: Optional[ConditionViewVariant]
) -> Tensor:
    if variant is None:
        return pooled.detach().float().cpu()
    module = copy.deepcopy(variant).cpu().eval()
    output = []
    with torch.inference_mode():
        for start in range(0, int(pooled.size(0)), 1024):
            output.append(module.adapted_feature(pooled[start : start + 1024].float()))
    return torch.cat(output).float().cpu()


def _knn_binary_statistics(
    *,
    reference_features: Tensor,
    reference_labels: Tensor,
    query_features: Tensor,
    query_labels: Tensor,
    device: torch.device,
) -> Dict[str, object]:
    reference = F.normalize(reference_features.float(), dim=1).to(device)
    reference_labels = reference_labels.to(device=device, dtype=torch.long)
    query = F.normalize(query_features.float(), dim=1)
    predictions = []
    class1_scores = []
    for start in range(0, int(query.size(0)), 256):
        values = query[start : start + 256].to(device)
        neighbors = torch.topk(
            values @ reference.T,
            k=KNN_NEIGHBORS,
            dim=1,
            largest=True,
            sorted=False,
        ).indices
        labels = reference_labels[neighbors]
        scores = labels.float().mean(dim=1)
        predictions.append((scores >= 0.5).long().cpu())
        class1_scores.append(scores.cpu())
    predicted = torch.cat(predictions)
    scores = torch.cat(class1_scores)
    labels = query_labels.long().cpu()
    aligned = predicted.eq(labels)
    class1 = labels.eq(1)
    return {
        "binary_cac": float(aligned.float().mean().item()),
        "class1_cac": float(aligned[class1].float().mean().item()),
        "nonclass1_cac": float(aligned[~class1].float().mean().item()),
        "class1_scores": scores,
        "finite": bool(torch.isfinite(scores).all().item()),
    }


def _same_sample_alignment_accuracy(
    clean_features: Tensor, condition_features: Tensor, *, device: torch.device
) -> float:
    clean = F.normalize(clean_features.float(), dim=1)
    condition = F.normalize(condition_features.float(), dim=1).to(device)
    correct = 0
    for start in range(0, int(clean.size(0)), 256):
        query = clean[start : start + 256].to(device)
        nearest = (query @ condition.T).argmax(dim=1).cpu()
        expected = torch.arange(start, start + int(query.size(0)), dtype=torch.long)
        correct += int(torch.count_nonzero(nearest == expected).item())
    return correct / float(max(1, clean.size(0)))


def _mean_off_diagonal_cosine(features: Tensor) -> float:
    values = F.normalize(features.double(), dim=1)
    count = int(values.size(0))
    if count < 2:
        return 0.0
    numerator = values.sum(dim=0).square().sum() - float(count)
    return float((numerator / float(count * (count - 1))).item())


def _representation_metrics(
    *,
    caches: Mapping[str, Mapping[str, Tensor]],
    cohorts: Mapping[str, object],
    variant: Optional[ConditionViewVariant],
    device: torch.device,
) -> Dict[str, object]:
    fit_indices = [int(value) for value in cohorts["fit_indices"]]
    holdout_indices = [int(value) for value in cohorts["holdout_indices"]]
    clean_holdout_cache = _subset_cache(caches["clean"], holdout_indices)
    raw_clean_predictions = clean_holdout_cache["raw_logits"].argmax(dim=1)
    holdout_targets = clean_holdout_cache["targets"]
    tp_mask = holdout_targets.eq(FOCUS_CLASS) & raw_clean_predictions.eq(FOCUS_CLASS)
    hard_mask = torch.isin(
        holdout_targets,
        torch.tensor(sorted(RESTRICTED_NEGATIVE_CLASSES), dtype=torch.long),
    ) & raw_clean_predictions.eq(FOCUS_CLASS)
    if int(tp_mask.sum().item()) != 107 or int(hard_mask.sum().item()) != 36:
        raise ValueError("Fixed clean TP/hard-FP cohorts differ from protocol.")
    clean_features = _adapt_features(clean_holdout_cache["pooled"], variant)
    output: Dict[str, object] = {
        "variant": "raw" if variant is None else "adapted",
        "knn_neighbors": KNN_NEIGHBORS,
        "fixed_clean_tp_rows": int(tp_mask.sum().item()),
        "fixed_clean_hard_fp_rows": int(hard_mask.sum().item()),
        "conditions": {},
    }
    for condition in CONDITIONS:
        fit_cache = _subset_cache(caches[condition], fit_indices)
        holdout_cache = _subset_cache(caches[condition], holdout_indices)
        fit_features = _adapt_features(fit_cache["pooled"], variant)
        holdout_features = _adapt_features(holdout_cache["pooled"], variant)
        fit_binary = fit_cache["targets"].eq(FOCUS_CLASS).long()
        holdout_binary = holdout_cache["targets"].eq(FOCUS_CLASS).long()
        knn = _knn_binary_statistics(
            reference_features=fit_features,
            reference_labels=fit_binary,
            query_features=holdout_features,
            query_labels=holdout_binary,
            device=device,
        )
        scores = knn.pop("class1_scores")
        tp_scores = scores[tp_mask]
        hard_scores = scores[hard_mask]
        auc_targets = np.concatenate(
            (np.ones(tp_scores.numel()), np.zeros(hard_scores.numel()))
        )
        auc_scores = np.concatenate((tp_scores.numpy(), hard_scores.numpy()))
        auc = float(roc_auc_score(auc_targets, auc_scores))
        class1_features = holdout_features[holdout_binary.eq(1)]
        nonclass1_features = holdout_features[holdout_binary.eq(0)]
        saa = (
            1.0
            if condition == "clean"
            else _same_sample_alignment_accuracy(
                clean_features, holdout_features, device=device
            )
        )
        condition_metrics = {
            **knn,
            "tp_hard_auroc": auc,
            "tp_class1_neighbor_mean": float(tp_scores.mean().item()),
            "hard_fp_class1_neighbor_mean": float(hard_scores.mean().item()),
            "tp_hard_neighbor_gap": float(
                tp_scores.mean().item() - hard_scores.mean().item()
            ),
            "clean_to_condition_saa": float(saa),
            "class1_mean_pairwise_cosine": _mean_off_diagonal_cosine(
                class1_features
            ),
            "nonclass1_mean_off_instance_cosine": _mean_off_diagonal_cosine(
                nonclass1_features
            ),
        }
        condition_metrics["finite"] = all(
            math.isfinite(float(value))
            for value in condition_metrics.values()
            if isinstance(value, (float, int))
        )
        output["conditions"][condition] = condition_metrics
        del fit_features, holdout_features
        torch.cuda.empty_cache()
    return output


def assess_pretraining_selectivity(
    metrics: Mapping[str, object]
) -> Dict[str, object]:
    checks: Dict[str, bool] = {}
    for condition in CONDITIONS:
        values = metrics["conditions"][condition]
        prefix = condition.replace("lighting_", "")
        checks[f"{prefix}_finite"] = bool(values["finite"])
        checks[f"{prefix}_tp_hard_auroc_gte_0p60"] = (
            float(values["tp_hard_auroc"]) >= 0.60
        )
        checks[f"{prefix}_tp_hard_gap_gte_0p02"] = (
            float(values["tp_hard_neighbor_gap"]) >= 0.02
        )
        checks[f"{prefix}_class1_cac_strictly_between_0p10_0p95"] = (
            0.10 < float(values["class1_cac"]) < 0.95
        )
        if condition != "clean":
            checks[f"{prefix}_saa_gte_0p85"] = (
                float(values["clean_to_condition_saa"]) >= 0.85
            )
    failed = [key for key, passed in checks.items() if not passed]
    return {
        "checks": checks,
        "failed_checks": failed,
        "all_gates_passed": not failed,
        "representation_training_authorized": not failed,
    }


def _cache_batch(
    cache: Mapping[str, Tensor], sample_indices: Sequence[int], *, key: str
) -> Tensor:
    indices = torch.as_tensor(sample_indices, dtype=torch.long)
    if cache["sample_indices"].tolist() == list(range(int(cache["sample_indices"].numel()))):
        return cache[key][indices]
    positions = _cache_positions(cache)
    rows = positions[indices]
    if bool((rows < 0).any().item()):
        raise ValueError("Training schedule references a missing cache row.")
    return cache[key][rows]


def _secondary_feature_batch(
    caches: Mapping[str, Mapping[str, Tensor]],
    sample_indices: Sequence[int],
    *,
    epoch: int,
) -> tuple[Tensor, Dict[str, int]]:
    output = torch.empty(len(sample_indices), EMBED_DIM, dtype=torch.float32)
    counts = {condition: 0 for condition in SECONDARY_CONDITIONS}
    grouped: Dict[str, list[tuple[int, int]]] = {
        condition: [] for condition in SECONDARY_CONDITIONS
    }
    for position, sample_index in enumerate(sample_indices):
        condition = locked_secondary_condition(
            epoch=epoch, sample_index=int(sample_index)
        )
        grouped[condition].append((position, int(sample_index)))
        counts[condition] += 1
    for condition, entries in grouped.items():
        if not entries:
            continue
        positions, indices = zip(*entries)
        output[torch.tensor(positions, dtype=torch.long)] = _cache_batch(
            caches[condition], indices, key="pooled"
        )
    return output, counts


def _gradient_contract(
    module: nn.Module,
    names: Iterable[str],
    seen: Dict[str, bool],
) -> tuple[bool, float]:
    parameters = dict(module.named_parameters())
    finite = True
    squared = 0.0
    for name in names:
        gradient = parameters[name].grad
        if gradient is None:
            continue
        finite = finite and bool(torch.isfinite(gradient).all().item())
        seen[name] = seen[name] or bool(torch.count_nonzero(gradient).item())
        squared += float(gradient.detach().float().square().sum().item())
    return finite, math.sqrt(squared)


def _train_representation_variant(
    *,
    name: str,
    prototype: ConditionViewVariant,
    caches: Mapping[str, Mapping[str, Tensor]],
    fit_indices: Sequence[int],
    device: torch.device,
    supervised_minority: bool,
) -> tuple[ConditionViewVariant, Dict[str, object]]:
    variant = copy.deepcopy(prototype).to(device).train()
    representation_names = [
        name
        for name, parameter in variant.named_parameters()
        if any(parameter is candidate for candidate in variant.representation_parameters())
    ]
    for parameter_name, parameter in variant.named_parameters():
        parameter.requires_grad_(parameter_name in representation_names)
    optimizer = torch.optim.SGD(
        [dict(variant.named_parameters())[key] for key in representation_names],
        lr=STAGE1_LEARNING_RATE,
        momentum=STAGE1_MOMENTUM,
        weight_decay=STAGE1_WEIGHT_DECAY,
    )
    history: list[Dict[str, object]] = []
    gradient_seen = {key: False for key in representation_names}
    all_gradients_finite = True
    all_losses_finite = True
    occurrences = 0
    secondary_counts = {condition: 0 for condition in SECONDARY_CONDITIONS}
    initial_state_sha = _state_sha256(variant)
    started = time.perf_counter()
    for epoch in range(CONTRASTIVE_EPOCHS):
        learning_rate = _stage1_learning_rate(epoch)
        for group in optimizer.param_groups:
            group["lr"] = learning_rate
        order = locked_contrastive_epoch_order(fit_indices, epoch=epoch)
        epoch_loss = 0.0
        epoch_gradient = 0.0
        epoch_rows = 0
        epoch_batches = 0
        positive_summary: Optional[Dict[str, object]] = None
        for start in range(0, len(order), CONTRASTIVE_BATCH_SIZE):
            indices = [int(value) for value in order[start : start + CONTRASTIVE_BATCH_SIZE]]
            clean = _cache_batch(caches["clean"], indices, key="pooled").to(device)
            secondary, counts = _secondary_feature_batch(
                caches, indices, epoch=epoch
            )
            secondary = secondary.to(device)
            targets = _cache_batch(caches["clean"], indices, key="targets")
            binary = targets.eq(FOCUS_CLASS).long().to(device)
            optimizer.zero_grad(set_to_none=True)
            projected = torch.stack(
                (
                    variant.projected_feature(clean),
                    variant.projected_feature(secondary),
                ),
                dim=1,
            )
            loss, positive_summary = supervised_contrastive_loss(
                projected,
                binary,
                supervised_minority=supervised_minority,
            )
            all_losses_finite = all_losses_finite and bool(torch.isfinite(loss).item())
            if not all_losses_finite:
                raise ValueError(f"Non-finite {name} contrastive loss.")
            loss.backward()
            finite, gradient_norm = _gradient_contract(
                variant, representation_names, gradient_seen
            )
            all_gradients_finite = all_gradients_finite and finite
            optimizer.step()
            count = len(indices)
            epoch_loss += float(loss.detach().item()) * count
            epoch_gradient += gradient_norm
            epoch_rows += count
            epoch_batches += 1
            occurrences += count
            for condition, value in counts.items():
                secondary_counts[condition] += int(value)
        history.append(
            {
                "stage": "representation",
                "variant": name,
                "epoch": epoch + 1,
                "learning_rate": learning_rate,
                "loss": epoch_loss / float(epoch_rows),
                "mean_batch_gradient_norm": epoch_gradient / float(epoch_batches),
                "rows": epoch_rows,
                "batches": epoch_batches,
            }
        )
        print(json.dumps(history[-1]), flush=True)
    final_state_sha = _state_sha256(variant)
    variant = variant.cpu().eval()
    gc.collect()
    torch.cuda.empty_cache()
    return variant, {
        "name": name,
        "history": history,
        "epochs": len(history),
        "occurrences": occurrences,
        "secondary_counts": secondary_counts,
        "schedule_sha256": contrastive_schedule_sha256(fit_indices),
        "initial_state_sha256": initial_state_sha,
        "final_state_sha256": final_state_sha,
        "state_changed": initial_state_sha != final_state_sha,
        "gradient_seen_nonzero": gradient_seen,
        "all_trainable_gradients_seen": all(gradient_seen.values()),
        "all_gradients_finite": all_gradients_finite,
        "all_losses_finite": all_losses_finite,
        "positive_mask_last_batch": positive_summary,
        "first_epoch_mean_loss": float(history[0]["loss"]),
        "final_epoch_mean_loss": float(history[-1]["loss"]),
        "final_to_first_loss_ratio": float(history[-1]["loss"])
        / max(1e-12, float(history[0]["loss"])),
        "elapsed_seconds": float(time.perf_counter() - started),
    }


def _representation_projection_diagnostics(
    variant: ConditionViewVariant,
    pooled: Tensor,
) -> Dict[str, object]:
    module = copy.deepcopy(variant).cpu().eval()
    adapted_rows = []
    projected_rows = []
    residual_rows = []
    with torch.inference_mode():
        for start in range(0, int(pooled.size(0)), 1024):
            values = pooled[start : start + 1024].float()
            adapted = module.adapted_feature(values)
            adapted_rows.append(adapted)
            residual_rows.append((adapted - values) / ADAPTER_RESIDUAL_SCALE)
            projected_rows.append(module.projected_feature(values))
    adapted = torch.cat(adapted_rows)
    projected = torch.cat(projected_rows)
    residual = torch.cat(residual_rows)
    variance = projected.var(dim=0, unbiased=False)
    singular_values = torch.linalg.svdvals(
        projected.double() - projected.double().mean(dim=0, keepdim=True)
    )
    energy = singular_values.square()
    probabilities = energy / energy.sum().clamp_min(1e-18)
    effective_rank = float(torch.exp(-(probabilities * probabilities.clamp_min(1e-18).log()).sum()).item())
    return {
        "adapted_shape": list(adapted.shape),
        "projected_shape": list(projected.shape),
        "adapter_residual_rms": float(residual.square().mean().sqrt().item()),
        "projection_variance_min": float(variance.min().item()),
        "projection_variance_mean": float(variance.mean().item()),
        "projection_variance_max": float(variance.max().item()),
        "projection_all_dimensions_finite_variance": bool(
            torch.isfinite(variance).all().item()
        ),
        "projection_positive_variance_dimensions": int(
            torch.count_nonzero(variance > 0.0).item()
        ),
        "projection_effective_rank": effective_rank,
        "finite": bool(
            torch.isfinite(adapted).all().item()
            and torch.isfinite(projected).all().item()
            and torch.isfinite(residual).all().item()
        ),
    }


def _train_all_representations(
    *,
    variants: Mapping[str, ConditionViewVariant],
    caches: Mapping[str, Mapping[str, Tensor]],
    fit_indices: Sequence[int],
    device: torch.device,
) -> tuple[Dict[str, ConditionViewVariant], Dict[str, object]]:
    initial_representation_hash = _state_sha256(variants["ce_identity"])
    trained: Dict[str, ConditionViewVariant] = {
        "ce_identity": copy.deepcopy(variants["ce_identity"]).cpu().eval()
    }
    summaries: Dict[str, object] = {
        "ce_identity": {
            "name": "ce_identity",
            "representation_stage_skipped": True,
            "detached_schedule_occurrences": EXPECTED_CONTRASTIVE_OCCURRENCES,
            "schedule_sha256": contrastive_schedule_sha256(fit_indices),
            "initial_state_sha256": initial_representation_hash,
            "final_state_sha256": _state_sha256(trained["ce_identity"]),
            "state_bit_exact": _state_sha256(trained["ce_identity"])
            == initial_representation_hash,
        }
    }
    for name, supervised_minority in (
        ("standard_supcon", False),
        ("supervised_minority", True),
    ):
        trained[name], summaries[name] = _train_representation_variant(
            name=name,
            prototype=variants[name],
            caches=caches,
            fit_indices=fit_indices,
            device=device,
            supervised_minority=supervised_minority,
        )
    fit_pooled = _cache_batch(caches["clean"], fit_indices, key="pooled")
    for name in trained:
        summaries[name]["projection_diagnostics"] = (
            _representation_projection_diagnostics(trained[name], fit_pooled)
        )
    return trained, summaries


def _train_probe_variant(
    *,
    name: str,
    prototype: ConditionViewVariant,
    clean_cache: Mapping[str, Tensor],
    fit_indices: Sequence[int],
    device: torch.device,
) -> tuple[ConditionViewVariant, Dict[str, object]]:
    variant = copy.deepcopy(prototype).to(device).eval()
    for parameter in variant.parameters():
        parameter.requires_grad_(False)
    for parameter in variant.probe.parameters():
        parameter.requires_grad_(True)
    variant.probe.train()
    representation_before = {
        key: value.detach().cpu().clone()
        for key, value in variant.state_dict().items()
        if not key.startswith("probe.")
    }
    initial_probe_sha = _state_sha256(variant.probe)
    optimizer = torch.optim.Adam(
        variant.probe.parameters(),
        lr=STAGE2_LEARNING_RATE,
        weight_decay=STAGE2_WEIGHT_DECAY,
    )
    gradient_seen = {key: False for key, _value in variant.probe.named_parameters()}
    all_gradients_finite = True
    all_losses_finite = True
    occurrences = 0
    history: list[Dict[str, object]] = []
    started = time.perf_counter()
    for epoch in range(PROBE_EPOCHS):
        order = locked_probe_epoch_order(fit_indices, epoch=epoch)
        epoch_loss = 0.0
        epoch_gradient = 0.0
        epoch_rows = 0
        epoch_batches = 0
        for start in range(0, len(order), PROBE_BATCH_SIZE):
            indices = [int(value) for value in order[start : start + PROBE_BATCH_SIZE]]
            pooled = _cache_batch(clean_cache, indices, key="pooled").to(device)
            raw_logits = _cache_batch(clean_cache, indices, key="raw_logits").to(device)
            targets = _cache_batch(clean_cache, indices, key="targets").to(device)
            optimizer.zero_grad(set_to_none=True)
            logits = variant.deployed_logits(pooled, raw_logits)
            loss = F.cross_entropy(logits, targets)
            all_losses_finite = all_losses_finite and bool(torch.isfinite(loss).item())
            if not all_losses_finite:
                raise ValueError(f"Non-finite {name} residual-probe loss.")
            loss.backward()
            finite, gradient_norm = _gradient_contract(
                variant.probe,
                gradient_seen.keys(),
                gradient_seen,
            )
            all_gradients_finite = all_gradients_finite and finite
            optimizer.step()
            count = len(indices)
            occurrences += count
            epoch_rows += count
            epoch_batches += 1
            epoch_loss += float(loss.detach().item()) * count
            epoch_gradient += gradient_norm
        history.append(
            {
                "stage": "probe",
                "variant": name,
                "epoch": epoch + 1,
                "learning_rate": STAGE2_LEARNING_RATE,
                "loss": epoch_loss / float(epoch_rows),
                "mean_batch_gradient_norm": epoch_gradient / float(epoch_batches),
                "rows": epoch_rows,
                "batches": epoch_batches,
            }
        )
        print(json.dumps(history[-1]), flush=True)
    final_representation = {
        key: value.detach().cpu()
        for key, value in variant.state_dict().items()
        if not key.startswith("probe.")
    }
    representation_exact = all(
        torch.equal(value, final_representation[key])
        for key, value in representation_before.items()
    )
    variant = variant.cpu().eval()
    gc.collect()
    torch.cuda.empty_cache()
    return variant, {
        "name": name,
        "history": history,
        "epochs": len(history),
        "occurrences": occurrences,
        "schedule_sha256": probe_schedule_sha256(fit_indices),
        "initial_probe_sha256": initial_probe_sha,
        "final_probe_sha256": _state_sha256(variant.probe),
        "probe_changed": initial_probe_sha != _state_sha256(variant.probe),
        "representation_state_bit_exact": representation_exact,
        "gradient_seen_nonzero": gradient_seen,
        "all_trainable_gradients_seen": all(gradient_seen.values()),
        "all_gradients_finite": all_gradients_finite,
        "all_losses_finite": all_losses_finite,
        "elapsed_seconds": float(time.perf_counter() - started),
    }


def _train_all_probes(
    *,
    variants: Mapping[str, ConditionViewVariant],
    clean_cache: Mapping[str, Tensor],
    fit_indices: Sequence[int],
    device: torch.device,
) -> tuple[Dict[str, ConditionViewVariant], Dict[str, object]]:
    trained: Dict[str, ConditionViewVariant] = {}
    summaries: Dict[str, object] = {}
    for name in ("ce_identity", "standard_supcon", "supervised_minority"):
        trained[name], summaries[name] = _train_probe_variant(
            name=name,
            prototype=variants[name],
            clean_cache=clean_cache,
            fit_indices=fit_indices,
            device=device,
        )
    return trained, summaries


def _prediction_row(
    *, sample_index: int, target: int, probabilities: Tensor
) -> Dict[str, object]:
    values = probabilities.detach().float().cpu()
    row: Dict[str, object] = {
        "sample_index": int(sample_index),
        "target": int(target),
        "prediction": int(values.argmax().item()),
    }
    for class_index in range(NUM_CLASSES):
        row[f"prob_{class_index}"] = float(values[class_index].item())
    return row


def _evaluate_variants(
    *,
    caches: Mapping[str, Mapping[str, Tensor]],
    variants: Mapping[str, ConditionViewVariant],
    holdout_indices: Sequence[int],
    device: torch.device,
) -> Dict[str, Dict[str, list[Dict[str, object]]]]:
    output = {
        condition: {
            name: []
            for name in ("raw", "ce_identity", "standard_supcon", "supervised_minority")
        }
        for condition in CONDITIONS
    }
    device_variants = {
        name: copy.deepcopy(variant).to(device).eval()
        for name, variant in variants.items()
    }
    for condition in CONDITIONS:
        cache = _subset_cache(caches[condition], holdout_indices)
        for start in range(0, len(holdout_indices), 1024):
            stop = min(start + 1024, len(holdout_indices))
            pooled = cache["pooled"][start:stop].to(device)
            raw_logits = cache["raw_logits"][start:stop].to(device)
            targets = cache["targets"][start:stop]
            indices = cache["sample_indices"][start:stop]
            with torch.inference_mode():
                probabilities = {"raw": F.softmax(raw_logits.float(), dim=1)}
                for name, variant in device_variants.items():
                    probabilities[name] = F.softmax(
                        variant.deployed_logits(pooled, raw_logits).float(), dim=1
                    )
            for local in range(stop - start):
                for name, values in probabilities.items():
                    output[condition][name].append(
                        _prediction_row(
                            sample_index=int(indices[local].item()),
                            target=int(targets[local].item()),
                            probabilities=values[local],
                        )
                    )
        expected = [int(value) for value in holdout_indices]
        for name, values in output[condition].items():
            if [int(row["sample_index"]) for row in values] != expected:
                raise ValueError(f"Evaluation order differs for {condition}/{name}.")
    del device_variants
    gc.collect()
    torch.cuda.empty_cache()
    return output


def _build_comparisons(
    predictions: Mapping[str, Mapping[str, Sequence[Mapping[str, object]]]]
) -> Dict[str, Dict[str, object]]:
    comparisons: Dict[str, Dict[str, object]] = {}
    for condition in CONDITIONS:
        values = predictions[condition]
        candidate = values["supervised_minority"]
        comparisons[condition] = {
            "raw_candidate": _comparison(
                control_rows=values["raw"],
                candidate_rows=candidate,
                num_classes=NUM_CLASSES,
                focus_class=FOCUS_CLASS,
            ),
            "ce_candidate": _comparison(
                control_rows=values["ce_identity"],
                candidate_rows=candidate,
                num_classes=NUM_CLASSES,
                focus_class=FOCUS_CLASS,
            ),
            "standard_candidate": _comparison(
                control_rows=values["standard_supcon"],
                candidate_rows=candidate,
                num_classes=NUM_CLASSES,
                focus_class=FOCUS_CLASS,
            ),
            "raw_ce": _comparison(
                control_rows=values["raw"],
                candidate_rows=values["ce_identity"],
                num_classes=NUM_CLASSES,
                focus_class=FOCUS_CLASS,
            ),
            "raw_standard": _comparison(
                control_rows=values["raw"],
                candidate_rows=values["standard_supcon"],
                num_classes=NUM_CLASSES,
                focus_class=FOCUS_CLASS,
            ),
        }
    return comparisons


def assess_representation_mechanism(
    *,
    representations: Mapping[str, Mapping[str, object]],
    representation_training: Mapping[str, object],
) -> Dict[str, object]:
    raw = representations["raw"]["conditions"]
    standard = representations["standard_supcon"]["conditions"]
    candidate = representations["supervised_minority"]["conditions"]
    checks: Dict[str, bool] = {}
    for condition in CONDITIONS:
        prefix = condition.replace("lighting_", "")
        checks[f"{prefix}_class1_cac_gain_gte_0p02"] = (
            float(candidate[condition]["class1_cac"])
            - float(raw[condition]["class1_cac"])
            >= 0.02
        )
        checks[f"{prefix}_tp_hard_auroc_gain_gte_0p02"] = (
            float(candidate[condition]["tp_hard_auroc"])
            - float(raw[condition]["tp_hard_auroc"])
            >= 0.02
        )
        checks[f"{prefix}_auroc_gte_standard"] = (
            float(candidate[condition]["tp_hard_auroc"])
            >= float(standard[condition]["tp_hard_auroc"])
        )
        checks[f"{prefix}_nonclass1_cac_drop_lte_0p02"] = (
            float(candidate[condition]["nonclass1_cac"])
            - float(raw[condition]["nonclass1_cac"])
            >= -0.02
        )
        checks[f"{prefix}_saa_drop_lte_0p02"] = (
            float(candidate[condition]["clean_to_condition_saa"])
            - float(raw[condition]["clean_to_condition_saa"])
            >= -0.02
        )
        checks[f"{prefix}_class1_pairwise_cosine_increased"] = (
            float(candidate[condition]["class1_mean_pairwise_cosine"])
            > float(raw[condition]["class1_mean_pairwise_cosine"])
        )
        checks[f"{prefix}_nonclass1_cosine_gain_lte_0p02"] = (
            float(candidate[condition]["nonclass1_mean_off_instance_cosine"])
            - float(raw[condition]["nonclass1_mean_off_instance_cosine"])
            <= 0.02
        )
    training = representation_training["supervised_minority"]
    diagnostics = training["projection_diagnostics"]
    checks.update(
        {
            "supmin_final_loss_lte_90pct_first": float(
                training["final_to_first_loss_ratio"]
            )
            <= 0.90,
            "adapter_residual_rms_nonzero": float(
                diagnostics["adapter_residual_rms"]
            )
            > 0.0,
            "projection_variance_all_finite": bool(
                diagnostics["projection_all_dimensions_finite_variance"]
            ),
            "projection_effective_rank_gte_16": float(
                diagnostics["projection_effective_rank"]
            )
            >= 16.0,
        }
    )
    failed = [key for key, passed in checks.items() if not passed]
    return {
        "checks": checks,
        "failed_checks": failed,
        "all_gates_passed": not failed,
    }


def assess_train_holdout_behavior(
    comparisons: Mapping[str, Mapping[str, Mapping[str, object]]]
) -> Dict[str, object]:
    checks: Dict[str, bool] = {}
    clean_raw = comparisons["clean"]["raw_candidate"]
    delta = clean_raw["delta"]
    transitions = clean_raw["transitions"]
    checks.update(
        {
            "clean_macro_delta_gte_minus_0p002": float(delta["macro_f1"]) >= -0.002,
            "clean_class1_f1_gain_gte_0p005": float(delta["class1_f1"]) >= 0.005,
            "clean_class1_precision_gain_gte_0p010": float(
                delta["class1_precision"]
            )
            >= 0.010,
            "clean_class1_recall_delta_gte_minus_0p005": float(
                delta["class1_recall"]
            )
            >= -0.005,
            "clean_tp_breaks_lte_1": int(transitions["focus_tp_break"]) <= 1,
            "clean_fn_rescues_gte_tp_breaks": int(transitions["focus_fn_rescue"])
            >= int(transitions["focus_tp_break"]),
            "clean_restricted_fp_reduction_gte_3": int(
                transitions["restricted_focus_fp_reduction"]
            )
            >= 3,
            "clean_corrections_gt_harms": int(transitions["candidate_correction"])
            > int(transitions["candidate_harm"]),
            "clean_max_nonclass1_f1_drop_lte_0p010": float(
                clean_raw["maximum_nonfocus_f1_drop"]
            )
            <= 0.010,
        }
    )
    for comparator in ("ce", "standard"):
        values = comparisons["clean"][f"{comparator}_candidate"]
        delta = values["delta"]
        transitions = values["transitions"]
        checks.update(
            {
                f"clean_macro_delta_vs_{comparator}_nonnegative": float(
                    delta["macro_f1"]
                )
                >= 0.0,
                f"clean_class1_f1_gain_vs_{comparator}_gte_0p003": float(
                    delta["class1_f1"]
                )
                >= 0.003,
                f"clean_class1_precision_gain_vs_{comparator}_gte_0p003": float(
                    delta["class1_precision"]
                )
                >= 0.003,
                f"clean_class1_recall_delta_vs_{comparator}_gte_minus_0p002": float(
                    delta["class1_recall"]
                )
                >= -0.002,
                f"clean_restricted_fp_not_worse_vs_{comparator}": int(
                    transitions["restricted_focus_fp_reduction"]
                )
                >= 0,
                f"clean_corrections_gte_harms_vs_{comparator}": int(
                    transitions["candidate_correction"]
                )
                >= int(transitions["candidate_harm"]),
            }
        )
    shifted_improvements = 0
    for condition in SECONDARY_CONDITIONS:
        prefix = condition.replace("lighting_", "")
        values = comparisons[condition]["raw_candidate"]
        delta = values["delta"]
        transitions = values["transitions"]
        checks.update(
            {
                f"{prefix}_class1_f1_delta_gte_minus_0p005": float(
                    delta["class1_f1"]
                )
                >= -0.005,
                f"{prefix}_class1_precision_delta_nonnegative": float(
                    delta["class1_precision"]
                )
                >= 0.0,
                f"{prefix}_class1_recall_delta_gte_minus_0p010": float(
                    delta["class1_recall"]
                )
                >= -0.010,
                f"{prefix}_restricted_fp_not_worse": int(
                    transitions["restricted_focus_fp_reduction"]
                )
                >= 0,
                f"{prefix}_tp_breaks_lte_2": int(transitions["focus_tp_break"])
                <= 2,
                f"{prefix}_fn_rescues_gte_tp_breaks": int(
                    transitions["focus_fn_rescue"]
                )
                >= int(transitions["focus_tp_break"]),
                f"{prefix}_corrections_gte_harms": int(
                    transitions["candidate_correction"]
                )
                >= int(transitions["candidate_harm"]),
            }
        )
        shifted_improvements += int(float(delta["class1_f1"]) >= 0.003)
    checks["at_least_two_shifted_class1_f1_gains_gte_0p003"] = (
        shifted_improvements >= 2
    )
    failed = [key for key, passed in checks.items() if not passed]
    return {
        "checks": checks,
        "shifted_conditions_improved": shifted_improvements,
        "failed_checks": failed,
        "all_gates_passed": not failed,
    }


def official_ttc_equation_replay(
    features: Tensor,
    labels: Tensor,
    *,
    supervised_minority: bool,
) -> Tensor:
    if features.ndim != 3 or int(features.size(1)) != 2:
        raise ValueError("Official TTC replay expects [batch,2,dim].")
    normalized = F.normalize(features.float(), dim=-1)
    batch = int(normalized.size(0))
    if supervised_minority:
        mask = torch.eye(batch, dtype=torch.float32, device=features.device)
        minority = labels.eq(FOCUS_CLASS)
        minority_indices = torch.where(minority)[0]
        mask[minority_indices.reshape(-1, 1), minority_indices] = 1.0
    else:
        mask = labels[:, None].eq(labels[None, :]).float()
    contrast = torch.cat(torch.unbind(normalized, dim=1), dim=0)
    logits = (contrast @ contrast.T) / TEMPERATURE
    logits = logits - logits.max(dim=1, keepdim=True).values.detach()
    mask = mask.repeat(2, 2)
    logits_mask = torch.scatter(
        torch.ones_like(mask),
        1,
        torch.arange(2 * batch, device=features.device).reshape(-1, 1),
        0,
    )
    mask = mask * logits_mask
    exp_logits = torch.exp(logits) * logits_mask
    log_prob = logits - torch.log(exp_logits.sum(dim=1, keepdim=True))
    mean_log_prob_pos = (mask * log_prob).sum(dim=1) / mask.sum(dim=1).clamp_min(1e-8)
    return (
        -(TEMPERATURE / BASE_TEMPERATURE) * mean_log_prob_pos
    ).reshape(2, batch).mean()


def _equation_diagnostics() -> Dict[str, object]:
    generator = torch.Generator().manual_seed(SEED + 707)
    features = torch.randn(11, 2, 17, generator=generator)
    labels = torch.tensor([0, 1, 0, 0, 1, 0, 0, 0, 1, 0, 0], dtype=torch.long)
    modes: Dict[str, object] = {}
    maximum_error = 0.0
    for name, supervised_minority in (
        ("standard_supcon", False),
        ("supervised_minority", True),
    ):
        local, mask_summary = supervised_contrastive_loss(
            features,
            labels,
            supervised_minority=supervised_minority,
        )
        independent = independent_contrastive_loss(
            features,
            labels,
            supervised_minority=supervised_minority,
        )
        official = official_ttc_equation_replay(
            features,
            labels,
            supervised_minority=supervised_minority,
        )
        local_independent = float((local - independent).abs().item())
        local_official = float((local - official).abs().item())
        maximum_error = max(maximum_error, local_independent, local_official)
        modes[name] = {
            "local_loss": float(local.item()),
            "independent_loss": float(independent.item()),
            "official_source_equation_loss": float(official.item()),
            "local_independent_absolute_error": local_independent,
            "local_official_absolute_error": local_official,
            "positive_mask": mask_summary,
        }
    supmin_mask = _positive_mask(labels, supervised_minority=True).repeat(2, 2)
    supmin_mask.fill_diagonal_(0.0)
    anchors = labels.repeat(2)
    expected_class1 = 2 * int(labels.eq(FOCUS_CLASS).sum().item()) - 1
    return {
        "modes": modes,
        "maximum_absolute_error": maximum_error,
        "matches_independent_and_official_within_1e_6": maximum_error <= 1e-6,
        "supmin_nonclass1_positive_count_exactly_one": bool(
            torch.equal(
                supmin_mask.sum(dim=1)[anchors.ne(FOCUS_CLASS)],
                torch.ones_like(supmin_mask.sum(dim=1)[anchors.ne(FOCUS_CLASS)]),
            )
        ),
        "supmin_class1_positive_count_exact": bool(
            torch.equal(
                supmin_mask.sum(dim=1)[anchors.eq(FOCUS_CLASS)],
                torch.full_like(
                    supmin_mask.sum(dim=1)[anchors.eq(FOCUS_CLASS)],
                    float(expected_class1),
                ),
            )
        ),
    }


def _metadata_to_device(
    metadata: Mapping[str, object], *, device: torch.device, count: int
) -> Dict[str, object]:
    return {
        key: value[:count].to(device=device, non_blocking=True)
        if torch.is_tensor(value)
        else value
        for key, value in metadata.items()
    }


def _repeat_metadata(
    metadata: Mapping[str, object], *, count: int, device: torch.device
) -> Dict[str, object]:
    output: Dict[str, object] = {}
    for key, value in metadata.items():
        if torch.is_tensor(value):
            source = value[:1].to(device=device)
            repeats = (int(count),) + (1,) * (source.ndim - 1)
            output[key] = source.repeat(repeats)
        else:
            output[key] = value
    return output


class _IsolatedCandidateExport(nn.Module):
    def __init__(self, variant: ConditionViewVariant) -> None:
        super().__init__()
        copied = copy.deepcopy(variant).cpu().eval()
        self.adapter_norm = copied.adapter_norm
        self.adapter_down = copied.adapter_down
        self.adapter_up = copied.adapter_up
        self.probe = copied.probe

    def forward(self, pooled: Tensor, raw_logits: Tensor) -> Tensor:
        residual = self.adapter_up(
            F.gelu(self.adapter_down(self.adapter_norm(pooled)))
        )
        adapted = pooled + ADAPTER_RESIDUAL_SCALE * residual
        return raw_logits + PROBE_RESIDUAL_SCALE * self.probe(adapted)


class _FullCandidateExport(nn.Module):
    def __init__(self, keeper: nn.Module, variant: ConditionViewVariant) -> None:
        super().__init__()
        self.keeper = copy.deepcopy(keeper).cpu().eval()
        self.variant = _IsolatedCandidateExport(variant)

    def forward(self, images: Tensor, bbox: Tensor, image_mask: Tensor) -> Tensor:
        features = self.keeper.forward_features(
            images,
            bbox_token_prior=bbox,
            image_valid_mask=image_mask,
        )
        features["bbox"] = bbox
        raw_logits = classification_logits_from_features(self.keeper, features)
        return F.softmax(self.variant(features["pooled"], raw_logits), dim=1)


def _onnx_graph_diagnostics(path: Path) -> Dict[str, object]:
    import onnx

    model = onnx.load(str(path))
    names = [
        str(value.name)
        for value in (
            list(model.graph.node)
            + list(model.graph.initializer)
            + list(model.graph.input)
            + list(model.graph.output)
        )
    ]
    projector_names = [name for name in names if "projector" in name.casefold()]
    return {
        "node_count": len(model.graph.node),
        "initializer_count": len(model.graph.initializer),
        "projector_names": projector_names,
        "training_projector_absent": not projector_names,
    }


def _export_diagnostics(
    *,
    keeper: nn.Module,
    candidate: ConditionViewVariant,
    clean_cache: Mapping[str, Tensor],
    resource_batch: tuple[Tensor, Tensor, Mapping[str, object]],
    output_dir: Path,
) -> Dict[str, object]:
    isolated_path = output_dir / "supervised_minority_adapter_probe.onnx"
    try:
        isolated = _onnx_compare(
            wrapper=_IsolatedCandidateExport(candidate),
            inputs=(
                clean_cache["pooled"][:1].detach().clone().cpu(),
                clean_cache["raw_logits"][:1].detach().clone().cpu(),
            ),
            input_names=("pooled", "raw_logits"),
            path=isolated_path,
        )
        isolated["graph"] = _onnx_graph_diagnostics(isolated_path)
    except Exception as error:
        isolated = _failed_export(isolated_path, error)
        isolated["graph"] = {
            "projector_names": [],
            "training_projector_absent": False,
        }
    images, _targets, metadata = resource_batch
    images = images[:1].detach().float().cpu()
    bbox = metadata.get("bbox")
    if not torch.is_tensor(bbox):
        bbox = torch.zeros(1, 4)
    bbox = bbox[:1].detach().float().cpu()
    image_mask = metadata.get("image_mask")
    if not torch.is_tensor(image_mask):
        image_mask = torch.ones(
            1, int(images.size(-2)), int(images.size(-1)), dtype=torch.bool
        )
    image_mask = image_mask[:1].detach().bool().cpu()
    full_path = output_dir / "supervised_minority_full_candidate.onnx"
    try:
        full = _onnx_compare(
            wrapper=_FullCandidateExport(keeper, candidate),
            inputs=(images, bbox, image_mask),
            input_names=("images", "bbox", "image_mask"),
            path=full_path,
        )
        full["comparison_space"] = "probability"
        full["graph"] = _onnx_graph_diagnostics(full_path)
    except Exception as error:
        full = _failed_export(full_path, error)
        full["comparison_space"] = "probability"
        full["graph"] = {
            "projector_names": [],
            "training_projector_absent": False,
        }
    return {"isolated": isolated, "full": full, "python_fallback": False}


def _benchmark_inference(
    *,
    keeper: nn.Module,
    variant: Optional[ConditionViewVariant],
    images_cpu: Tensor,
    metadata_cpu: Mapping[str, object],
    device: torch.device,
    repeats: int = 20,
) -> Dict[str, object]:
    gc.collect()
    torch.cuda.empty_cache()
    keeper = keeper.to(device).eval()
    variant_device = (
        copy.deepcopy(variant).to(device).eval() if variant is not None else None
    )
    if int(images_cpu.size(0)) < 32:
        raise ValueError("Runtime benchmark requires a complete batch of 32.")
    images = images_cpu[:32].to(device)
    metadata = _metadata_to_device(metadata_cpu, device=device, count=32)

    def iteration() -> Tensor:
        with torch.inference_mode(), torch.autocast(
            device_type="cuda", dtype=torch.float16, enabled=True
        ):
            raw_logits, features = _forward_classification_with_metadata(
                keeper, images, metadata, device=device
            )
        if variant_device is None:
            return raw_logits.float()
        if not isinstance(features, Mapping) or not torch.is_tensor(
            features.get("pooled")
        ):
            raise ValueError("Runtime benchmark has no pooled feature.")
        return variant_device.deployed_logits(
            features["pooled"].float(), raw_logits.float()
        )

    for _ in range(5):
        logits = iteration()
    torch.cuda.synchronize(device)
    torch.cuda.reset_peak_memory_stats(device)
    elapsed = []
    for _ in range(int(repeats)):
        torch.cuda.synchronize(device)
        started = time.perf_counter()
        logits = iteration()
        torch.cuda.synchronize(device)
        elapsed.append(float(time.perf_counter() - started))
    result = {
        "seconds": elapsed,
        "median_seconds": float(statistics.median(elapsed)),
        "peak_vram_gib": float(torch.cuda.max_memory_allocated(device) / (1024**3)),
        "logits_finite": bool(torch.isfinite(logits).all().item()),
        "batch_size": 32,
        "repeats": int(repeats),
    }
    del variant_device, images, logits
    keeper.cpu()
    gc.collect()
    torch.cuda.empty_cache()
    return result


def _resource_diagnostics(
    *,
    keeper: nn.Module,
    candidate: ConditionViewVariant,
    resource_batch: tuple[Tensor, Tensor, Mapping[str, object]],
    device: torch.device,
) -> Dict[str, object]:
    images, _targets, metadata = resource_batch
    raw = _benchmark_inference(
        keeper=keeper,
        variant=None,
        images_cpu=images,
        metadata_cpu=metadata,
        device=device,
    )
    candidate_result = _benchmark_inference(
        keeper=keeper,
        variant=candidate,
        images_cpu=images,
        metadata_cpu=metadata,
        device=device,
    )
    runtime_ratio = float(candidate_result["median_seconds"]) / max(
        1e-12, float(raw["median_seconds"])
    )
    incremental = max(
        0.0,
        float(candidate_result["peak_vram_gib"]) - float(raw["peak_vram_gib"]),
    )
    return {
        "raw": raw,
        "candidate": candidate_result,
        "runtime_ratio": runtime_ratio,
        "incremental_peak_vram_gib": incremental,
        "checks": {
            "runtime_ratio_lte_1p10": runtime_ratio <= 1.10,
            "incremental_peak_vram_lte_0p25_gib": incremental <= 0.25,
            "full_peak_vram_lte_4_gib": float(candidate_result["peak_vram_gib"])
            <= 4.0,
            "outputs_finite": bool(raw["logits_finite"])
            and bool(candidate_result["logits_finite"]),
        },
    }


def _required_xai_events(
    predictions: Mapping[str, Mapping[str, Sequence[Mapping[str, object]]]],
    *,
    minimum_rows: int = 12,
) -> tuple[list[tuple[str, int]], Dict[str, list[str]], set[str]]:
    categories: Dict[str, list[str]] = {}

    def add(condition: str, sample_index: int, category: str) -> None:
        event_id = f"{condition}:{int(sample_index)}"
        categories.setdefault(event_id, [])
        if category not in categories[event_id]:
            categories[event_id].append(category)

    for condition in CONDITIONS:
        raw_rows = predictions[condition]["raw"]
        candidate_rows = predictions[condition]["supervised_minority"]
        for raw, candidate in zip(raw_rows, candidate_rows):
            sample_index = int(raw["sample_index"])
            target = int(raw["target"])
            raw_prediction = int(raw["prediction"])
            candidate_prediction = int(candidate["prediction"])
            if (
                condition == "clean"
                and raw_prediction != candidate_prediction
                and (
                    target == FOCUS_CLASS
                    or raw_prediction == FOCUS_CLASS
                    or candidate_prediction == FOCUS_CLASS
                )
            ):
                add(condition, sample_index, "clean_class1_related_change")
            if target == FOCUS_CLASS and raw_prediction == FOCUS_CLASS and candidate_prediction != FOCUS_CLASS:
                add(condition, sample_index, "class1_tp_break")
            if target == FOCUS_CLASS and raw_prediction != FOCUS_CLASS and candidate_prediction == FOCUS_CLASS:
                add(condition, sample_index, "class1_fn_rescue")
            if target in RESTRICTED_NEGATIVE_CLASSES and raw_prediction == FOCUS_CLASS and candidate_prediction != FOCUS_CLASS:
                add(condition, sample_index, "restricted_fp_removal")
            if target in RESTRICTED_NEGATIVE_CLASSES and raw_prediction != FOCUS_CLASS and candidate_prediction == FOCUS_CLASS:
                add(condition, sample_index, "restricted_fp_creation")
    required = set(categories)
    if len(categories) < int(minimum_rows):
        clean_raw = predictions["clean"]["raw"]
        clean_candidate = predictions["clean"]["supervised_minority"]
        ranked = sorted(
            zip(clean_raw, clean_candidate),
            key=lambda pair: (
                -abs(float(pair[1]["prob_1"]) - float(pair[0]["prob_1"])),
                int(pair[0]["sample_index"]),
            ),
        )
        for raw, _candidate in ranked:
            add("clean", int(raw["sample_index"]), "diagnostic_largest_class1_delta")
            if len(categories) >= int(minimum_rows):
                break
    events = []
    for condition in CONDITIONS:
        indices = sorted(
            int(event_id.split(":", 1)[1])
            for event_id in categories
            if event_id.startswith(f"{condition}:")
        )
        events.extend((condition, sample_index) for sample_index in indices)
    return events, categories, required


def _normalize_map(value: Tensor) -> np.ndarray:
    tensor = value.detach().float().cpu()
    minimum = float(tensor.min().item())
    maximum = float(tensor.max().item())
    if maximum - minimum <= 1e-12:
        return np.zeros(tuple(tensor.shape), dtype=np.float32)
    return ((tensor - minimum) / (maximum - minimum)).numpy().astype(np.float32)


def _positive_mass(value: Tensor, mask: Tensor) -> float:
    positive = value.detach().float().clamp_min(0.0)
    total = positive.sum()
    if float(total.item()) <= 0.0:
        return 0.0
    return float((positive[mask].sum() / total).item())


def _xai_region_masks(
    *,
    bbox_prior: Tensor,
    memory_key_padding_mask: Optional[Tensor],
) -> Dict[str, Tensor]:
    if bbox_prior.numel() != 256:
        raise ValueError("XAI bbox prior must contain the full 16x16 patch grid.")
    prior = bbox_prior.detach().float().reshape(16, 16)
    if torch.is_tensor(memory_key_padding_mask):
        valid = ~memory_key_padding_mask.detach().bool().reshape(16, 16)
    else:
        valid = torch.ones(16, 16, dtype=torch.bool, device=prior.device)
    foreground = prior.gt(0.05) & valid
    if not bool(foreground.any().item()):
        foreground = prior.eq(prior[valid].max()) & valid
    foreground_float = foreground.float().reshape(1, 1, 16, 16)
    eroded = 1.0 - F.max_pool2d(
        1.0 - foreground_float, kernel_size=3, stride=1, padding=1
    )
    interior = eroded[0, 0].gt(0.5) & foreground
    boundary = foreground & ~interior
    outer = valid & ~foreground
    return {
        "valid": valid,
        "foreground": foreground,
        "boundary": boundary,
        "outer_context": outer,
        "bbox_prior": prior,
    }


def _forward_xai_maps(
    *,
    keeper: nn.Module,
    candidate: ConditionViewVariant,
    images: Tensor,
    metadata: Mapping[str, object],
    device: torch.device,
) -> Dict[str, object]:
    keeper = keeper.to(device).eval()
    candidate = candidate.to(device).eval()
    for module in (keeper, candidate):
        for parameter in module.parameters():
            parameter.requires_grad_(False)

    def run(candidate_path: bool) -> tuple[Tensor, Tensor, Mapping[str, object], Optional[Tensor]]:
        activation: Dict[str, Tensor] = {}

        def hook(_module, _inputs, output: Tensor) -> None:
            activation["value"] = output
            output.retain_grad()

        handle = keeper.patch_embed.proj.register_forward_hook(hook)
        x = images.detach().to(device).requires_grad_(True)
        bbox = metadata.get("bbox")
        image_mask = metadata.get("image_mask")
        bbox_device = (
            bbox.to(device=device, dtype=torch.float32)
            if torch.is_tensor(bbox)
            else None
        )
        image_mask_device = (
            image_mask.to(device=device, dtype=torch.bool)
            if torch.is_tensor(image_mask)
            else None
        )
        features = keeper.forward_features(
            x,
            bbox_token_prior=bbox_device,
            image_valid_mask=image_mask_device,
            return_attention=True,
            attention_layers=[0],
        )
        if bbox_device is not None:
            features["bbox"] = bbox_device
        raw_logits = classification_logits_from_features(keeper, features)
        logits = (
            candidate.deployed_logits(features["pooled"].float(), raw_logits.float())
            if candidate_path
            else raw_logits.float()
        )
        logits[:, FOCUS_CLASS].sum().backward()
        value = activation.get("value")
        if not torch.is_tensor(value) or not torch.is_tensor(value.grad):
            handle.remove()
            raise ValueError("XAI patch activation or gradient is unavailable.")
        if tuple(value.shape[1:]) != (EMBED_DIM, 16, 16):
            handle.remove()
            raise ValueError(f"XAI patch activation shape differs: {tuple(value.shape)}")
        gta = torch.relu((value.float() * value.grad.float()).sum(dim=1))
        native = None
        if not candidate_path:
            attentions = features.get("attentions")
            if not isinstance(attentions, Mapping) or not torch.is_tensor(attentions.get(0)):
                handle.remove()
                raise ValueError("Native layer-0 attention is unavailable.")
            attention = attentions[0]
            if tuple(attention.shape[1:]) != (8, 263, 263):
                handle.remove()
                raise ValueError(f"Native attention shape differs: {tuple(attention.shape)}")
            prefix = int(getattr(keeper, "num_prefix_tokens", -1))
            if prefix != 7:
                handle.remove()
                raise ValueError(f"Native attention prefix count differs: {prefix}")
            native = attention[:, :, 0, prefix : prefix + 256].float().mean(dim=1)
            native = native.reshape(-1, 16, 16)
        handle.remove()
        return (
            F.softmax(logits.detach().float(), dim=1).cpu(),
            gta.detach().float().cpu(),
            features,
            native.detach().float().cpu() if torch.is_tensor(native) else None,
        )

    raw_probabilities, raw_gta, raw_features, native = run(False)
    candidate_probabilities, candidate_gta, candidate_features, _unused = run(True)
    if native is None:
        raise ValueError("Native attention was not retained.")
    bbox_prior = raw_features.get("patch_bbox_prior")
    if not torch.is_tensor(bbox_prior) or tuple(bbox_prior.shape[1:]) != (256,):
        raise ValueError("XAI requires a full patch_bbox_prior.")
    memory_mask = raw_features.get("memory_key_padding_mask")
    return {
        "raw_probabilities": raw_probabilities,
        "candidate_probabilities": candidate_probabilities,
        "raw_gta": raw_gta,
        "candidate_gta": candidate_gta,
        "native_attention": native,
        "bbox_prior": bbox_prior.detach().float().cpu(),
        "memory_key_padding_mask": (
            memory_mask.detach().bool().cpu()
            if torch.is_tensor(memory_mask)
            else None
        ),
        "native_provenance": {
            "source": "keeper.forward_features(return_attention=True, attention_layers=[0])",
            "layer_index": 0,
            "representation": str(raw_features.get("attention_representation")),
            "member": str(raw_features.get("attention_member")),
            "tensor_shape": [
                int(value) for value in raw_features["attentions"][0].shape
            ],
            "fallback_used": False,
        },
        "candidate_feature_shape": [
            int(value) for value in candidate_features["pooled"].shape
        ],
    }


def _bbox_pixel_mask(
    bbox: Tensor, *, height: int, width: int, valid_mask: Optional[Tensor]
) -> Tensor:
    cx, cy, box_width, box_height = [float(value) for value in bbox[:4].tolist()]
    x1 = max(0.0, cx - 0.5 * box_width)
    x2 = min(1.0, cx + 0.5 * box_width)
    y1 = max(0.0, cy - 0.5 * box_height)
    y2 = min(1.0, cy + 0.5 * box_height)
    x = (torch.arange(width, dtype=torch.float32) + 0.5) / float(width)
    y = (torch.arange(height, dtype=torch.float32) + 0.5) / float(height)
    mask = (y[:, None] >= y1) & (y[:, None] <= y2) & (x[None, :] >= x1) & (x[None, :] <= x2)
    if torch.is_tensor(valid_mask):
        mask = mask & valid_mask.detach().bool().cpu()
    if not bool(mask.any().item()):
        raise ValueError("XAI bbox produced an empty pixel mask.")
    return mask


def _counterfactual_tensors(
    *,
    image: Tensor,
    bbox: Tensor,
    image_valid_mask: Optional[Tensor],
    mean: Sequence[float],
    std: Sequence[float],
) -> tuple[Tensor, Dict[str, object]]:
    mean_tensor = torch.tensor(mean, dtype=torch.float32).reshape(3, 1, 1)
    std_tensor = torch.tensor(std, dtype=torch.float32).reshape(3, 1, 1)
    rgb = (image.detach().float().cpu() * std_tensor + mean_tensor).clamp(0.0, 1.0)
    height, width = int(rgb.size(1)), int(rgb.size(2))
    object_mask = _bbox_pixel_mask(
        bbox.detach().float().cpu(),
        height=height,
        width=width,
        valid_mask=image_valid_mask,
    )
    valid = (
        image_valid_mask.detach().bool().cpu()
        if torch.is_tensor(image_valid_mask)
        else torch.ones(height, width, dtype=torch.bool)
    )
    background_mask = valid & ~object_mask
    gray = (
        0.2989 * rgb[0:1] + 0.5870 * rgb[1:2] + 0.1140 * rgb[2:3]
    ).expand_as(rgb)
    uint8 = (
        rgb.permute(1, 2, 0).mul(255.0).round().to(torch.uint8).numpy()
    )
    blurred = torch.from_numpy(
        np.asarray(
            Image.fromarray(uint8).filter(ImageFilter.GaussianBlur(radius=8)),
            dtype=np.uint8,
        ).copy()
    ).permute(2, 0, 1).float() / 255.0
    object_desaturate = torch.where(object_mask.unsqueeze(0), gray, rgb)
    background_gray = torch.where(background_mask.unsqueeze(0), gray, rgb)
    background_blur = torch.where(background_mask.unsqueeze(0), blurred, rgb)
    variants = torch.stack((object_desaturate, background_blur, background_gray))
    normalized = (variants - mean_tensor.unsqueeze(0)) / std_tensor.unsqueeze(0)
    return normalized, {
        "names": ["object_desaturate", "background_blur", "background_gray"],
        "bbox_mode": "normalized_cxcywh_exact_model_metadata",
        "blur": "PIL_GaussianBlur_radius_8",
        "object_pixel_fraction": float(object_mask.float().mean().item()),
        "background_pixel_fraction": float(background_mask.float().mean().item()),
    }


def _counterfactual_probabilities(
    *,
    keeper: nn.Module,
    candidate: ConditionViewVariant,
    images: Tensor,
    metadata: Mapping[str, object],
    mean: Sequence[float],
    std: Sequence[float],
    device: torch.device,
) -> tuple[Dict[str, Tensor], list[Dict[str, object]]]:
    batches = []
    contracts = []
    bbox = metadata.get("bbox")
    image_mask = metadata.get("image_mask")
    if not torch.is_tensor(bbox):
        raise ValueError("Counterfactual XAI requires bbox metadata.")
    for row in range(int(images.size(0))):
        variants, contract = _counterfactual_tensors(
            image=images[row],
            bbox=bbox[row],
            image_valid_mask=(image_mask[row] if torch.is_tensor(image_mask) else None),
            mean=mean,
            std=std,
        )
        batches.append(variants)
        contracts.append(contract)
    stacked = torch.cat(batches, dim=0).to(device)
    repeated: Dict[str, object] = {}
    for key, value in metadata.items():
        if torch.is_tensor(value):
            repeated[key] = value.repeat_interleave(3, dim=0).to(device)
        else:
            repeated[key] = value
    with torch.inference_mode(), torch.autocast(
        device_type="cuda", dtype=torch.float16, enabled=True
    ):
        raw_logits, features = _forward_classification_with_metadata(
            keeper, stacked, repeated, device=device
        )
    if not isinstance(features, Mapping) or not torch.is_tensor(features.get("pooled")):
        raise ValueError("Counterfactual XAI has no pooled feature.")
    candidate_logits = candidate.deployed_logits(
        features["pooled"].float(), raw_logits.float()
    )
    batch = int(images.size(0))
    return {
        "raw": F.softmax(raw_logits.float(), dim=1).reshape(batch, 3, NUM_CLASSES).cpu(),
        "candidate": F.softmax(candidate_logits.float(), dim=1).reshape(
            batch, 3, NUM_CLASSES
        ).cpu(),
    }, contracts


def _render_xai_pages(
    records: Sequence[Mapping[str, object]], output_dir: Path
) -> tuple[list[str], list[str]]:
    columns = ("image", "native", "raw_gta", "candidate_gta", "bbox", "outer")
    panel_width = 150
    row_height = 190
    pages = []
    page_event_ids = []
    for page_index, start in enumerate(range(0, len(records), 3)):
        page_records = records[start : start + 3]
        canvas = Image.new(
            "RGB",
            (len(columns) * panel_width, len(page_records) * row_height),
            "white",
        )
        draw = ImageDraw.Draw(canvas)
        for row_number, record in enumerate(page_records):
            rgb = np.asarray(record["rgb"], dtype=np.uint8)
            display = record["display_maps"]
            for column, name in enumerate(columns):
                panel = (
                    Image.fromarray(rgb)
                    if name == "image"
                    else _heat_overlay(rgb, np.asarray(display[name]), alpha=0.55)
                )
                panel = panel.resize(
                    (panel_width, panel_width), Image.Resampling.BILINEAR
                )
                x = column * panel_width
                y = row_number * row_height
                canvas.paste(panel, (x, y))
                draw.text((x + 3, y + panel_width + 2), name, fill="black")
            draw.text(
                (3, row_number * row_height + panel_width + 19),
                (
                    f"{record['event_id']} y={record['target']} "
                    f"raw/cand={record['raw_prediction']}/{record['candidate_prediction']}"
                ),
                fill="black",
            )
            page_event_ids.append(str(record["event_id"]))
        path = output_dir / f"supervised_minority_xai_page_{page_index:03d}.png"
        canvas.save(path)
        pages.append(str(path.resolve()))
    return pages, page_event_ids


def _xai_audit(
    *,
    keeper: nn.Module,
    candidate: ConditionViewVariant,
    dataset: MangoYOLOCropDataset,
    transform,
    rows: Sequence[CleanTrainRow],
    predictions: Mapping[str, Mapping[str, Sequence[Mapping[str, object]]]],
    mean: Sequence[float],
    std: Sequence[float],
    args: argparse.Namespace,
    device: torch.device,
    output_dir: Path,
) -> Dict[str, object]:
    events, categories, required = _required_xai_events(predictions)
    prediction_lookup = {
        condition: {
            name: {int(row["sample_index"]): row for row in values}
            for name, values in condition_values.items()
        }
        for condition, condition_values in predictions.items()
    }
    event_set = set(events)
    keeper = keeper.to(device).eval()
    candidate_device = copy.deepcopy(candidate).to(device).eval()
    records = []
    display_records = []
    arrays: Dict[str, np.ndarray] = {}
    tensor_event_ids = []
    prediction_mismatches = 0
    native_provenance: Optional[Dict[str, object]] = None
    for condition in CONDITIONS:
        indices = [sample_index for name, sample_index in events if name == condition]
        if not indices:
            continue
        xai_args = argparse.Namespace(**vars(args))
        xai_args.batch_size = int(args.xai_batch_size)
        loader, _loader_summary = _condition_loader(
            name=condition,
            dataset=dataset,
            transform=transform,
            indices=indices,
            args=xai_args,
            context=f"ttc_supmin_xai_{condition}",
        )
        for images_cpu, targets_cpu, metadata_cpu in loader:
            sample_indices = metadata_cpu.get("sample_index")
            if not torch.is_tensor(sample_indices):
                raise ValueError("XAI loader is missing sample_index.")
            maps = _forward_xai_maps(
                keeper=keeper,
                candidate=candidate_device,
                images=images_cpu,
                metadata=metadata_cpu,
                device=device,
            )
            perturb_probabilities, perturb_contracts = _counterfactual_probabilities(
                keeper=keeper,
                candidate=candidate_device,
                images=images_cpu,
                metadata=metadata_cpu,
                mean=mean,
                std=std,
                device=device,
            )
            if native_provenance is None:
                native_provenance = dict(maps["native_provenance"])
            for local, sample_index_tensor in enumerate(sample_indices):
                sample_index = int(sample_index_tensor.item())
                if (condition, sample_index) not in event_set:
                    raise ValueError("XAI loader produced an unselected event.")
                event_id = f"{condition}:{sample_index}"
                raw_row = prediction_lookup[condition]["raw"][sample_index]
                ce_row = prediction_lookup[condition]["ce_identity"][sample_index]
                standard_row = prediction_lookup[condition]["standard_supcon"][sample_index]
                candidate_row = prediction_lookup[condition]["supervised_minority"][sample_index]
                raw_probabilities = maps["raw_probabilities"][local]
                candidate_probabilities = maps["candidate_probabilities"][local]
                prediction_mismatches += int(
                    int(raw_probabilities.argmax().item()) != int(raw_row["prediction"])
                )
                prediction_mismatches += int(
                    int(candidate_probabilities.argmax().item())
                    != int(candidate_row["prediction"])
                )
                memory_mask = maps["memory_key_padding_mask"]
                regions = _xai_region_masks(
                    bbox_prior=maps["bbox_prior"][local],
                    memory_key_padding_mask=(
                        memory_mask[local] if torch.is_tensor(memory_mask) else None
                    ),
                )
                native = maps["native_attention"][local]
                raw_gta = maps["raw_gta"][local]
                candidate_gta = maps["candidate_gta"][local]
                masses = {}
                for name, value in (
                    ("native_attention", native),
                    ("raw_gta", raw_gta),
                    ("candidate_gta", candidate_gta),
                ):
                    masses[name] = {
                        "foreground": _positive_mass(value, regions["foreground"]),
                        "boundary": _positive_mass(value, regions["boundary"]),
                        "outer_context": _positive_mass(value, regions["outer_context"]),
                    }
                perturbations: Dict[str, object] = {}
                names = perturb_contracts[local]["names"]
                for model_name, clean_probabilities, locked_row in (
                    ("raw", raw_probabilities, raw_row),
                    ("candidate", candidate_probabilities, candidate_row),
                ):
                    model_values = {}
                    locked_prediction = int(locked_row["prediction"])
                    target = int(locked_row["target"])
                    for probe_index, probe_name in enumerate(names):
                        values = perturb_probabilities[model_name][local, probe_index]
                        model_values[probe_name] = {
                            "probabilities": [float(value) for value in values.tolist()],
                            "prediction": int(values.argmax().item()),
                            "class1_probability_drop": float(
                                clean_probabilities[FOCUS_CLASS].item()
                                - values[FOCUS_CLASS].item()
                            ),
                            "original_prediction_drop": float(
                                clean_probabilities[locked_prediction].item()
                                - values[locked_prediction].item()
                            ),
                            "target_probability_drop": float(
                                clean_probabilities[target].item() - values[target].item()
                            ),
                        }
                    perturbations[model_name] = model_values
                raw_prob_locked = [
                    float(raw_row[f"prob_{index}"]) for index in range(NUM_CLASSES)
                ]
                ce_prob_locked = [
                    float(ce_row[f"prob_{index}"]) for index in range(NUM_CLASSES)
                ]
                standard_prob_locked = [
                    float(standard_row[f"prob_{index}"]) for index in range(NUM_CLASSES)
                ]
                candidate_prob_locked = [
                    float(candidate_row[f"prob_{index}"]) for index in range(NUM_CLASSES)
                ]
                serial = {
                    "event_id": event_id,
                    "condition": condition,
                    "sample_index": sample_index,
                    "image_path": str(rows[sample_index].image_path),
                    "target": int(targets_cpu[local].item()),
                    "categories": categories[event_id],
                    "required": event_id in required,
                    "raw_prediction": int(raw_row["prediction"]),
                    "ce_identity_prediction": int(ce_row["prediction"]),
                    "standard_supcon_prediction": int(standard_row["prediction"]),
                    "candidate_prediction": int(candidate_row["prediction"]),
                    "raw_probabilities": raw_prob_locked,
                    "ce_identity_probabilities": ce_prob_locked,
                    "standard_supcon_probabilities": standard_prob_locked,
                    "candidate_probabilities": candidate_prob_locked,
                    "masses": masses,
                    "perturbations": perturbations,
                    "perturbation_contract": perturb_contracts[local],
                }
                serial["case_sha256"] = _canonical_record_sha256([serial])
                records.append(serial)
                array_prefix = event_id.replace(":", "_")
                arrays[f"{array_prefix}_rgb"] = _rgb_from_tensor(
                    images_cpu[local], mean=mean, std=std
                )
                for key, value in (
                    ("native_attention", native),
                    ("raw_gta", raw_gta),
                    ("candidate_gta", candidate_gta),
                    ("bbox_prior", regions["bbox_prior"]),
                ):
                    arrays[f"{array_prefix}_{key}"] = (
                        value.detach().float().cpu().numpy().astype(np.float32)
                    )
                for key in ("valid", "foreground", "boundary", "outer_context"):
                    arrays[f"{array_prefix}_{key}_mask"] = (
                        regions[key].detach().cpu().numpy().astype(np.uint8)
                    )
                tensor_event_ids.append(event_id)
                display_records.append(
                    {
                        **serial,
                        "rgb": arrays[f"{array_prefix}_rgb"],
                        "display_maps": {
                            "native": _normalize_map(native),
                            "raw_gta": _normalize_map(raw_gta),
                            "candidate_gta": _normalize_map(candidate_gta),
                            "bbox": _normalize_map(regions["bbox_prior"]),
                            "outer": regions["outer_context"].float().cpu().numpy(),
                        },
                    }
                )
            print(
                json.dumps(
                    {
                        "phase": "xai",
                        "condition": condition,
                        "completed": len(records),
                        "total": len(events),
                    }
                ),
                flush=True,
            )
    if [str(record["event_id"]) for record in records] != [
        f"{condition}:{sample_index}" for condition, sample_index in events
    ]:
        raise ValueError("XAI record order differs from selected event order.")
    tensor_path = output_dir / "supervised_minority_xai_tensors.npz"
    np.savez_compressed(tensor_path, **arrays)
    pages, page_event_ids = _render_xai_pages(display_records, output_dir)
    case_path = output_dir / "supervised_minority_xai_cases.json"
    case_path.write_text(
        json.dumps(records, indent=2, sort_keys=True, ensure_ascii=True) + "\n",
        encoding="utf-8",
    )
    record_ids = [str(record["event_id"]) for record in records]
    required_covered = sorted(required.intersection(record_ids))
    candidate_foreground = [
        float(record["masses"]["candidate_gta"]["foreground"])
        for record in records
    ]
    raw_boundary = [
        float(record["masses"]["raw_gta"]["boundary"]) for record in records
    ]
    candidate_boundary = [
        float(record["masses"]["candidate_gta"]["boundary"]) for record in records
    ]
    raw_outer = [
        float(record["masses"]["raw_gta"]["outer_context"]) for record in records
    ]
    candidate_outer = [
        float(record["masses"]["candidate_gta"]["outer_context"])
        for record in records
    ]
    mean_value = lambda values: float(sum(values) / max(1, len(values)))
    aggregate = {
        "candidate_foreground_mass_mean": mean_value(candidate_foreground),
        "raw_boundary_mass_mean": mean_value(raw_boundary),
        "candidate_boundary_mass_mean": mean_value(candidate_boundary),
        "boundary_mass_delta": mean_value(candidate_boundary) - mean_value(raw_boundary),
        "raw_outer_context_mass_mean": mean_value(raw_outer),
        "candidate_outer_context_mass_mean": mean_value(candidate_outer),
        "outer_context_mass_delta": mean_value(candidate_outer) - mean_value(raw_outer),
    }
    checks = {
        "required_coverage_100pct": len(required_covered) == len(required),
        "record_tensor_event_order_exact": record_ids == tensor_event_ids,
        "record_page_event_order_exact": record_ids == page_event_ids,
        "native_attention_no_fallback": bool(native_provenance)
        and not bool(native_provenance.get("fallback_used", True)),
        "prediction_replay_argmax_exact": prediction_mismatches == 0,
        "candidate_foreground_majority_mean": float(
            aggregate["candidate_foreground_mass_mean"]
        )
        > 0.50,
        "candidate_boundary_mass_delta_lte_0p05": float(
            aggregate["boundary_mass_delta"]
        )
        <= 0.05,
        "candidate_outer_context_mass_delta_lte_0p05": float(
            aggregate["outer_context_mass_delta"]
        )
        <= 0.05,
        "all_tensor_values_finite": all(
            np.isfinite(value).all() for value in arrays.values()
        ),
    }
    manifest = {
        "method": str(_protocol_contract(args)["method"]),
        "selected_events": len(events),
        "required_events": len(required),
        "required_covered": len(required_covered),
        "event_ids": record_ids,
        "required_event_ids": sorted(required),
        "required_covered_event_ids": required_covered,
        "categories": categories,
        "case_sha256": [str(record["case_sha256"]) for record in records],
        "case_order_sha256": _canonical_record_sha256(
            [{"event_id": event_id} for event_id in record_ids]
        ),
        "tensor_event_order_sha256": _canonical_record_sha256(
            [{"event_id": event_id} for event_id in tensor_event_ids]
        ),
        "page_event_order_sha256": _canonical_record_sha256(
            [{"event_id": event_id} for event_id in page_event_ids]
        ),
        "native_attention_provenance": native_provenance,
        "gradient_map": "positive_gradient_times_patch_embed_activation_class1",
        "gradient_precision": "cuda_fp32",
        "aggregate": aggregate,
        "prediction_argmax_mismatches": prediction_mismatches,
        "checks": checks,
        "failed_checks": [key for key, passed in checks.items() if not passed],
        "all_gates_passed": all(checks.values()),
        "tensors": str(tensor_path.resolve()),
        "tensors_sha256": _sha256(tensor_path),
        "cases": str(case_path.resolve()),
        "cases_sha256": _sha256(case_path),
        "pages": pages,
        "page_sha256": [_sha256(Path(path)) for path in pages],
    }
    manifest_path = output_dir / "supervised_minority_xai_manifest.json"
    manifest_path.write_text(
        json.dumps(manifest, indent=2, sort_keys=True, ensure_ascii=True) + "\n",
        encoding="utf-8",
    )
    keeper.cpu()
    del candidate_device
    gc.collect()
    torch.cuda.empty_cache()
    return {**manifest, "manifest": str(manifest_path.resolve()), "manifest_sha256": _sha256(manifest_path)}


def _initial_deployed_exactness(
    *,
    variants: Mapping[str, ConditionViewVariant],
    clean_cache: Mapping[str, Tensor],
) -> Dict[str, object]:
    pooled = clean_cache["pooled"][:32].float()
    raw_logits = clean_cache["raw_logits"][:32].float()
    rows: Dict[str, object] = {}
    for name, variant in variants.items():
        with torch.inference_mode():
            observed = variant.cpu().eval().deployed_logits(pooled, raw_logits)
        rows[name] = {
            "bit_exact": torch.equal(observed, raw_logits),
            "maximum_absolute_error": float((observed - raw_logits).abs().amax().item()),
        }
    return {
        "rows": 32,
        "variants": rows,
        "all_bit_exact": all(bool(value["bit_exact"]) for value in rows.values()),
    }


def _initial_optimizer_exactness(
    variants: Mapping[str, ConditionViewVariant]
) -> Dict[str, object]:
    representation_states = []
    probe_states = []
    for name in ("standard_supcon", "supervised_minority"):
        optimizer = torch.optim.SGD(
            variants[name].representation_parameters(),
            lr=STAGE1_LEARNING_RATE,
            momentum=STAGE1_MOMENTUM,
            weight_decay=STAGE1_WEIGHT_DECAY,
        )
        representation_states.append(optimizer.state_dict())
    for name in ("ce_identity", "standard_supcon", "supervised_minority"):
        optimizer = torch.optim.Adam(
            variants[name].probe.parameters(),
            lr=STAGE2_LEARNING_RATE,
            weight_decay=STAGE2_WEIGHT_DECAY,
        )
        probe_states.append(optimizer.state_dict())
    return {
        "representation_optimizer_initial_states_bit_exact": representation_states[0]
        == representation_states[1],
        "probe_optimizer_initial_states_bit_exact": probe_states[0]
        == probe_states[1]
        == probe_states[2],
        "representation_optimizer_state_entries": len(
            representation_states[0]["state"]
        ),
        "probe_optimizer_state_entries": len(probe_states[0]["state"]),
    }


def _write_predictions(
    path: Path,
    *,
    rows: Sequence[CleanTrainRow],
    predictions: Mapping[str, Mapping[str, Sequence[Mapping[str, object]]]],
) -> None:
    model_names = (
        "raw",
        "ce_identity",
        "standard_supcon",
        "supervised_minority",
    )
    fields = [
        "condition",
        "sample_index",
        "source_stem",
        "image_path",
        "fold",
        "target",
    ]
    for name in model_names:
        fields.append(f"{name}_prediction")
        fields.extend(f"{name}_prob_{index}" for index in range(NUM_CLASSES))
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields)
        writer.writeheader()
        for condition in CONDITIONS:
            condition_values = predictions[condition]
            for aligned in zip(*(condition_values[name] for name in model_names)):
                sample_index = int(aligned[0]["sample_index"])
                source = rows[sample_index]
                output: Dict[str, object] = {
                    "condition": condition,
                    "sample_index": sample_index,
                    "source_stem": source.source_stem,
                    "image_path": str(source.image_path),
                    "fold": source.fold,
                    "target": source.target,
                }
                for name, values in zip(model_names, aligned):
                    if int(values["sample_index"]) != sample_index:
                        raise ValueError("Prediction models are not row-aligned.")
                    output[f"{name}_prediction"] = int(values["prediction"])
                    for class_index in range(NUM_CLASSES):
                        output[f"{name}_prob_{class_index}"] = float(
                            values[f"prob_{class_index}"]
                        )
                writer.writerow(output)


def _write_curves(
    path: Path,
    *,
    representation_training: Mapping[str, object],
    probe_training: Mapping[str, object],
) -> None:
    fields = (
        "stage",
        "variant",
        "epoch",
        "learning_rate",
        "loss",
        "mean_batch_gradient_norm",
        "rows",
        "batches",
    )
    values = []
    for summaries in (representation_training, probe_training):
        for name in ("ce_identity", "standard_supcon", "supervised_minority"):
            values.extend(summaries[name].get("history", []))
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields)
        writer.writeheader()
        writer.writerows(values)


def _replay_predictions(
    path: Path,
    *,
    expected: Mapping[str, Mapping[str, Mapping[str, object]]],
) -> Dict[str, object]:
    model_names = (
        "raw",
        "ce_identity",
        "standard_supcon",
        "supervised_minority",
    )
    grouped = {
        condition: {name: [] for name in model_names} for condition in CONDITIONS
    }
    with path.open("r", encoding="utf-8-sig", newline="") as handle:
        for source in csv.DictReader(handle):
            condition = str(source["condition"])
            if condition not in grouped:
                raise ValueError(f"Replay found an unexpected condition: {condition}")
            for name in model_names:
                row: Dict[str, object] = {
                    "sample_index": int(source["sample_index"]),
                    "target": int(source["target"]),
                    "prediction": int(source[f"{name}_prediction"]),
                }
                for class_index in range(NUM_CLASSES):
                    row[f"prob_{class_index}"] = float(
                        source[f"{name}_prob_{class_index}"]
                    )
                grouped[condition][name].append(row)
    replay = _build_comparisons(grouped)
    exact = replay == expected
    return {
        "rows": sum(len(values["raw"]) for values in grouped.values()),
        "condition_rows": {
            condition: len(values["raw"]) for condition, values in grouped.items()
        },
        "comparisons_exact": exact,
        "prediction_sha256": _sha256(path),
        "replayed_comparisons": replay,
    }


def _write_report(path: Path, summary: Mapping[str, object]) -> None:
    status = str(summary["status"])
    method = str(summary["method"])
    gate = summary["gate"]
    lines = [
        "# Supervised-Minority Condition-View Result",
        "",
        f"- Method: `{method}`",
        f"- Status: `{status}`",
        f"- Stage B authorized: `{gate.get('stage_b_authorized', False)}`",
        f"- Validation used: `{summary.get('validation_predictions_used', False)}`",
        f"- Test used: `{summary.get('test_data_used', False)}`",
    ]
    if "comparisons" in summary:
        clean = summary["comparisons"]["clean"]["raw_candidate"]
        lines.extend(
            [
                f"- Clean macro/class1 F1 delta vs raw: `{clean['delta']['macro_f1']:+.6f}/{clean['delta']['class1_f1']:+.6f}`",
                f"- Clean class1 precision/recall delta vs raw: `{clean['delta']['class1_precision']:+.6f}/{clean['delta']['class1_recall']:+.6f}`",
                f"- Restricted FP reduction / TP breaks: `{clean['transitions']['restricted_focus_fp_reduction']}/{clean['transitions']['focus_tp_break']}`",
                f"- XAI selected/required: `{summary['xai']['selected_events']}/{summary['xai']['required_events']}`",
            ]
        )
    else:
        failed = summary["pretraining_selectivity_gate"]["failed_checks"]
        lines.append(f"- Pre-training selectivity failures: `{', '.join(failed) or 'none'}`")
    lines.extend(
        [
            f"- Failed final checks: `{', '.join(gate.get('failed_checks', [])) or 'none'}`",
            "",
            "Only source-disjoint yolo_f/train rows were used. Validation and test were not accessed.",
        ]
    )
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def _write_manifest(output_dir: Path, *, method: str) -> Dict[str, object]:
    manifest_path = output_dir / "artifact_manifest.json"
    forbidden = {".pt", ".pth", ".ckpt", ".engine", ".trt"}
    artifacts = []
    for path in sorted(output_dir.rglob("*")):
        if not path.is_file() or path == manifest_path:
            continue
        if path.suffix.casefold() in forbidden:
            raise ValueError(f"Forbidden trainable binary artifact: {path}")
        artifacts.append(
            {
                "path": str(path.resolve()),
                "relative_path": str(path.relative_to(output_dir)).replace("\\", "/"),
                "sha256": _sha256(path),
                "bytes": int(path.stat().st_size),
            }
        )
    manifest = {
        "method": str(method),
        "artifact_count": len(artifacts),
        "total_bytes": sum(int(value["bytes"]) for value in artifacts),
        "artifacts": artifacts,
    }
    manifest_path.write_text(
        json.dumps(manifest, indent=2, sort_keys=True, ensure_ascii=True) + "\n",
        encoding="utf-8",
    )
    return {
        **manifest,
        "path": str(manifest_path.resolve()),
        "sha256": _sha256(manifest_path),
    }


def _structural_gate(
    *,
    construction: Mapping[str, object],
    cache_diagnostics: Mapping[str, object],
    equation: Mapping[str, object],
    initial_exactness: Mapping[str, object],
    optimizer_exactness: Mapping[str, object],
    representation_training: Mapping[str, object],
    probe_training: Mapping[str, object],
    keeper_state_before: str,
    keeper: nn.Module,
) -> Dict[str, object]:
    expected_secondary = {
        "lighting_dim": 49_145,
        "lighting_bright": 49_126,
        "low_contrast": 49_169,
    }
    checks = {
        "keeper_parameter_count_exact": int(construction["keeper_parameters"])
        == EXPECTED_KEEPER_PARAMETERS,
        "representation_parameter_count_exact": int(
            construction["representation_parameters"]
        )
        == EXPECTED_REPRESENTATION_PARAMETERS,
        "probe_parameter_count_exact": int(construction["probe_parameters"])
        == EXPECTED_PROBE_PARAMETERS,
        "total_parameter_count_exact": int(construction["total_parameters"])
        == EXPECTED_TOTAL_PARAMETERS,
        "constructor_rng_restored": bool(construction["rng_restored"]),
        "initial_variant_states_bit_exact": bool(
            construction["initial_states_bit_exact"]
        ),
        "adapter_up_zero": bool(construction["adapter_up_zero"]),
        "probe_zero": bool(construction["probe_zero"]),
        "cache_replay_exact": bool(
            cache_diagnostics["replay"]["deterministic_within_1e_6"]
        ),
        "cache_cidt_replay_exact": bool(
            cache_diagnostics["replay"]["cidt_within_2e_3"]
        ),
        "equation_matches_within_1e_6": bool(
            equation["matches_independent_and_official_within_1e_6"]
        ),
        "supmin_nonclass1_positive_count_exact": bool(
            equation["supmin_nonclass1_positive_count_exactly_one"]
        ),
        "supmin_class1_positive_count_exact": bool(
            equation["supmin_class1_positive_count_exact"]
        ),
        "initial_deployed_logits_bit_exact": bool(initial_exactness["all_bit_exact"]),
        "representation_optimizer_initial_states_bit_exact": bool(
            optimizer_exactness["representation_optimizer_initial_states_bit_exact"]
        ),
        "probe_optimizer_initial_states_bit_exact": bool(
            optimizer_exactness["probe_optimizer_initial_states_bit_exact"]
        ),
        "ce_identity_representation_state_bit_exact": bool(
            representation_training["ce_identity"]["state_bit_exact"]
        ),
        "representation_occurrences_exact": all(
            int(representation_training[name]["occurrences"])
            == EXPECTED_CONTRASTIVE_OCCURRENCES
            for name in ("standard_supcon", "supervised_minority")
        ),
        "secondary_condition_counts_exact": all(
            representation_training[name]["secondary_counts"] == expected_secondary
            for name in ("standard_supcon", "supervised_minority")
        ),
        "probe_occurrences_exact": all(
            int(probe_training[name]["occurrences"]) == EXPECTED_PROBE_OCCURRENCES
            for name in ("ce_identity", "standard_supcon", "supervised_minority")
        ),
        "all_representation_gradients_seen": all(
            bool(representation_training[name]["all_trainable_gradients_seen"])
            for name in ("standard_supcon", "supervised_minority")
        ),
        "all_probe_gradients_seen": all(
            bool(probe_training[name]["all_trainable_gradients_seen"])
            for name in ("ce_identity", "standard_supcon", "supervised_minority")
        ),
        "all_training_gradients_finite": all(
            bool(representation_training[name]["all_gradients_finite"])
            for name in ("standard_supcon", "supervised_minority")
        )
        and all(
            bool(probe_training[name]["all_gradients_finite"])
            for name in ("ce_identity", "standard_supcon", "supervised_minority")
        ),
        "all_training_losses_finite": all(
            bool(representation_training[name]["all_losses_finite"])
            for name in ("standard_supcon", "supervised_minority")
        )
        and all(
            bool(probe_training[name]["all_losses_finite"])
            for name in ("ce_identity", "standard_supcon", "supervised_minority")
        ),
        "probe_did_not_change_representation": all(
            bool(probe_training[name]["representation_state_bit_exact"])
            for name in ("ce_identity", "standard_supcon", "supervised_minority")
        ),
        "keeper_has_no_gradient": all(parameter.grad is None for parameter in keeper.parameters()),
        "keeper_state_bit_exact": _state_sha256(keeper) == keeper_state_before,
    }
    failed = [key for key, passed in checks.items() if not passed]
    return {"checks": checks, "failed_checks": failed, "all_gates_passed": not failed}


def _deployment_gate(
    *,
    export: Mapping[str, object],
    resources: Mapping[str, object],
) -> Dict[str, object]:
    isolated = export["isolated"]
    full = export["full"]
    checks = {
        "isolated_onnx_succeeded": bool(isolated["succeeded"]),
        "isolated_onnx_error_lte_1e_5": float(
            isolated["maximum_absolute_error"]
        )
        <= 1e-5,
        "isolated_onnx_argmax_exact": bool(isolated["argmax_match"]),
        "full_onnx_succeeded": bool(full["succeeded"]),
        "full_onnx_probability_error_lte_5e_4": float(
            full["maximum_absolute_error"]
        )
        <= 5e-4,
        "full_onnx_argmax_exact": bool(full["argmax_match"]),
        "isolated_projector_absent": bool(
            isolated["graph"]["training_projector_absent"]
        ),
        "full_projector_absent": bool(full["graph"]["training_projector_absent"]),
        **{str(key): bool(value) for key, value in resources["checks"].items()},
    }
    failed = [key for key, passed in checks.items() if not passed]
    return {"checks": checks, "failed_checks": failed, "all_gates_passed": not failed}


def _final_gate(
    *,
    structural: Mapping[str, object],
    selectivity: Mapping[str, object],
    mechanism: Mapping[str, object],
    behavior: Mapping[str, object],
    xai: Mapping[str, object],
    deployment: Mapping[str, object],
    replay: Mapping[str, object],
) -> Dict[str, object]:
    components = {
        "structural": bool(structural["all_gates_passed"]),
        "pretraining_selectivity": bool(selectivity["all_gates_passed"]),
        "representation_mechanism": bool(mechanism["all_gates_passed"]),
        "train_holdout_behavior": bool(behavior["all_gates_passed"]),
        "xai": bool(xai["all_gates_passed"]),
        "deployment_resources": bool(deployment["all_gates_passed"]),
        "independent_replay": bool(replay["comparisons_exact"]),
    }
    failed = [key for key, passed in components.items() if not passed]
    return {
        "components": components,
        "failed_checks": failed,
        "all_gates_passed": not failed,
        "stage_b_authorized": not failed,
        "validation_authorized": not failed,
        "test_authorized": False,
        "shared_trainer_integration_authorized": False,
        "full_train_authorized": False,
        "keeper_replacement_authorized": False,
        "command_file_revision_authorized": False,
    }


def _pretraining_rejection_outputs(
    *,
    method: str,
    output_dir: Path,
    provenance: Mapping[str, object],
    cohorts: Mapping[str, object],
    dataset_summary: Mapping[str, object],
    construction: Mapping[str, object],
    cache_diagnostics: Mapping[str, object],
    representation_metrics: Mapping[str, object],
    selectivity: Mapping[str, object],
) -> Dict[str, object]:
    metrics_path = output_dir / "representation_metrics.json"
    metrics_path.write_text(
        json.dumps(
            {"raw": representation_metrics},
            indent=2,
            sort_keys=True,
            ensure_ascii=True,
        )
        + "\n",
        encoding="utf-8",
    )
    gate = {
        "failed_checks": ["pretraining_selectivity"],
        "all_gates_passed": False,
        "stage_b_authorized": False,
        "validation_authorized": False,
        "test_authorized": False,
        "full_train_authorized": False,
        "keeper_replacement_authorized": False,
        "command_file_revision_authorized": False,
    }
    summary: Dict[str, object] = {
        "method": str(method),
        "status": "rejected_pretraining_selectivity",
        "provenance": provenance,
        "cohort": _cohort_serializable(cohorts),
        "dataset": dataset_summary,
        "construction": construction,
        "cache_diagnostics": cache_diagnostics,
        "pretraining_representation_metrics": representation_metrics,
        "pretraining_selectivity_gate": selectivity,
        "representation_training_started": False,
        "probe_training_started": False,
        "xai_required": False,
        "gate": gate,
        "validation_predictions_used": False,
        "test_data_used": False,
    }
    summary_path = output_dir / "summary.json"
    summary_path.write_text(
        json.dumps(summary, indent=2, sort_keys=True, ensure_ascii=True) + "\n",
        encoding="utf-8",
    )
    _write_report(output_dir / "report.md", summary)
    manifest = _write_manifest(output_dir, method=method)
    return {**summary, "artifact_manifest": manifest}


def run_audit(args: argparse.Namespace) -> Dict[str, object]:
    contract = _protocol_contract(args)
    method = str(contract["method"])
    provenance, rows, cohorts = _load_locked_inputs(args)
    if bool(args.preflight_only):
        return {
            "status": "preflight_passed",
            "method": method,
            "provenance": provenance,
            "cohort": _cohort_serializable(cohorts),
            "output_directory_created": False,
            "image_data_opened": False,
            "validation_predictions_used": False,
            "test_data_used": False,
        }
    if not torch.cuda.is_available():
        raise RuntimeError(f"Locked TTC-SupMin {contract['contract_id']} requires CUDA.")
    output_path = Path(args.output_dir).resolve()
    raw_root = Path(args.data).resolve().parent
    if output_path == raw_root or raw_root in output_path.parents:
        raise ValueError("Audit output may not be written under the raw dataset tree.")
    if output_path.exists() and any(output_path.iterdir()):
        raise FileExistsError(f"Formal output directory must be absent or empty: {output_path}")
    set_seed(SEED, deterministic=True)
    device = torch.device("cuda")
    checkpoint = load_checkpoint(Path(args.checkpoint), map_location="cpu")
    keeper, initial_variants, construction = _construct_keeper_and_variants(checkpoint)
    keeper_state_before = _state_sha256(keeper)
    dataset, transform, dataset_summary = _build_dataset(
        checkpoint, rows, Path(args.data)
    )
    caches, cache_diagnostics = _extract_all_condition_caches(
        keeper=keeper,
        dataset=dataset,
        transform=transform,
        rows=rows,
        args=args,
        device=device,
    )
    raw_representation = _representation_metrics(
        caches=caches,
        cohorts=cohorts,
        variant=None,
        device=device,
    )
    selectivity = assess_pretraining_selectivity(raw_representation)
    output_dir = _prepare_output_dir(output_path)
    if not bool(selectivity["representation_training_authorized"]):
        return _pretraining_rejection_outputs(
            method=method,
            output_dir=output_dir,
            provenance=provenance,
            cohorts=cohorts,
            dataset_summary=dataset_summary,
            construction=construction,
            cache_diagnostics=cache_diagnostics,
            representation_metrics=raw_representation,
            selectivity=selectivity,
        )
    initial_exactness = _initial_deployed_exactness(
        variants=initial_variants, clean_cache=caches["clean"]
    )
    optimizer_exactness = _initial_optimizer_exactness(initial_variants)
    equation = _equation_diagnostics()
    representation_variants, representation_training = _train_all_representations(
        variants=initial_variants,
        caches=caches,
        fit_indices=cohorts["fit_indices"],
        device=device,
    )
    trained_variants, probe_training = _train_all_probes(
        variants=representation_variants,
        clean_cache=caches["clean"],
        fit_indices=cohorts["fit_indices"],
        device=device,
    )
    representations: Dict[str, object] = {"raw": raw_representation}
    for name in ("ce_identity", "standard_supcon", "supervised_minority"):
        representations[name] = _representation_metrics(
            caches=caches,
            cohorts=cohorts,
            variant=trained_variants[name],
            device=device,
        )
    mechanism = assess_representation_mechanism(
        representations=representations,
        representation_training=representation_training,
    )
    predictions = _evaluate_variants(
        caches=caches,
        variants=trained_variants,
        holdout_indices=cohorts["holdout_indices"],
        device=device,
    )
    comparisons = _build_comparisons(predictions)
    behavior = assess_train_holdout_behavior(comparisons)
    prediction_path = output_dir / "predictions_all_conditions.csv"
    _write_predictions(prediction_path, rows=rows, predictions=predictions)
    curve_path = output_dir / "training_curves.csv"
    _write_curves(
        curve_path,
        representation_training=representation_training,
        probe_training=probe_training,
    )
    representation_path = output_dir / "representation_metrics.json"
    representation_path.write_text(
        json.dumps(representations, indent=2, sort_keys=True, ensure_ascii=True)
        + "\n",
        encoding="utf-8",
    )
    replay = _replay_predictions(prediction_path, expected=comparisons)
    replay_path = output_dir / "independent_replay.json"
    replay_path.write_text(
        json.dumps(replay, indent=2, sort_keys=True, ensure_ascii=True) + "\n",
        encoding="utf-8",
    )
    resource_loader, resource_loader_summary = _condition_loader(
        name="clean",
        dataset=dataset,
        transform=transform,
        indices=cohorts["holdout_indices"][:32],
        args=args,
        context="ttc_supmin_resource_batch",
    )
    resource_batch = next(iter(resource_loader))
    mean, std = checkpoint_input_normalization(checkpoint)
    xai = _xai_audit(
        keeper=keeper,
        candidate=trained_variants["supervised_minority"],
        dataset=dataset,
        transform=transform,
        rows=rows,
        predictions=predictions,
        mean=mean,
        std=std,
        args=args,
        device=device,
        output_dir=output_dir,
    )
    export = _export_diagnostics(
        keeper=keeper,
        candidate=trained_variants["supervised_minority"],
        clean_cache=caches["clean"],
        resource_batch=resource_batch,
        output_dir=output_dir,
    )
    resources = _resource_diagnostics(
        keeper=keeper,
        candidate=trained_variants["supervised_minority"],
        resource_batch=resource_batch,
        device=device,
    )
    resources["loader"] = resource_loader_summary
    structural = _structural_gate(
        construction=construction,
        cache_diagnostics=cache_diagnostics,
        equation=equation,
        initial_exactness=initial_exactness,
        optimizer_exactness=optimizer_exactness,
        representation_training=representation_training,
        probe_training=probe_training,
        keeper_state_before=keeper_state_before,
        keeper=keeper,
    )
    deployment = _deployment_gate(export=export, resources=resources)
    gate = _final_gate(
        structural=structural,
        selectivity=selectivity,
        mechanism=mechanism,
        behavior=behavior,
        xai=xai,
        deployment=deployment,
        replay=replay,
    )
    summary: Dict[str, object] = {
        "method": method,
        "status": "stage_a_passed" if gate["stage_b_authorized"] else "rejected_stage_a",
        "provenance": provenance,
        "cohort": _cohort_serializable(cohorts),
        "dataset": dataset_summary,
        "construction": construction,
        "cache_diagnostics": cache_diagnostics,
        "equation_diagnostics": equation,
        "initial_deployed_exactness": initial_exactness,
        "initial_optimizer_exactness": optimizer_exactness,
        "representation_training": representation_training,
        "probe_training": probe_training,
        "pretraining_selectivity_gate": selectivity,
        "representation_mechanism_gate": mechanism,
        "train_holdout_behavior_gate": behavior,
        "comparisons": comparisons,
        "structural_gate": structural,
        "xai": xai,
        "export": export,
        "resources": resources,
        "deployment_gate": deployment,
        "independent_replay": replay,
        "gate": gate,
        "representation_metrics_path": str(representation_path.resolve()),
        "training_curves_path": str(curve_path.resolve()),
        "predictions_path": str(prediction_path.resolve()),
        "validation_predictions_used": False,
        "test_data_used": False,
        "raw_dataset_modified": False,
        "tensor_rt_claimed": False,
    }
    summary_path = output_dir / "summary.json"
    summary_path.write_text(
        json.dumps(summary, indent=2, sort_keys=True, ensure_ascii=True) + "\n",
        encoding="utf-8",
    )
    _write_report(output_dir / "report.md", summary)
    manifest = _write_manifest(output_dir, method=method)
    return {**summary, "artifact_manifest": manifest}


def main(argv: Optional[Sequence[str]] = None) -> None:
    args = parse_args(argv)
    result = run_audit(args)
    output = {
        "status": result["status"],
        "method": str(result["method"]),
        "stage_b_authorized": bool(result.get("gate", {}).get("stage_b_authorized", False)),
        "validation_predictions_used": False,
        "test_data_used": False,
    }
    if "artifact_manifest" in result:
        output["artifact_manifest_sha256"] = result["artifact_manifest"]["sha256"]
    print(json.dumps(output, indent=2, sort_keys=True), flush=True)


if __name__ == "__main__":
    main()
