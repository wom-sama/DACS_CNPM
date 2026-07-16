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
import time
from typing import Dict, Mapping, Optional, Sequence

import numpy as np

os.environ.setdefault("CUBLAS_WORKSPACE_CONFIG", ":4096:8")

import torch
from PIL import Image, ImageDraw
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
from trkh.tools.audit_augself_color_adapter_readiness import _required_xai_indices
from trkh.tools.audit_ceconv_stem_residual_readiness import (
    _build_comparisons,
    _build_dataset,
    _failed_export,
    _full_worktree_clean,
    _heat_overlay,
    _onnx_compare,
    _replay_predictions,
    _rgb_from_tensor,
    _tracked_worktree_clean,
    _write_predictions,
)
from trkh.tools.audit_class1_boundary_cagrad_readiness import CONDITIONS
from trkh.tools.audit_class1_reference_agem_readiness import (
    FOCUS_CLASS,
    RESTRICTED_NEGATIVE_CLASSES,
    _locked_cohort_summary,
)
from trkh.tools.audit_deep_class_prompt_readiness import LIGHTING_CONDITIONS
from trkh.tools.audit_more_model_rebalancing_readiness import (
    CleanTrainRow,
    _ordered_index_sha256,
    _read_clean_train_rows,
)
from trkh.tools.audit_mutual_channel_patch_readiness import (
    _batch_from_cache,
    _cache_positions,
    _extract_fit_cache,
    _feature_tensors,
    _gradient_norm,
    _normalize_map,
    _parameter_gradient_map,
    _prediction_row,
    _repeat_metadata,
    _state_dict_sha256,
    _stream_tensor_sha256,
)
from trkh.tools.audit_patch_style_srm_readiness import _make_lighting_loader
from trkh.tools.audit_visual_contrast_attention_readiness import _state_sha256
from trkh.tools.audit_xca_dual_axis_readiness import (
    _amp_dtype,
    _git_commit,
    _make_loader,
    _metadata_to_device,
    _prepare_output_dir,
    _sha256,
    _tensor_sha256,
    _verify_sha256,
)


METHOD = "support_constrained_capsule_alm_a0"
NUM_CLASSES = 5
EMBED_DIM = 256
PRIMARY_DIM = 8
CAPSULE_DIM = 16
GRID_HEIGHT = 16
GRID_WIDTH = 16
EXPECTED_PATCHES = 167
EXPECTED_KEEPER_PARAMETERS = 7_245_590
EXPECTED_ADAPTER_PARAMETERS = 166_006
SEED = 42
FOLD = 0
EPOCHS = 30
NATURAL_BATCH_SIZE = 32
BOUNDARY_HALF_BATCH = 16
BOUNDARY_BATCHES_PER_CONDITION = 27
BOUNDARY_EVENTS_PER_EPOCH = 108
LEARNING_RATE = 0.1
MOMENTUM = 0.9
WEIGHT_DECAY = 2e-4
RESIDUAL_SCALE = 0.05
RANK_MARGIN = 0.05
ROUTING_ITERATIONS = 3
MAX_RUNTIME_RATIO = 1.15
MAX_FULL_VRAM_GIB = 3.25
MAX_INCREMENTAL_VRAM_GIB = 0.75
MAX_ONNX_ERROR = 1e-5
STATIC_EXPORT_BATCH_SIZE = 1

LOCKED_KEEPER_SHA256 = (
    "1f49d577240c69dc63c30af70db52ec2aa9da65a17aef1c4b1c09ece6c482677"
)
LOCKED_LAUNCHER_ARGS_SHA256 = (
    "908a05cf66b2a01162cae62e4ff2251eaae1297d31e70510144e4954159b7eff"
)
LOCKED_DATA_SHA256 = (
    "716e33df24c63a9e9920f97b685199707fb84ab4c7154544f5dd9a3e00d884ef"
)
LOCKED_CIDT_SUMMARY_SHA256 = (
    "d4891edf2963ab12385b7ce5bdc812ec3e19c5c098acd25c66eb557af541d7ad"
)
LOCKED_CIDT_PREDICTIONS_SHA256 = (
    "2e0993752d58d99ea429bfefe1e2bfe6fa949e45aea1a26cc4bdfee97d4db21c"
)
LOCKED_MCL_SUMMARY_SHA256 = (
    "cf4a0143afde61ce2a737fca4a859397d19d058ad3ca01c85d624a34e7183b0f"
)
LOCKED_MCL_PREDICTIONS_SHA256 = (
    "1feff96cedf182b003163f2c22b14b075481de892ca3acdab878e728c925a1d5"
)
LOCKED_PROTOCOL_SHA256 = (
    "4344e6b16957b4e75ce432c9d04e70b511198d983fa380060fd190d5e7cd9667"
)
LOCKED_SABOUR_PAPER_SHA256 = (
    "e87bc5186f7767d49d0ea6535183b82b84e3d330d147d41abf27fd98a423069d"
)
LOCKED_CAPSULE_COMMIT = "984fbc754943c849c55a57923f4223099a1ff88c"
LOCKED_CAPSULE_LAYERS_SHA256 = (
    "b6d94ee33c4a0bb4d29c8520fdd15937ad2a8c59995e79b63bcf95ee898a8b89"
)
LOCKED_CAPSULE_MODEL_SHA256 = (
    "e1a115dbda2855241a9acc450e3f4bf6361721a99fb5007bd8120d4149de7862"
)
LOCKED_CAPSULE_README_SHA256 = (
    "d535b01f266ee3dba661cac5ad677c487b52448ac009900d9c39ce010ecaed07"
)
LOCKED_CAPSULE_LICENSE_SHA256 = (
    "b42e0eeff6d2d55ef63bc57c328b5a219a9fa20ddaecf7b9d48a6786c9e17e6b"
)
LOCKED_SANGALLI_PAPER_SHA256 = (
    "908915f5d5f09ae218ab8c387e6f0b33a9b64a5e707418bb4ffc98fb0a05052c"
)
LOCKED_SANGALLI_SUPPLEMENT_SHA256 = (
    "e78dc837a82e4c57c900068e4ecad7d2ab5deada8ca8eb632dbf26cc8122b6c3"
)
LOCKED_ALM_COMMIT = "ab3da74d0d80dcffd3666d5ee4371f57908279e2"
LOCKED_ALM_LOSSES_SHA256 = (
    "d9cb0f1db8c8c1c7e369cc44d9a8c1b5a51ad1a4f84d43878d1d161c45bd3cf8"
)
LOCKED_ALM_TRAIN_SHA256 = (
    "5a513187e9510cf00e7b461838a207144cd78eae135085a4f65cf3f7f9323f93"
)
LOCKED_ALM_README_SHA256 = (
    "1bde578e23dfaebae21c406a668f1063ee56c5c17a621e863557cb1fdc185652"
)
LOCKED_CURRENT_COMMAND_SHA256 = (
    "36b9aa1a21b765829acf4c8321be147bd76297de4ccdb8a40e6dee8e37940faf"
)
LOCKED_COMMAND_HISTORY_SHA256 = (
    "39bd2879ce66fddf36a953021ea1e40f8d9de6cb4334b9b825011b2b8dc98f53"
)
LOCKED_FIT_INDEX_SHA256 = (
    "22edca99022fe2dcd0287a08b5f6b7c0d699603b904f65c82b51fdea7f31ce5d"
)
LOCKED_HOLDOUT_INDEX_SHA256 = (
    "a628686b491c8b8f10bbf1782e6c84a923f21c8b53617cf0b0325260e97e97ae"
)
LOCKED_REFERENCE_INDEX_SHA256 = (
    "1474f2d8e8fe83767e3c250fd89e827f1544fb058c762bc9c114b150a18d2574"
)
LOCKED_HARD_INDEX_SHA256 = (
    "56dfdbc6180d99b80605799bcd8e2ef412f0abd5c0191054b4fe7d6554d7f020"
)
LOCKED_NATURAL_ORDER_SHA256 = (
    "9fefbbac66a1ce22190744092edbfa3a9b44c0c1b7d4accf257b09e2b01ffb5b"
)
LOCKED_BOUNDARY_SCHEDULE_SHA256 = (
    "54d9a40365e250d10e582258194064e936428b08541d132b6a299afafa7b0788"
)


def parse_args(argv: Optional[Sequence[str]] = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Locked train-only support-constrained capsule ALM audit. "
            "Validation and test access are forbidden."
        )
    )
    parser.add_argument(
        "--checkpoint",
        type=Path,
        default=Path(
            "runs/probe_v8_yolof_pairroute_teacherfocusbinary015_boundarydrop_"
            "bboxprior_120b_2e_20260701/checkpoints/best.pt"
        ),
    )
    parser.add_argument(
        "--launcher-args",
        type=Path,
        default=Path(
            "runs/probe_v8_yolof_pairroute_teacherfocusbinary015_boundarydrop_"
            "bboxprior_120b_2e_20260701/launcher_args.json"
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
            "runs/audit_cidt_readiness_full_train_20260714/"
            "predictions_all_conditions.csv"
        ),
    )
    parser.add_argument(
        "--mcl-summary",
        type=Path,
        default=Path("runs/audit_mutual_channel_patch_readiness_20260716/summary.json"),
    )
    parser.add_argument(
        "--mcl-predictions",
        type=Path,
        default=Path(
            "runs/audit_mutual_channel_patch_readiness_20260716/"
            "predictions_all_conditions.csv"
        ),
    )
    parser.add_argument(
        "--protocol",
        type=Path,
        default=Path(
            "docs/TRKH_5CLASS_SUPPORT_CONSTRAINED_CAPSULE_ALM_"
            "READINESS_PROTOCOL_20260716.md"
        ),
    )
    parser.add_argument(
        "--sabour-paper",
        type=Path,
        default=Path(
            r"D:\DataAI\external_sources\papers\sabour2017_dynamic_routing_capsules.pdf"
        ),
    )
    parser.add_argument(
        "--capsule-root",
        type=Path,
        default=Path(
            r"D:\DataAI\external_sources\official\dynamic_routing_capsules"
        ),
    )
    parser.add_argument(
        "--sangalli-paper",
        type=Path,
        default=Path(
            r"D:\DataAI\external_sources\papers\sangalli2021_constrained_critical_classes.pdf"
        ),
    )
    parser.add_argument(
        "--sangalli-supplement",
        type=Path,
        default=Path(
            r"D:\DataAI\external_sources\papers\sangalli2021_constrained_critical_classes_supplement.pdf"
        ),
    )
    parser.add_argument(
        "--alm-root",
        type=Path,
        default=Path(r"D:\DataAI\external_sources\official\alm-dnn-github"),
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=Path("runs/audit_support_constrained_capsule_alm_readiness_20260716"),
    )
    parser.add_argument("--preflight-only", action="store_true", default=False)
    parser.add_argument("--device", choices=("cuda",), default="cuda")
    parser.add_argument("--batch-size", type=int, default=NATURAL_BATCH_SIZE)
    parser.add_argument("--num-workers", type=int, default=4)
    parser.add_argument("--epochs", type=int, default=EPOCHS)
    parser.add_argument("--seed", type=int, default=SEED)
    parser.add_argument("--fold", type=int, default=FOLD)
    parser.add_argument("--learning-rate", type=float, default=LEARNING_RATE)
    parser.add_argument("--momentum", type=float, default=MOMENTUM)
    parser.add_argument("--weight-decay", type=float, default=WEIGHT_DECAY)
    parser.add_argument("--residual-scale", type=float, default=RESIDUAL_SCALE)
    parser.add_argument("--rank-margin", type=float, default=RANK_MARGIN)
    parser.add_argument("--routing-iterations", type=int, default=ROUTING_ITERATIONS)
    parser.add_argument("--benchmark-repeats", type=int, default=7)
    parser.add_argument("--xai-batch-size", type=int, default=4)
    return parser.parse_args(argv)


def _locked_args_exact(args: argparse.Namespace) -> bool:
    return bool(
        str(args.device) == "cuda"
        and int(args.batch_size) == NATURAL_BATCH_SIZE
        and int(args.num_workers) == 4
        and int(args.epochs) == EPOCHS
        and int(args.seed) == SEED
        and int(args.fold) == FOLD
        and math.isclose(float(args.learning_rate), LEARNING_RATE, abs_tol=0.0)
        and math.isclose(float(args.momentum), MOMENTUM, abs_tol=0.0)
        and math.isclose(float(args.weight_decay), WEIGHT_DECAY, abs_tol=0.0)
        and math.isclose(float(args.residual_scale), RESIDUAL_SCALE, abs_tol=0.0)
        and math.isclose(float(args.rank_margin), RANK_MARGIN, abs_tol=0.0)
        and int(args.routing_iterations) == ROUTING_ITERATIONS
        and int(args.benchmark_repeats) >= 3
        and int(args.xai_batch_size) >= 1
    )


def _source_paths(args: argparse.Namespace) -> Dict[str, Path]:
    capsule = Path(args.capsule_root).resolve()
    alm = Path(args.alm_root).resolve()
    return {
        "checkpoint": Path(args.checkpoint).resolve(),
        "launcher_args": Path(args.launcher_args).resolve(),
        "data": Path(args.data).resolve(),
        "cidt_summary": Path(args.cidt_summary).resolve(),
        "cidt_predictions": Path(args.cidt_predictions).resolve(),
        "mcl_summary": Path(args.mcl_summary).resolve(),
        "mcl_predictions": Path(args.mcl_predictions).resolve(),
        "protocol": Path(args.protocol).resolve(),
        "sabour_paper": Path(args.sabour_paper).resolve(),
        "capsule_root": capsule,
        "capsule_layers": capsule / "research/capsules/models/layers/layers.py",
        "capsule_model": capsule / "research/capsules/models/capsule_model.py",
        "capsule_readme": capsule / "research/capsules/README.md",
        "capsule_license": capsule / "LICENSE",
        "sangalli_paper": Path(args.sangalli_paper).resolve(),
        "sangalli_supplement": Path(args.sangalli_supplement).resolve(),
        "alm_root": alm,
        "alm_losses": alm / "utils/custom_losses.py",
        "alm_train": alm / "utils/train_test_CIFAR.py",
        "alm_readme": alm / "README.md",
        "current_commands": Path(
            "docs/TRKH_CURRENT_BEST_FULL_TRAIN_COMMANDS_20260706.txt"
        ).resolve(),
        "command_history": Path(
            "docs/TRKH_CURRENT_BEST_COMMAND_UPDATE_HISTORY.txt"
        ).resolve(),
    }


def locked_natural_order(
    fit_indices: Sequence[int], *, seed: int = SEED, epochs: int = EPOCHS
) -> list[int]:
    values = np.asarray([int(value) for value in fit_indices], dtype=np.int64)
    generator = np.random.default_rng(int(seed))
    return [
        int(value)
        for _ in range(int(epochs))
        for value in generator.permutation(values).tolist()
    ]


def locked_boundary_schedule(
    reference_indices: Sequence[int],
    hard_indices: Sequence[int],
    *,
    seed: int = SEED,
    epochs: int = EPOCHS,
) -> list[Dict[str, object]]:
    reference = np.asarray(
        [int(value) for value in reference_indices], dtype=np.int64
    )
    hard = np.asarray([int(value) for value in hard_indices], dtype=np.int64)
    if reference.size != 432 or hard.size != 186:
        raise ValueError("CapsALM boundary cohort sizes differ from protocol.")
    generator = np.random.default_rng(int(seed))
    events: list[Dict[str, object]] = []
    for epoch in range(int(epochs)):
        for condition_index, condition in enumerate(CONDITIONS):
            positives = generator.permutation(reference)
            negative_stream: list[int] = []
            while len(negative_stream) < int(positives.size):
                negative_stream.extend(int(value) for value in generator.permutation(hard))
            negative_stream = negative_stream[: int(positives.size)]
            for batch in range(BOUNDARY_BATCHES_PER_CONDITION):
                start = batch * BOUNDARY_HALF_BATCH
                stop = (batch + 1) * BOUNDARY_HALF_BATCH
                events.append(
                    {
                        "epoch": epoch,
                        "condition": condition,
                        "condition_index": condition_index,
                        "batch": batch,
                        "positive_indices": [
                            int(value) for value in positives[start:stop].tolist()
                        ],
                        "negative_indices": [
                            int(value) for value in negative_stream[start:stop]
                        ],
                    }
                )
    return events


def boundary_schedule_sha256(events: Sequence[Mapping[str, object]]) -> str:
    digest = hashlib.sha256()
    for event in events:
        positives = ",".join(str(int(value)) for value in event["positive_indices"])
        negatives = ",".join(str(int(value)) for value in event["negative_indices"])
        digest.update(
            (
                f"{int(event['epoch'])},{event['condition']},"
                f"{int(event['batch'])}|{positives}|{negatives}\n"
            ).encode("ascii")
        )
    return digest.hexdigest()


def interleaved_boundary_batches(
    *, natural_batches: int, boundary_events: int = BOUNDARY_EVENTS_PER_EPOCH
) -> list[int]:
    if natural_batches < boundary_events or boundary_events <= 0:
        raise ValueError("Boundary events require at least as many natural batches.")
    output = [
        int(event * int(natural_batches) // int(boundary_events))
        for event in range(int(boundary_events))
    ]
    if len(set(output)) != int(boundary_events):
        raise ValueError("Boundary interleave unexpectedly shares a natural batch.")
    return output


class SupportConstrainedCapsuleAdapter(nn.Module):
    def __init__(
        self,
        *,
        feature_dim: int = EMBED_DIM,
        primary_dim: int = PRIMARY_DIM,
        num_classes: int = NUM_CLASSES,
        capsule_dim: int = CAPSULE_DIM,
        residual_scale: float = RESIDUAL_SCALE,
        routing_iterations: int = ROUTING_ITERATIONS,
        seed: int = SEED,
    ) -> None:
        super().__init__()
        self.feature_dim = int(feature_dim)
        self.primary_dim = int(primary_dim)
        self.num_classes = int(num_classes)
        self.capsule_dim = int(capsule_dim)
        self.residual_scale = float(residual_scale)
        self.routing_iterations = int(routing_iterations)
        with torch.random.fork_rng(devices=[]):
            torch.manual_seed(int(seed))
            self.primary_projection = nn.Linear(
                self.feature_dim, self.primary_dim, bias=True
            )
            self.vote_weights = nn.Parameter(
                torch.empty(
                    GRID_HEIGHT * GRID_WIDTH,
                    self.primary_dim,
                    self.num_classes,
                    self.capsule_dim,
                )
            )
            nn.init.trunc_normal_(self.vote_weights, std=0.1, a=-0.2, b=0.2)
            self.class_capsule_bias = nn.Parameter(
                torch.full((self.num_classes, self.capsule_dim), 0.1)
            )
            self.residual_head = nn.Linear(
                self.num_classes, self.num_classes, bias=True
            )
            nn.init.zeros_(self.residual_head.weight)
            nn.init.zeros_(self.residual_head.bias)

    @staticmethod
    def squash(values: Tensor, *, epsilon: float = 1e-9) -> Tensor:
        norm = torch.linalg.vector_norm(values, dim=-1, keepdim=True)
        norm_squared = norm.square()
        return (values / norm.clamp_min(float(epsilon))) * (
            norm_squared / (1.0 + norm_squared)
        )

    def route(
        self,
        primary_capsules: Tensor,
        patch_indices: Tensor,
        token_valid: Tensor,
        *,
        iterations: Optional[int] = None,
    ) -> Dict[str, Tensor]:
        if primary_capsules.ndim != 3 or int(primary_capsules.size(-1)) != self.primary_dim:
            raise ValueError("Primary capsules must have shape [B,N,8].")
        if patch_indices.shape != primary_capsules.shape[:2]:
            raise ValueError("Patch indices do not align with primary capsules.")
        if token_valid.shape != primary_capsules.shape[:2]:
            raise ValueError("Token-valid mask does not align with primary capsules.")
        indices = patch_indices.to(dtype=torch.long, device=primary_capsules.device)
        valid = token_valid.to(dtype=torch.bool, device=primary_capsules.device)
        if bool((indices < 0).any().item()) or bool(
            (indices >= GRID_HEIGHT * GRID_WIDTH).any().item()
        ):
            raise ValueError("Patch index is outside the native 16x16 grid.")
        weights = self.vote_weights[indices]
        votes = torch.einsum("bni,bnicd->bncd", primary_capsules, weights)
        routing_logits = primary_capsules.new_zeros(
            primary_capsules.size(0), primary_capsules.size(1), self.num_classes
        )
        count = self.routing_iterations if iterations is None else int(iterations)
        if count < 1:
            raise ValueError("Routing iterations must be positive.")
        couplings = routing_logits
        class_capsules = votes.new_zeros(
            votes.size(0), self.num_classes, self.capsule_dim
        )
        agreement = routing_logits
        for _ in range(count):
            couplings = F.softmax(routing_logits, dim=2)
            couplings = couplings * valid.unsqueeze(-1).to(dtype=couplings.dtype)
            preactivation = torch.einsum("bnc,bncd->bcd", couplings, votes)
            preactivation = preactivation + self.class_capsule_bias.unsqueeze(0)
            class_capsules = self.squash(preactivation)
            agreement = torch.einsum("bncd,bcd->bnc", votes, class_capsules)
            agreement = agreement * valid.unsqueeze(-1).to(dtype=agreement.dtype)
            routing_logits = routing_logits + agreement
        return {
            "votes": votes,
            "couplings": couplings,
            "class_capsules": class_capsules,
            "agreement": agreement,
            "routing_logits": routing_logits,
        }

    def forward_sparse(
        self,
        patches: Tensor,
        patch_indices: Tensor,
        token_valid: Tensor,
        raw_logits: Tensor,
        *,
        routing_iterations: Optional[int] = None,
    ) -> Dict[str, Tensor]:
        if patches.ndim != 3 or int(patches.size(-1)) != self.feature_dim:
            raise ValueError("Patch tokens must have shape [B,N,256].")
        primary_pre = self.primary_projection(patches)
        primary = self.squash(primary_pre)
        routed = self.route(
            primary,
            patch_indices,
            token_valid,
            iterations=routing_iterations,
        )
        capsule_lengths = torch.linalg.vector_norm(
            routed["class_capsules"], dim=-1
        )
        residual = self.residual_head(capsule_lengths)
        deployed = raw_logits.float() + self.residual_scale * residual.float()
        return {
            "primary_pre": primary_pre,
            "primary_capsules": primary,
            **routed,
            "capsule_lengths": capsule_lengths,
            "residual_logits": residual,
            "logits": deployed,
        }


def class1_margin(logits: Tensor) -> Tensor:
    other = torch.tensor(
        [0, 2, 3, 4], device=logits.device, dtype=torch.long
    )
    return logits[:, FOCUS_CLASS] - logits.index_select(1, other).amax(dim=1)


def normalized_alm_constraints(
    positive_logits: Tensor,
    negative_logits: Tensor,
    raw_positive_logits: Tensor,
    *,
    margin: float = RANK_MARGIN,
) -> Dict[str, Tensor]:
    if positive_logits.ndim != 2 or negative_logits.ndim != 2:
        raise ValueError("ALM logits must be rank-two tensors.")
    if positive_logits.size(0) == 0 or negative_logits.size(0) == 0:
        raise ValueError("ALM requires positive and negative rows.")
    positive_p1 = F.softmax(positive_logits.float(), dim=1)[:, FOCUS_CLASS]
    negative_p1 = F.softmax(negative_logits.float(), dim=1)[:, FOCUS_CLASS]
    pairwise = F.relu(
        -(positive_p1[:, None] - negative_p1[None, :]) + float(margin)
    )
    q_rank = pairwise.mean(dim=1)
    q_support = F.relu(
        class1_margin(raw_positive_logits.float())
        - class1_margin(positive_logits.float())
    )
    return {
        "positive_p1": positive_p1,
        "negative_p1": negative_p1,
        "pairwise_rank_violation": pairwise,
        "q_rank": q_rank,
        "q_support": q_support,
    }


def augmented_lagrangian_loss(
    q_rank: Tensor,
    q_support: Tensor,
    lambda_rank: Tensor,
    lambda_support: Tensor,
) -> Dict[str, Tensor]:
    if not (
        q_rank.shape == q_support.shape == lambda_rank.shape == lambda_support.shape
    ):
        raise ValueError("ALM constraints and multipliers must align per positive row.")
    rank_penalty = 0.5 * q_rank.square().mean()
    rank_dual = (lambda_rank * q_rank).mean()
    support_penalty = 0.5 * q_support.square().mean()
    support_dual = (lambda_support * q_support).mean()
    return {
        "rank_penalty": rank_penalty,
        "rank_dual": rank_dual,
        "support_penalty": support_penalty,
        "support_dual": support_dual,
        "loss": rank_penalty + rank_dual + support_penalty + support_dual,
    }


def _reference_squash(values: Tensor) -> Tensor:
    output = torch.empty_like(values)
    flattened_input = values.reshape(-1, values.size(-1))
    flattened_output = output.reshape_as(flattened_input)
    for row in range(flattened_input.size(0)):
        vector = flattened_input[row]
        norm = torch.sqrt(sum(vector[index].square() for index in range(vector.numel())))
        norm_squared = norm.square()
        flattened_output[row] = (vector / norm.clamp_min(1e-9)) * (
            norm_squared / (1.0 + norm_squared)
        )
    return output


def _reference_route(
    adapter: SupportConstrainedCapsuleAdapter,
    patches: Tensor,
    patch_indices: Tensor,
    token_valid: Tensor,
    raw_logits: Tensor,
    *,
    iterations: int = ROUTING_ITERATIONS,
) -> Dict[str, Tensor]:
    primary_pre = patches @ adapter.primary_projection.weight.transpose(0, 1)
    primary_pre = primary_pre + adapter.primary_projection.bias
    primary = _reference_squash(primary_pre)
    batch, tokens, _ = primary.shape
    votes = primary.new_zeros(batch, tokens, NUM_CLASSES, CAPSULE_DIM)
    for row in range(batch):
        for token in range(tokens):
            position = int(patch_indices[row, token].item())
            for parent in range(NUM_CLASSES):
                votes[row, token, parent] = primary[row, token] @ adapter.vote_weights[
                    position, :, parent, :
                ]
    logits = primary.new_zeros(batch, tokens, NUM_CLASSES)
    couplings = logits
    class_capsules = votes.new_zeros(batch, NUM_CLASSES, CAPSULE_DIM)
    agreement = logits
    for _ in range(int(iterations)):
        maximum = logits.amax(dim=2, keepdim=True)
        exponent = torch.exp(logits - maximum)
        couplings = exponent / exponent.sum(dim=2, keepdim=True)
        couplings = couplings * token_valid.unsqueeze(-1).to(couplings.dtype)
        preactivation = votes.new_zeros(batch, NUM_CLASSES, CAPSULE_DIM)
        for row in range(batch):
            for parent in range(NUM_CLASSES):
                preactivation[row, parent] = adapter.class_capsule_bias[parent]
                for token in range(tokens):
                    preactivation[row, parent] += (
                        couplings[row, token, parent] * votes[row, token, parent]
                    )
        class_capsules = _reference_squash(preactivation)
        agreement = logits.new_zeros(logits.shape)
        for row in range(batch):
            for token in range(tokens):
                if bool(token_valid[row, token].item()):
                    for parent in range(NUM_CLASSES):
                        agreement[row, token, parent] = (
                            votes[row, token, parent] * class_capsules[row, parent]
                        ).sum()
        logits = logits + agreement
    lengths = torch.sqrt(class_capsules.square().sum(dim=2))
    residual = lengths @ adapter.residual_head.weight.transpose(0, 1)
    residual = residual + adapter.residual_head.bias
    return {
        "primary_pre": primary_pre,
        "primary_capsules": primary,
        "votes": votes,
        "couplings": couplings,
        "class_capsules": class_capsules,
        "agreement": agreement,
        "routing_logits": logits,
        "capsule_lengths": lengths,
        "residual_logits": residual,
        "logits": raw_logits.float() + adapter.residual_scale * residual.float(),
    }


def _equation_diagnostics() -> Dict[str, object]:
    generator = torch.Generator(device="cpu")
    generator.manual_seed(SEED + 991)
    patches = torch.randn(2, 4, EMBED_DIM, generator=generator)
    patch_indices = torch.tensor([[0, 3, 17, 255], [2, 31, 128, 240]])
    token_valid = torch.tensor([[True, True, False, True], [True, True, True, False]])
    raw_logits = torch.randn(2, NUM_CLASSES, generator=generator)
    adapter = SupportConstrainedCapsuleAdapter()
    with torch.random.fork_rng(devices=[]):
        torch.manual_seed(SEED + 992)
        nn.init.normal_(adapter.residual_head.weight, std=0.1)
        nn.init.normal_(adapter.residual_head.bias, std=0.1)
    local = adapter.forward_sparse(patches, patch_indices, token_valid, raw_logits)
    reference = _reference_route(
        adapter, patches, patch_indices, token_valid, raw_logits
    )
    keys = (
        "primary_pre",
        "primary_capsules",
        "votes",
        "couplings",
        "class_capsules",
        "agreement",
        "routing_logits",
        "capsule_lengths",
        "residual_logits",
        "logits",
    )
    errors = {
        key: float((local[key] - reference[key]).abs().amax().item()) for key in keys
    }
    valid_sum = local["couplings"].sum(dim=2)[token_valid]
    invalid_mass = local["couplings"][~token_valid].abs().sum()

    positive = torch.randn(3, NUM_CLASSES, generator=generator, requires_grad=True)
    negative = torch.randn(4, NUM_CLASSES, generator=generator, requires_grad=True)
    raw_positive = positive.detach().clone()
    constraints = normalized_alm_constraints(positive, negative, raw_positive)
    reference_rank = []
    reference_support = []
    for row in range(positive.size(0)):
        positive_p1 = torch.softmax(positive[row], dim=0)[FOCUS_CLASS]
        terms = []
        for negative_row in range(negative.size(0)):
            negative_p1 = torch.softmax(negative[negative_row], dim=0)[FOCUS_CLASS]
            terms.append(F.relu(-(positive_p1 - negative_p1) + RANK_MARGIN))
        reference_rank.append(torch.stack(terms).mean())
        reference_support.append(
            F.relu(
                class1_margin(raw_positive[row : row + 1])[0]
                - class1_margin(positive[row : row + 1])[0]
            )
        )
    reference_rank_tensor = torch.stack(reference_rank)
    reference_support_tensor = torch.stack(reference_support)
    constraint_errors = {
        "q_rank": float(
            (constraints["q_rank"] - reference_rank_tensor).abs().amax().item()
        ),
        "q_support": float(
            (constraints["q_support"] - reference_support_tensor).abs().amax().item()
        ),
    }
    return {
        "maximum_absolute_errors": errors,
        "constraint_maximum_absolute_errors": constraint_errors,
        "routing_matches_reference_lte_1e6": max(errors.values()) <= 1e-6,
        "constraints_match_reference_lte_1e7": max(constraint_errors.values()) <= 1e-7,
        "valid_coupling_parent_sum_error": float((valid_sum - 1.0).abs().amax().item()),
        "invalid_coupling_mass": float(invalid_mass.item()),
        "coupling_contract_exact": bool(
            (valid_sum - 1.0).abs().amax().item() <= 1e-6
            and invalid_mass.item() == 0.0
        ),
    }


def _cohort_serializable(cohorts: Mapping[str, object]) -> Dict[str, object]:
    excluded = {"fit_indices", "holdout_indices", "reference_indices", "hard_indices"}
    return {
        key: value
        for key, value in cohorts.items()
        if key not in excluded
    } | {
        "fit_rows": len(cohorts["fit_indices"]),
        "holdout_rows": len(cohorts["holdout_indices"]),
        "reference_rows": len(cohorts["reference_indices"]),
        "hard_rows": len(cohorts["hard_indices"]),
    }


def _load_locked_inputs(
    args: argparse.Namespace,
) -> tuple[
    Dict[str, object],
    list[CleanTrainRow],
    Dict[str, object],
    list[int],
    list[Dict[str, object]],
]:
    if not _locked_args_exact(args):
        raise ValueError("Arguments differ from the precommitted CapsALM protocol.")
    paths = _source_paths(args)
    hashes = {
        "checkpoint": _verify_sha256(
            paths["checkpoint"], LOCKED_KEEPER_SHA256, "keeper checkpoint"
        ),
        "launcher_args": _verify_sha256(
            paths["launcher_args"], LOCKED_LAUNCHER_ARGS_SHA256, "launcher args"
        ),
        "data": _verify_sha256(paths["data"], LOCKED_DATA_SHA256, "data YAML"),
        "cidt_summary": _verify_sha256(
            paths["cidt_summary"], LOCKED_CIDT_SUMMARY_SHA256, "CIDT summary"
        ),
        "cidt_predictions": _verify_sha256(
            paths["cidt_predictions"],
            LOCKED_CIDT_PREDICTIONS_SHA256,
            "CIDT predictions",
        ),
        "mcl_summary": _verify_sha256(
            paths["mcl_summary"], LOCKED_MCL_SUMMARY_SHA256, "MCL summary"
        ),
        "mcl_predictions": _verify_sha256(
            paths["mcl_predictions"],
            LOCKED_MCL_PREDICTIONS_SHA256,
            "MCL predictions",
        ),
        "protocol": _verify_sha256(
            paths["protocol"], LOCKED_PROTOCOL_SHA256, "CapsALM protocol"
        ),
        "sabour_paper": _verify_sha256(
            paths["sabour_paper"], LOCKED_SABOUR_PAPER_SHA256, "Sabour paper"
        ),
        "capsule_layers": _verify_sha256(
            paths["capsule_layers"],
            LOCKED_CAPSULE_LAYERS_SHA256,
            "official capsule layers",
        ),
        "capsule_model": _verify_sha256(
            paths["capsule_model"],
            LOCKED_CAPSULE_MODEL_SHA256,
            "official capsule model",
        ),
        "capsule_readme": _verify_sha256(
            paths["capsule_readme"],
            LOCKED_CAPSULE_README_SHA256,
            "official capsule README",
        ),
        "capsule_license": _verify_sha256(
            paths["capsule_license"],
            LOCKED_CAPSULE_LICENSE_SHA256,
            "official capsule license",
        ),
        "sangalli_paper": _verify_sha256(
            paths["sangalli_paper"],
            LOCKED_SANGALLI_PAPER_SHA256,
            "Sangalli paper",
        ),
        "sangalli_supplement": _verify_sha256(
            paths["sangalli_supplement"],
            LOCKED_SANGALLI_SUPPLEMENT_SHA256,
            "Sangalli supplement",
        ),
        "alm_losses": _verify_sha256(
            paths["alm_losses"], LOCKED_ALM_LOSSES_SHA256, "official ALM losses"
        ),
        "alm_train": _verify_sha256(
            paths["alm_train"], LOCKED_ALM_TRAIN_SHA256, "official ALM training"
        ),
        "alm_readme": _verify_sha256(
            paths["alm_readme"], LOCKED_ALM_README_SHA256, "official ALM README"
        ),
        "current_commands": _verify_sha256(
            paths["current_commands"],
            LOCKED_CURRENT_COMMAND_SHA256,
            "current-best commands",
        ),
        "command_history": _verify_sha256(
            paths["command_history"],
            LOCKED_COMMAND_HISTORY_SHA256,
            "current-best command history",
        ),
    }
    capsule_commit = _git_commit(paths["capsule_root"])
    alm_commit = _git_commit(paths["alm_root"])
    capsule_clean = _full_worktree_clean(paths["capsule_root"])
    alm_clean = _full_worktree_clean(paths["alm_root"])
    if capsule_commit != LOCKED_CAPSULE_COMMIT or not capsule_clean:
        raise ValueError("Official capsule source commit or cleanliness differs.")
    if alm_commit != LOCKED_ALM_COMMIT or not alm_clean:
        raise ValueError("Official ALM source commit or cleanliness differs.")
    alm_license_paths = [
        path
        for pattern in ("LICENSE*", "COPYING*")
        for path in paths["alm_root"].glob(pattern)
        if path.is_file()
    ]
    if alm_license_paths:
        raise ValueError("ALM repository license state differs from locked disclosure.")

    cidt_summary = json.loads(paths["cidt_summary"].read_text(encoding="utf-8"))
    mcl_summary = json.loads(paths["mcl_summary"].read_text(encoding="utf-8"))
    for name, summary in (("CIDT", cidt_summary), ("MCL", mcl_summary)):
        if bool(summary.get("test_data_used", True)):
            raise ValueError(f"{name} provenance indicates test data use.")
        if bool(
            summary.get(
                "validation_predictions_used",
                summary.get("validation_data_used", True),
            )
        ):
            raise ValueError(f"{name} provenance indicates validation use.")
    rows = _read_clean_train_rows(paths["cidt_predictions"])
    cohorts = _locked_cohort_summary(rows, fold=int(args.fold))
    expected_hashes = {
        "fit_index_sha256": LOCKED_FIT_INDEX_SHA256,
        "holdout_index_sha256": LOCKED_HOLDOUT_INDEX_SHA256,
        "reference_index_sha256": LOCKED_REFERENCE_INDEX_SHA256,
        "hard_index_sha256": LOCKED_HARD_INDEX_SHA256,
    }
    for key, expected in expected_hashes.items():
        if cohorts[key] != expected:
            raise ValueError(f"CapsALM cohort hash differs for {key}.")
    natural_order = locked_natural_order(
        cohorts["fit_indices"], seed=int(args.seed), epochs=int(args.epochs)
    )
    natural_hash = _ordered_index_sha256(natural_order)
    if natural_hash != LOCKED_NATURAL_ORDER_SHA256:
        raise ValueError("CapsALM natural order hash differs from protocol.")
    boundary_schedule = locked_boundary_schedule(
        cohorts["reference_indices"],
        cohorts["hard_indices"],
        seed=int(args.seed),
        epochs=int(args.epochs),
    )
    boundary_hash = boundary_schedule_sha256(boundary_schedule)
    if boundary_hash != LOCKED_BOUNDARY_SCHEDULE_SHA256:
        raise ValueError("CapsALM boundary schedule hash differs from protocol.")
    root = Path.cwd().resolve()
    tracked_clean = _tracked_worktree_clean(root)
    if not tracked_clean:
        raise ValueError("Tracked TRKH worktree must be clean for formal CapsALM A0.")
    return (
        {
            "paths": {key: str(value) for key, value in paths.items()},
            "sha256": hashes,
            "capsule_commit": capsule_commit,
            "capsule_worktree_clean": capsule_clean,
            "capsule_license_present": paths["capsule_license"].is_file(),
            "alm_commit": alm_commit,
            "alm_worktree_clean": alm_clean,
            "alm_license_present": False,
            "alm_github_last_positive_loop_discrepancy_disclosed": True,
            "repository_commit": _git_commit(root),
            "tracked_worktree_clean": tracked_clean,
            "natural_order_sha256": natural_hash,
            "boundary_schedule_sha256": boundary_hash,
            "natural_occurrences": len(natural_order),
            "boundary_events": len(boundary_schedule),
            "validation_predictions_used": False,
            "test_data_used": False,
        },
        rows,
        cohorts,
        natural_order,
        boundary_schedule,
    )


def _construct_keeper_and_adapters(
    checkpoint: Mapping[str, object],
) -> tuple[
    nn.Module,
    SupportConstrainedCapsuleAdapter,
    SupportConstrainedCapsuleAdapter,
    Dict[str, object],
]:
    model_config = checkpoint.get("model_config")
    model_state = checkpoint.get("model_state")
    class_names = checkpoint.get("class_names")
    if not isinstance(model_config, Mapping) or not isinstance(model_state, Mapping):
        raise ValueError("Keeper model config/state is invalid.")
    if not isinstance(class_names, list) or len(class_names) != NUM_CLASSES:
        raise ValueError("Keeper class order is invalid.")
    set_seed(SEED, deterministic=True)
    keeper = create_model(num_classes=NUM_CLASSES, model_config=model_config).eval()
    load_model_state(keeper, dict(model_state), strict=True)
    for parameter in keeper.parameters():
        parameter.requires_grad_(False)
    rng_before = torch.get_rng_state().clone()
    prototype = SupportConstrainedCapsuleAdapter()
    rng_after = torch.get_rng_state().clone()
    control = copy.deepcopy(prototype)
    candidate = copy.deepcopy(prototype)
    control_optimizer = torch.optim.SGD(
        control.parameters(),
        lr=LEARNING_RATE,
        momentum=MOMENTUM,
        weight_decay=WEIGHT_DECAY,
    )
    candidate_optimizer = torch.optim.SGD(
        candidate.parameters(),
        lr=LEARNING_RATE,
        momentum=MOMENTUM,
        weight_decay=WEIGHT_DECAY,
    )
    adapter_parameters = sum(parameter.numel() for parameter in prototype.parameters())
    return keeper, control, candidate, {
        "keeper_parameters": sum(parameter.numel() for parameter in keeper.parameters()),
        "adapter_parameters": adapter_parameters,
        "control_state_sha256": _state_sha256(control),
        "candidate_state_sha256": _state_sha256(candidate),
        "candidate_control_bit_exact": _state_sha256(control)
        == _state_sha256(candidate),
        "candidate_control_initial_optimizer_bit_exact": (
            control_optimizer.state_dict() == candidate_optimizer.state_dict()
        ),
        "parameter_schema": {
            name: [int(value) for value in parameter.shape]
            for name, parameter in prototype.named_parameters()
        },
        "isolated_initialization_rng_restored": torch.equal(rng_before, rng_after),
        "residual_head_weight_zero": bool(
            torch.count_nonzero(prototype.residual_head.weight).item() == 0
        ),
        "residual_head_bias_zero": bool(
            torch.count_nonzero(prototype.residual_head.bias).item() == 0
        ),
        "vote_initialization": "truncated_normal_std_0p1_clip_2std",
        "class_capsule_bias_initialization": 0.1,
    }


def _subset_cache(
    cache: Mapping[str, Tensor], indices: Sequence[int]
) -> Dict[str, Tensor]:
    positions = _cache_positions(cache)
    requested = torch.tensor([int(value) for value in indices], dtype=torch.long)
    if int(requested.max().item()) >= int(positions.numel()):
        raise ValueError("Cache subset index is out of range.")
    rows = positions[requested]
    if bool((rows < 0).any().item()):
        raise ValueError("Cache subset contains a row outside the cache.")
    result = {key: value[rows].clone() for key, value in cache.items()}
    if result["sample_indices"].tolist() != [int(value) for value in indices]:
        raise ValueError("Cache subset changed sample order.")
    return result


def _condition_loader(
    *,
    name: str,
    dataset: MangoYOLOCropDataset,
    transform,
    indices: Sequence[int],
    args: argparse.Namespace,
    context: str,
):
    if name == "clean":
        return _make_loader(
            base_dataset=dataset,
            transform=transform,
            indices=indices,
            batch_size=int(args.batch_size),
            num_workers=int(args.num_workers),
            context=context,
            seed=int(args.seed) + 200,
        )
    lighting = {
        condition: (brightness, contrast)
        for condition, brightness, contrast in LIGHTING_CONDITIONS
    }
    brightness, contrast = lighting[name]
    return _make_lighting_loader(
        base_dataset=dataset,
        transform=transform,
        indices=indices,
        brightness=brightness,
        contrast=contrast,
        batch_size=int(args.batch_size),
        num_workers=int(args.num_workers),
        context=context,
        seed=int(args.seed) + 201 + list(lighting).index(name),
    )


def _extract_condition_cache(
    *,
    keeper: nn.Module,
    dataset: MangoYOLOCropDataset,
    transform,
    indices: Sequence[int],
    condition: str,
    args: argparse.Namespace,
    device: torch.device,
    amp_dtype: torch.dtype,
) -> tuple[Dict[str, Tensor], Dict[str, object]]:
    loader, loader_summary = _condition_loader(
        name=condition,
        dataset=dataset,
        transform=transform,
        indices=indices,
        args=args,
        context=f"capsalm_boundary_cache_{condition}",
    )
    cache: Optional[Dict[str, Tensor]] = None
    offset = 0
    started = time.perf_counter()
    keeper = keeper.to(device).eval()
    with torch.inference_mode():
        for images_cpu, targets_cpu, metadata_cpu in loader:
            images = images_cpu.to(device=device, non_blocking=True)
            with torch.autocast(device_type="cuda", dtype=amp_dtype, enabled=True):
                raw_logits, features = _forward_classification_with_metadata(
                    keeper, images, metadata_cpu, device=device
                )
            if not isinstance(features, Mapping):
                raise ValueError("Boundary cache keeper forward returned no features.")
            patches, patch_indices, token_valid = _feature_tensors(features)
            sample_indices = metadata_cpu.get("sample_index")
            if not torch.is_tensor(sample_indices):
                raise ValueError("Boundary cache batch is missing sample_index.")
            count = int(targets_cpu.numel())
            if cache is None:
                if tuple(patches.shape[1:]) != (EXPECTED_PATCHES, EMBED_DIM):
                    raise ValueError("Boundary cache patch shape differs from protocol.")
                cache = {
                    "patches": torch.empty(
                        len(indices), EXPECTED_PATCHES, EMBED_DIM, dtype=torch.float32
                    ),
                    "patch_indices": torch.empty(
                        len(indices), EXPECTED_PATCHES, dtype=torch.long
                    ),
                    "token_valid": torch.empty(
                        len(indices), EXPECTED_PATCHES, dtype=torch.bool
                    ),
                    "raw_logits": torch.empty(len(indices), NUM_CLASSES),
                    "targets": torch.empty(len(indices), dtype=torch.long),
                    "sample_indices": torch.empty(len(indices), dtype=torch.long),
                }
            target_slice = slice(offset, offset + count)
            cache["patches"][target_slice].copy_(patches.detach().float().cpu())
            cache["patch_indices"][target_slice].copy_(
                patch_indices.detach().long().cpu()
            )
            cache["token_valid"][target_slice].copy_(token_valid.detach().bool().cpu())
            cache["raw_logits"][target_slice].copy_(raw_logits.detach().float().cpu())
            cache["targets"][target_slice].copy_(targets_cpu.detach().long())
            cache["sample_indices"][target_slice].copy_(
                sample_indices.detach().long()
            )
            offset += count
    if cache is None or offset != len(indices):
        raise RuntimeError(f"Boundary cache extraction is incomplete for {condition}.")
    expected = [int(value) for value in indices]
    if cache["sample_indices"].tolist() != expected:
        raise ValueError(f"Boundary cache changed {condition} sample order.")
    return cache, {
        "condition": condition,
        "rows": len(indices),
        "sample_index_sha256": _ordered_index_sha256(expected),
        "patch_tensor_sha256": _stream_tensor_sha256(cache["patches"]),
        "raw_logit_sha256": _stream_tensor_sha256(cache["raw_logits"]),
        "all_finite": bool(
            torch.isfinite(cache["patches"]).all()
            and torch.isfinite(cache["raw_logits"]).all()
        ),
        "loader": loader_summary,
        "elapsed_seconds": float(time.perf_counter() - started),
    }


def _boundary_event_batch(
    caches: Mapping[str, Mapping[str, Tensor]],
    event: Mapping[str, object],
    *,
    device: torch.device,
) -> tuple[Dict[str, Tensor], list[int], list[int]]:
    condition = str(event["condition"])
    cache = caches[condition]
    positive_indices = [int(value) for value in event["positive_indices"]]
    negative_indices = [int(value) for value in event["negative_indices"]]
    indices = [*positive_indices, *negative_indices]
    batch = _batch_from_cache(
        cache, _cache_positions(cache), indices, device=device
    )
    return batch, positive_indices, negative_indices


def _initial_gradient_audit(
    *,
    prototype: SupportConstrainedCapsuleAdapter,
    natural_cache: Mapping[str, Tensor],
    boundary_caches: Mapping[str, Mapping[str, Tensor]],
    natural_order: Sequence[int],
    boundary_schedule: Sequence[Mapping[str, object]],
    reference_indices: Sequence[int],
    device: torch.device,
) -> Dict[str, object]:
    natural = _batch_from_cache(
        natural_cache,
        _cache_positions(natural_cache),
        natural_order[:NATURAL_BATCH_SIZE],
        device=device,
    )
    boundary, positive_indices, _negative_indices = _boundary_event_batch(
        boundary_caches, boundary_schedule[0], device=device
    )
    reference_position = {
        int(sample_index): position
        for position, sample_index in enumerate(reference_indices)
    }
    multiplier_rows = torch.tensor(
        [reference_position[index] for index in positive_indices],
        device=device,
        dtype=torch.long,
    )
    zeros = torch.zeros(len(reference_indices), device=device)

    def forward_losses(adapter: SupportConstrainedCapsuleAdapter):
        natural_result = adapter.forward_sparse(
            natural["patches"],
            natural["patch_indices"],
            natural["token_valid"],
            natural["raw_logits"],
        )
        boundary_result = adapter.forward_sparse(
            boundary["patches"],
            boundary["patch_indices"],
            boundary["token_valid"],
            boundary["raw_logits"],
        )
        constraints = normalized_alm_constraints(
            boundary_result["logits"][:BOUNDARY_HALF_BATCH],
            boundary_result["logits"][BOUNDARY_HALF_BATCH:],
            boundary["raw_logits"][:BOUNDARY_HALF_BATCH],
        )
        alm = augmented_lagrangian_loss(
            constraints["q_rank"],
            constraints["q_support"],
            zeros[multiplier_rows],
            zeros[multiplier_rows],
        )
        ce = F.cross_entropy(natural_result["logits"], natural["targets"])
        return ce, constraints, alm

    ce_only = copy.deepcopy(prototype).to(device).train()
    ce, _constraints, _alm = forward_losses(ce_only)
    ce.backward()
    ce_gradients = _parameter_gradient_map(ce_only)

    control = copy.deepcopy(prototype).to(device).train()
    control_ce, control_constraints, control_alm = forward_losses(control)
    control_loss = control_ce + 0.0 * control_alm["loss"].detach()
    control_loss.backward()
    control_gradients = _parameter_gradient_map(control)

    candidate = copy.deepcopy(prototype).to(device).train()
    candidate_ce, candidate_constraints, candidate_alm = forward_losses(candidate)
    candidate_loss = candidate_ce + candidate_alm["loss"]
    candidate_loss.backward()
    candidate_gradients = _parameter_gradient_map(candidate)
    control_matches_ce = ce_gradients.keys() == control_gradients.keys() and all(
        ce_gradients[name] is not None
        and control_gradients[name] is not None
        and torch.equal(ce_gradients[name], control_gradients[name])
        for name in ce_gradients
    )
    candidate_differs = any(
        candidate_gradients[name] is not None
        and ce_gradients[name] is not None
        and not torch.equal(candidate_gradients[name], ce_gradients[name])
        for name in ce_gradients
    )
    rank_only = copy.deepcopy(prototype).to(device).train()
    _rank_ce, rank_constraints, rank_alm = forward_losses(rank_only)
    (rank_alm["rank_penalty"] + rank_alm["rank_dual"]).backward()
    rank_gradients = _parameter_gradient_map(rank_only)
    return {
        "natural_rows": int(natural["targets"].numel()),
        "boundary_positive_rows": BOUNDARY_HALF_BATCH,
        "boundary_negative_rows": BOUNDARY_HALF_BATCH,
        "control_detached_gradient_matches_pure_ce_bit_exact": control_matches_ce,
        "candidate_gradient_differs_from_ce": candidate_differs,
        "candidate_gradient_norm": _gradient_norm(candidate_gradients),
        "candidate_gradient_finite": all(
            value is not None and bool(torch.isfinite(value).all())
            for value in candidate_gradients.values()
        ),
        "rank_gradient_norm": _gradient_norm(rank_gradients),
        "rank_gradient_nonzero": _gradient_norm(rank_gradients) > 0.0,
        "initial_support_violation_max": float(
            candidate_constraints["q_support"].detach().amax().item()
        ),
        "initial_support_violation_exact_zero": bool(
            torch.count_nonzero(candidate_constraints["q_support"]).item() == 0
        ),
        "initial_rank_violation_mean": float(
            candidate_constraints["q_rank"].detach().mean().item()
        ),
        "control_rank_violation_mean": float(
            control_constraints["q_rank"].detach().mean().item()
        ),
        "candidate_loss": float(candidate_loss.detach().item()),
        "control_loss": float(control_loss.detach().item()),
    }


def _train_variant(
    *,
    name: str,
    prototype: SupportConstrainedCapsuleAdapter,
    natural_cache: Mapping[str, Tensor],
    boundary_caches: Mapping[str, Mapping[str, Tensor]],
    natural_order: Sequence[int],
    boundary_schedule: Sequence[Mapping[str, object]],
    reference_indices: Sequence[int],
    candidate: bool,
    args: argparse.Namespace,
    device: torch.device,
) -> tuple[SupportConstrainedCapsuleAdapter, Dict[str, object]]:
    adapter = copy.deepcopy(prototype).to(device).train()
    initial = {
        key: value.detach().cpu().clone() for key, value in adapter.state_dict().items()
    }
    optimizer = torch.optim.SGD(
        adapter.parameters(),
        lr=float(args.learning_rate),
        momentum=float(args.momentum),
        weight_decay=float(args.weight_decay),
    )
    natural_positions = _cache_positions(natural_cache)
    reference_position = {
        int(sample_index): position
        for position, sample_index in enumerate(reference_indices)
    }
    lambda_rank = torch.zeros(len(CONDITIONS), len(reference_indices), device=device)
    lambda_support = torch.zeros_like(lambda_rank)
    fit_rows = int(natural_cache["sample_indices"].numel())
    natural_batches = math.ceil(fit_rows / int(args.batch_size))
    boundary_batches = interleaved_boundary_batches(natural_batches=natural_batches)
    event_by_batch = {
        natural_batch: event_offset
        for event_offset, natural_batch in enumerate(boundary_batches)
    }
    history = []
    gradient_seen = {parameter_name: False for parameter_name, _ in adapter.named_parameters()}
    all_gradients_finite = True
    all_losses_finite = True
    natural_offset = 0
    boundary_offset = 0
    started = time.perf_counter()
    for epoch in range(int(args.epochs)):
        epoch_order = natural_order[natural_offset : natural_offset + fit_rows]
        natural_offset += fit_rows
        epoch_events = boundary_schedule[
            boundary_offset : boundary_offset + BOUNDARY_EVENTS_PER_EPOCH
        ]
        boundary_offset += BOUNDARY_EVENTS_PER_EPOCH
        sums = {
            "loss": 0.0,
            "ce": 0.0,
            "rank_penalty": 0.0,
            "rank_dual": 0.0,
            "support_penalty": 0.0,
            "support_dual": 0.0,
            "rank_violation": 0.0,
            "support_violation": 0.0,
            "natural_rows": 0,
            "natural_batches": 0,
            "boundary_events": 0,
            "boundary_positive_rows": 0,
        }
        for batch_start in range(0, fit_rows, int(args.batch_size)):
            batch_number = batch_start // int(args.batch_size)
            sample_indices = epoch_order[
                batch_start : batch_start + int(args.batch_size)
            ]
            natural = _batch_from_cache(
                natural_cache,
                natural_positions,
                sample_indices,
                device=device,
            )
            optimizer.zero_grad(set_to_none=True)
            natural_result = adapter.forward_sparse(
                natural["patches"],
                natural["patch_indices"],
                natural["token_valid"],
                natural["raw_logits"],
            )
            ce = F.cross_entropy(natural_result["logits"], natural["targets"])
            loss = ce
            constraints = None
            alm = None
            multiplier_rows = None
            condition_index = None
            if batch_number in event_by_batch:
                event = epoch_events[event_by_batch[batch_number]]
                boundary, positive_indices, _negative_indices = _boundary_event_batch(
                    boundary_caches, event, device=device
                )
                boundary_result = adapter.forward_sparse(
                    boundary["patches"],
                    boundary["patch_indices"],
                    boundary["token_valid"],
                    boundary["raw_logits"],
                )
                constraints = normalized_alm_constraints(
                    boundary_result["logits"][:BOUNDARY_HALF_BATCH],
                    boundary_result["logits"][BOUNDARY_HALF_BATCH:],
                    boundary["raw_logits"][:BOUNDARY_HALF_BATCH],
                    margin=float(args.rank_margin),
                )
                condition_index = int(event["condition_index"])
                multiplier_rows = torch.tensor(
                    [reference_position[index] for index in positive_indices],
                    device=device,
                    dtype=torch.long,
                )
                alm = augmented_lagrangian_loss(
                    constraints["q_rank"],
                    constraints["q_support"],
                    lambda_rank[condition_index, multiplier_rows],
                    lambda_support[condition_index, multiplier_rows],
                )
                if candidate:
                    loss = loss + alm["loss"]
                else:
                    loss = loss + 0.0 * alm["loss"].detach()
            if not bool(torch.isfinite(loss).item()):
                all_losses_finite = False
                raise FloatingPointError(f"CapsALM {name} loss became non-finite.")
            loss.backward()
            for parameter_name, parameter in adapter.named_parameters():
                gradient = parameter.grad
                if gradient is None or not bool(torch.isfinite(gradient).all()):
                    all_gradients_finite = False
                elif torch.count_nonzero(gradient).item() > 0:
                    gradient_seen[parameter_name] = True
            optimizer.step()
            if constraints is not None and multiplier_rows is not None:
                assert condition_index is not None and alm is not None
                with torch.no_grad():
                    lambda_rank[condition_index, multiplier_rows] += constraints[
                        "q_rank"
                    ].detach()
                    lambda_support[condition_index, multiplier_rows] += constraints[
                        "q_support"
                    ].detach()
                sums["rank_penalty"] += float(alm["rank_penalty"].detach().item())
                sums["rank_dual"] += float(alm["rank_dual"].detach().item())
                sums["support_penalty"] += float(
                    alm["support_penalty"].detach().item()
                )
                sums["support_dual"] += float(alm["support_dual"].detach().item())
                sums["rank_violation"] += float(
                    constraints["q_rank"].detach().mean().item()
                )
                sums["support_violation"] += float(
                    constraints["q_support"].detach().mean().item()
                )
                sums["boundary_events"] += 1
                sums["boundary_positive_rows"] += BOUNDARY_HALF_BATCH
            count = int(natural["targets"].numel())
            sums["loss"] += float(loss.detach().item()) * count
            sums["ce"] += float(ce.detach().item()) * count
            sums["natural_rows"] += count
            sums["natural_batches"] += 1
        event_count = max(1, int(sums["boundary_events"]))
        history.append(
            {
                "variant": name,
                "epoch": epoch,
                "natural_rows": int(sums["natural_rows"]),
                "natural_batches": int(sums["natural_batches"]),
                "boundary_events": int(sums["boundary_events"]),
                "boundary_positive_rows": int(sums["boundary_positive_rows"]),
                "loss": sums["loss"] / int(sums["natural_rows"]),
                "ce": sums["ce"] / int(sums["natural_rows"]),
                "rank_penalty": sums["rank_penalty"] / event_count,
                "rank_dual": sums["rank_dual"] / event_count,
                "support_penalty": sums["support_penalty"] / event_count,
                "support_dual": sums["support_dual"] / event_count,
                "rank_violation": sums["rank_violation"] / event_count,
                "support_violation": sums["support_violation"] / event_count,
                "lambda_rank_mean": float(lambda_rank.mean().item()),
                "lambda_rank_max": float(lambda_rank.max().item()),
                "lambda_support_mean": float(lambda_support.mean().item()),
                "lambda_support_max": float(lambda_support.max().item()),
            }
        )
        print(json.dumps(history[-1]), flush=True)
    final = adapter.state_dict()
    movement = {
        key: {
            "initial_sha256": _tensor_sha256(value),
            "final_sha256": _tensor_sha256(final[key].detach().cpu()),
            "changed": not torch.equal(value, final[key].detach().cpu()),
        }
        for key, value in initial.items()
    }
    dual_finite_nonnegative = bool(
        torch.isfinite(lambda_rank).all()
        and torch.isfinite(lambda_support).all()
        and (lambda_rank >= 0).all()
        and (lambda_support >= 0).all()
    )
    dual = {
        "rank": lambda_rank.detach().cpu(),
        "support": lambda_support.detach().cpu(),
    }
    adapter = adapter.cpu().eval()
    gc.collect()
    torch.cuda.empty_cache()
    return adapter, {
        "name": name,
        "epochs": int(args.epochs),
        "natural_rows": len(natural_order),
        "natural_batches": int(args.epochs) * natural_batches,
        "boundary_events": len(boundary_schedule),
        "boundary_positive_rows": len(boundary_schedule) * BOUNDARY_HALF_BATCH,
        "natural_order_sha256": _ordered_index_sha256(natural_order),
        "boundary_schedule_sha256": boundary_schedule_sha256(boundary_schedule),
        "interleave_natural_batch_indices": boundary_batches,
        "history": history,
        "initial_state_sha256": _state_dict_sha256(initial),
        "final_state_sha256": _state_sha256(adapter),
        "gradient_seen_nonzero": gradient_seen,
        "all_trainable_gradients_seen": all(gradient_seen.values()),
        "all_gradients_finite": all_gradients_finite,
        "all_losses_finite": all_losses_finite,
        "movement": movement,
        "all_trainable_parameters_changed": all(
            bool(row["changed"]) for row in movement.values()
        ),
        "dual_finite_nonnegative": dual_finite_nonnegative,
        "rank_multiplier_positive": bool(torch.count_nonzero(dual["rank"]).item()),
        "rank_multiplier_mean": float(dual["rank"].mean().item()),
        "rank_multiplier_std": float(dual["rank"].std(unbiased=False).item()),
        "rank_multiplier_max": float(dual["rank"].max().item()),
        "support_multiplier_mean": float(dual["support"].mean().item()),
        "support_multiplier_max": float(dual["support"].max().item()),
        "dual_tensors": dual,
        "elapsed_seconds": float(time.perf_counter() - started),
    }


def _routing_statistics(
    result: Mapping[str, Tensor], token_valid: Tensor
) -> Dict[str, Tensor]:
    valid = token_valid.to(dtype=torch.bool, device=result["couplings"].device)
    class1 = result["couplings"][:, :, FOCUS_CLASS] * valid.to(
        dtype=result["couplings"].dtype
    )
    spatial = class1 / class1.sum(dim=1, keepdim=True).clamp_min(1e-12)
    entropy = -(spatial.clamp_min(1e-12).log() * spatial).sum(dim=1)
    valid_count = valid.sum(dim=1).clamp_min(2).to(dtype=entropy.dtype)
    return {
        "normalized_entropy": entropy / valid_count.log(),
        "effective_patch_count": entropy.exp(),
        "class1_spatial": spatial,
        "parent_sum_error": (
            result["couplings"].sum(dim=2)[valid] - 1.0
        ).abs(),
        "invalid_mass": result["couplings"][~valid].abs(),
    }


def _mechanism_accumulator() -> Dict[str, object]:
    return {
        "sample_indices": [],
        "targets": [],
        "p1": [],
        "raw_margin": [],
        "deployed_margin": [],
        "residual_margin": [],
        "normalized_entropy": [],
        "effective_patch_count": [],
        "parent_sum_error_max": 0.0,
        "invalid_coupling_mass": 0.0,
        "dynamic_uniform_maximum_errors": {
            "couplings": 0.0,
            "class_capsules": 0.0,
            "capsule_lengths": 0.0,
            "residual_logits": 0.0,
            "logits": 0.0,
        },
        "dynamic_uniform_logit_changed_rows": 0,
    }


def _update_mechanism_accumulator(
    accumulator: Dict[str, object],
    *,
    result: Mapping[str, Tensor],
    raw_logits: Tensor,
    targets: Tensor,
    sample_indices: Tensor,
    token_valid: Tensor,
    uniform_result: Optional[Mapping[str, Tensor]] = None,
) -> None:
    routing = _routing_statistics(result, token_valid)
    probabilities = F.softmax(result["logits"].float(), dim=1)
    restricted = torch.tensor(
        RESTRICTED_NEGATIVE_CLASSES,
        device=result["residual_logits"].device,
        dtype=torch.long,
    )
    residual_margin = (
        result["residual_logits"][:, FOCUS_CLASS]
        - result["residual_logits"].index_select(1, restricted).amax(dim=1)
    )
    accumulator["sample_indices"].append(sample_indices.detach().long().cpu())
    accumulator["targets"].append(targets.detach().long().cpu())
    accumulator["p1"].append(probabilities[:, FOCUS_CLASS].detach().cpu())
    accumulator["raw_margin"].append(class1_margin(raw_logits).detach().cpu())
    accumulator["deployed_margin"].append(
        class1_margin(result["logits"]).detach().cpu()
    )
    accumulator["residual_margin"].append(residual_margin.detach().cpu())
    accumulator["normalized_entropy"].append(
        routing["normalized_entropy"].detach().cpu()
    )
    accumulator["effective_patch_count"].append(
        routing["effective_patch_count"].detach().cpu()
    )
    if routing["parent_sum_error"].numel():
        accumulator["parent_sum_error_max"] = max(
            float(accumulator["parent_sum_error_max"]),
            float(routing["parent_sum_error"].amax().item()),
        )
    accumulator["invalid_coupling_mass"] = float(
        accumulator["invalid_coupling_mass"]
    ) + float(routing["invalid_mass"].sum().item())
    if uniform_result is not None:
        errors = accumulator["dynamic_uniform_maximum_errors"]
        for key in errors:
            errors[key] = max(
                float(errors[key]),
                float((result[key] - uniform_result[key]).abs().amax().item()),
            )
        accumulator["dynamic_uniform_logit_changed_rows"] = int(
            accumulator["dynamic_uniform_logit_changed_rows"]
        ) + int(
            ((result["logits"] - uniform_result["logits"]).abs().amax(dim=1) > 0)
            .sum()
            .item()
        )


def _finalize_mechanism(
    accumulator: Mapping[str, object], *, hard_holdout_indices: Sequence[int]
) -> Dict[str, object]:
    sample_indices = torch.cat(list(accumulator["sample_indices"]))
    targets = torch.cat(list(accumulator["targets"]))
    p1 = torch.cat(list(accumulator["p1"])).float()
    raw_margin = torch.cat(list(accumulator["raw_margin"])).float()
    deployed_margin = torch.cat(list(accumulator["deployed_margin"])).float()
    residual_margin = torch.cat(list(accumulator["residual_margin"])).float()
    normalized_entropy = torch.cat(list(accumulator["normalized_entropy"])).float()
    effective_count = torch.cat(list(accumulator["effective_patch_count"])).float()
    positive_mask = targets == FOCUS_CLASS
    hard_values = torch.tensor(
        [int(value) for value in hard_holdout_indices], dtype=torch.long
    )
    hard_mask = torch.isin(sample_indices, hard_values)
    if not bool(positive_mask.any()) or not bool(hard_mask.any()):
        raise ValueError("Holdout constraint mechanism cohort is empty.")
    pairwise = F.relu(
        -(p1[positive_mask, None] - p1[None, hard_mask]) + RANK_MARGIN
    )
    q_rank = pairwise.mean(dim=1)
    q_support = F.relu(raw_margin[positive_mask] - deployed_margin[positive_mask])
    direction_mask = positive_mask | torch.isin(
        targets, torch.tensor(RESTRICTED_NEGATIVE_CLASSES, dtype=torch.long)
    )
    direction_labels = positive_mask[direction_mask].long().numpy()
    direction_scores = residual_margin[direction_mask].numpy()
    auroc = (
        float(roc_auc_score(direction_labels, direction_scores))
        if np.unique(direction_labels).size == 2
        else None
    )
    finite_tensors = (
        p1,
        raw_margin,
        deployed_margin,
        residual_margin,
        normalized_entropy,
        effective_count,
        q_rank,
        q_support,
    )
    return {
        "rows": int(targets.numel()),
        "positive_rows": int(positive_mask.sum().item()),
        "hard_negative_rows": int(hard_mask.sum().item()),
        "hard_negative_index_sha256": _ordered_index_sha256(
            sample_indices[hard_mask].tolist()
        ),
        "rank_violation_mean": float(q_rank.mean().item()),
        "rank_violation_p95": float(torch.quantile(q_rank, 0.95).item()),
        "support_violation_mean": float(q_support.mean().item()),
        "support_violation_p95": float(torch.quantile(q_support, 0.95).item()),
        "class1_capsule_residual_margin_auroc_vs_0_2_4": auroc,
        "normalized_class1_routing_entropy_mean": float(
            normalized_entropy.mean().item()
        ),
        "normalized_class1_routing_entropy_min": float(
            normalized_entropy.min().item()
        ),
        "normalized_class1_routing_entropy_max": float(
            normalized_entropy.max().item()
        ),
        "effective_class1_routed_patch_count_mean": float(
            effective_count.mean().item()
        ),
        "effective_class1_routed_patch_count_min": float(
            effective_count.min().item()
        ),
        "effective_class1_routed_patch_count_max": float(
            effective_count.max().item()
        ),
        "coupling_parent_sum_maximum_error": float(
            accumulator["parent_sum_error_max"]
        ),
        "invalid_coupling_mass": float(accumulator["invalid_coupling_mass"]),
        "dynamic_uniform_maximum_errors": dict(
            accumulator["dynamic_uniform_maximum_errors"]
        ),
        "dynamic_uniform_logit_changed_rows": int(
            accumulator["dynamic_uniform_logit_changed_rows"]
        ),
        "all_finite": all(bool(torch.isfinite(value).all()) for value in finite_tensors),
    }


def _evaluate_conditions(
    *,
    keeper: nn.Module,
    control: SupportConstrainedCapsuleAdapter,
    candidate: SupportConstrainedCapsuleAdapter,
    dataset: MangoYOLOCropDataset,
    transform,
    indices: Sequence[int],
    rows: Sequence[CleanTrainRow],
    hard_holdout_indices: Sequence[int],
    args: argparse.Namespace,
    device: torch.device,
    amp_dtype: torch.dtype,
) -> tuple[
    Dict[str, list[Dict[str, object]]],
    Dict[str, list[Dict[str, object]]],
    Dict[str, list[Dict[str, object]]],
    Dict[str, Dict[str, object]],
    Dict[str, object],
]:
    keeper = keeper.to(device).eval()
    control_device = copy.deepcopy(control).to(device).eval()
    candidate_device = copy.deepcopy(candidate).to(device).eval()
    for module in (control_device, candidate_device):
        for parameter in module.parameters():
            parameter.requires_grad_(False)
    raw_conditions: Dict[str, list[Dict[str, object]]] = {}
    control_conditions: Dict[str, list[Dict[str, object]]] = {}
    candidate_conditions: Dict[str, list[Dict[str, object]]] = {}
    mechanisms: Dict[str, Dict[str, object]] = {}
    loaders: Dict[str, object] = {}
    expected = [int(value) for value in indices]
    for condition in CONDITIONS:
        loader, loader_summary = _condition_loader(
            name=condition,
            dataset=dataset,
            transform=transform,
            indices=indices,
            args=args,
            context=f"capsalm_holdout_{condition}",
        )
        loaders[condition] = loader_summary
        raw_rows: list[Dict[str, object]] = []
        control_rows: list[Dict[str, object]] = []
        candidate_rows: list[Dict[str, object]] = []
        control_accumulator = _mechanism_accumulator()
        candidate_accumulator = _mechanism_accumulator()
        processed = 0
        started = time.perf_counter()
        with torch.inference_mode():
            for images_cpu, targets_cpu, metadata_cpu in loader:
                images = images_cpu.to(device=device, non_blocking=True)
                targets = targets_cpu.to(device=device, dtype=torch.long)
                with torch.autocast(device_type="cuda", dtype=amp_dtype, enabled=True):
                    raw_logits, features = _forward_classification_with_metadata(
                        keeper, images, metadata_cpu, device=device
                    )
                if not isinstance(features, Mapping):
                    raise ValueError("Holdout keeper forward returned no features.")
                patches, patch_indices, token_valid = _feature_tensors(features)
                patches = patches.float()
                raw_logits = raw_logits.float()
                sample_tensor = metadata_cpu.get("sample_index")
                if not torch.is_tensor(sample_tensor):
                    raise ValueError("Holdout batch is missing sample_index.")
                sample_device = sample_tensor.to(device=device, dtype=torch.long)
                control_result = control_device.forward_sparse(
                    patches, patch_indices, token_valid, raw_logits
                )
                candidate_result = candidate_device.forward_sparse(
                    patches, patch_indices, token_valid, raw_logits
                )
                candidate_uniform = candidate_device.forward_sparse(
                    patches,
                    patch_indices,
                    token_valid,
                    raw_logits,
                    routing_iterations=1,
                )
                _update_mechanism_accumulator(
                    control_accumulator,
                    result=control_result,
                    raw_logits=raw_logits,
                    targets=targets,
                    sample_indices=sample_device,
                    token_valid=token_valid,
                )
                _update_mechanism_accumulator(
                    candidate_accumulator,
                    result=candidate_result,
                    raw_logits=raw_logits,
                    targets=targets,
                    sample_indices=sample_device,
                    token_valid=token_valid,
                    uniform_result=candidate_uniform,
                )
                raw_prob = F.softmax(raw_logits, dim=1)
                control_prob = F.softmax(control_result["logits"], dim=1)
                candidate_prob = F.softmax(candidate_result["logits"], dim=1)
                for local, sample_index in enumerate(sample_tensor.tolist()):
                    target = int(targets_cpu[local].item())
                    raw_rows.append(
                        _prediction_row(
                            sample_index=int(sample_index),
                            target=target,
                            probabilities=raw_prob[local],
                        )
                    )
                    control_rows.append(
                        _prediction_row(
                            sample_index=int(sample_index),
                            target=target,
                            probabilities=control_prob[local],
                        )
                    )
                    candidate_rows.append(
                        _prediction_row(
                            sample_index=int(sample_index),
                            target=target,
                            probabilities=candidate_prob[local],
                        )
                    )
                processed += int(targets.numel())
                if processed % 512 < int(targets.numel()) or processed == len(indices):
                    print(
                        json.dumps(
                            {
                                "phase": "holdout_evaluation",
                                "condition": condition,
                                "processed": processed,
                                "rows": len(indices),
                                "elapsed_seconds": time.perf_counter() - started,
                            }
                        ),
                        flush=True,
                    )
        for name, values in (
            ("raw", raw_rows),
            ("control", control_rows),
            ("candidate", candidate_rows),
        ):
            if [int(row["sample_index"]) for row in values] != expected:
                raise ValueError(f"{condition} {name} changed holdout order.")
        raw_conditions[condition] = raw_rows
        control_conditions[condition] = control_rows
        candidate_conditions[condition] = candidate_rows
        mechanisms[condition] = {
            "control": _finalize_mechanism(
                control_accumulator, hard_holdout_indices=hard_holdout_indices
            ),
            "candidate": _finalize_mechanism(
                candidate_accumulator, hard_holdout_indices=hard_holdout_indices
            ),
        }
    clean_raw_mismatches = sum(
        int(row["prediction"])
        != int(rows[int(row["sample_index"])].keeper_prediction)
        for row in raw_conditions["clean"]
    )
    del control_device, candidate_device
    gc.collect()
    torch.cuda.empty_cache()
    return (
        raw_conditions,
        control_conditions,
        candidate_conditions,
        mechanisms,
        {
            "loaders": loaders,
            "clean_raw_cidt_argmax_mismatches": clean_raw_mismatches,
            "hard_holdout_rows": len(hard_holdout_indices),
            "hard_holdout_index_sha256": _ordered_index_sha256(
                hard_holdout_indices
            ),
        },
    )


def _load_mcl_comparator(
    *, paths: Mapping[str, Path]
) -> tuple[
    Dict[str, list[Dict[str, object]]],
    Dict[str, list[Dict[str, object]]],
    Dict[str, Dict[str, object]],
    Dict[str, object],
]:
    raw_conditions = {condition: [] for condition in CONDITIONS}
    candidate_conditions = {condition: [] for condition in CONDITIONS}
    with paths["mcl_predictions"].open(
        "r", encoding="utf-8-sig", newline=""
    ) as handle:
        for source in csv.DictReader(handle):
            condition = str(source["condition"])
            if condition not in raw_conditions:
                raise ValueError(f"MCL comparator has unexpected condition {condition}.")
            for name, destination in (
                ("raw", raw_conditions),
                ("candidate", candidate_conditions),
            ):
                row: Dict[str, object] = {
                    "sample_index": int(source["sample_index"]),
                    "target": int(source["target"]),
                    "prediction": int(source[f"{name}_prediction"]),
                }
                for class_index in range(NUM_CLASSES):
                    row[f"prob_{class_index}"] = float(
                        source[f"{name}_prob_{class_index}"]
                    )
                destination[condition].append(row)
    replay = _build_comparisons(
        raw_conditions=raw_conditions,
        control_conditions=raw_conditions,
        candidate_conditions=candidate_conditions,
    )
    summary = json.loads(paths["mcl_summary"].read_text(encoding="utf-8"))
    expected = summary["comparisons"]
    comparison_exact = all(
        replay[condition]["raw_candidate"]
        == expected[condition]["raw_candidate"]
        for condition in CONDITIONS
    )
    return raw_conditions, candidate_conditions, replay, {
        "rows": sum(len(values) for values in raw_conditions.values()),
        "condition_rows": {
            condition: len(values) for condition, values in raw_conditions.items()
        },
        "summary_comparisons_exact": comparison_exact,
        "sample_order_exact": all(
            [int(row["sample_index"]) for row in raw_conditions[condition]]
            == [int(row["sample_index"]) for row in candidate_conditions[condition]]
            for condition in CONDITIONS
        ),
        "probabilities_locked_by_prediction_sha256": _sha256(
            paths["mcl_predictions"]
        )
        == LOCKED_MCL_PREDICTIONS_SHA256,
    }


def _compare_raw_to_mcl(
    current: Mapping[str, Sequence[Mapping[str, object]]],
    prior: Mapping[str, Sequence[Mapping[str, object]]],
) -> Dict[str, object]:
    maximum_error = 0.0
    argmax_mismatches = 0
    order_exact = True
    for condition in CONDITIONS:
        if len(current[condition]) != len(prior[condition]):
            order_exact = False
            continue
        for current_row, prior_row in zip(current[condition], prior[condition]):
            order_exact = order_exact and (
                int(current_row["sample_index"]) == int(prior_row["sample_index"])
                and int(current_row["target"]) == int(prior_row["target"])
            )
            argmax_mismatches += int(current_row["prediction"] != prior_row["prediction"])
            for class_index in range(NUM_CLASSES):
                maximum_error = max(
                    maximum_error,
                    abs(
                        float(current_row[f"prob_{class_index}"])
                        - float(prior_row[f"prob_{class_index}"])
                    ),
                )
    return {
        "sample_order_exact": order_exact,
        "argmax_mismatches": argmax_mismatches,
        "maximum_probability_error": maximum_error,
        "probabilities_bit_exact": maximum_error == 0.0,
    }


def _dual_diagnostics(
    training: Mapping[str, object],
    *,
    reference_indices: Sequence[int],
    rows: Sequence[CleanTrainRow],
) -> Dict[str, object]:
    dual = training["dual_tensors"]
    rank = dual["rank"].float()
    support = dual["support"].float()
    rank_by_row = rank.sum(dim=0)
    source_totals: Dict[str, float] = {}
    for position, sample_index in enumerate(reference_indices):
        source = rows[int(sample_index)].source_stem
        source_totals[source] = source_totals.get(source, 0.0) + float(
            rank_by_row[position].item()
        )
    total = sum(source_totals.values())
    source_shares = {
        source: value / total if total > 0.0 else 0.0
        for source, value in source_totals.items()
    }
    return {
        "rank_shape": list(rank.shape),
        "support_shape": list(support.shape),
        "finite_nonnegative": bool(
            torch.isfinite(rank).all()
            and torch.isfinite(support).all()
            and (rank >= 0).all()
            and (support >= 0).all()
        ),
        "rank_nonzero": int(torch.count_nonzero(rank).item()),
        "rank_mean": float(rank.mean().item()),
        "rank_std": float(rank.std(unbiased=False).item()),
        "rank_max": float(rank.max().item()),
        "support_nonzero": int(torch.count_nonzero(support).item()),
        "support_mean": float(support.mean().item()),
        "support_max": float(support.max().item()),
        "source_count": len(source_totals),
        "largest_source_share": max(source_shares.values(), default=0.0),
        "largest_sources": sorted(
            source_shares.items(), key=lambda item: (-item[1], item[0])
        )[:10],
    }


class _IsolatedAdapterExport(nn.Module):
    def __init__(self, adapter: SupportConstrainedCapsuleAdapter) -> None:
        super().__init__()
        self.adapter = copy.deepcopy(adapter).cpu().eval()

    def forward(
        self,
        patches: Tensor,
        patch_indices: Tensor,
        token_valid: Tensor,
        raw_logits: Tensor,
    ) -> Tensor:
        return self.adapter.forward_sparse(
            patches, patch_indices, token_valid, raw_logits
        )["logits"]


class _FullCandidateExport(nn.Module):
    def __init__(
        self, keeper: nn.Module, adapter: SupportConstrainedCapsuleAdapter
    ) -> None:
        super().__init__()
        self.keeper = copy.deepcopy(keeper).cpu().eval()
        self.adapter = copy.deepcopy(adapter).cpu().eval()

    def forward(self, images: Tensor, bbox: Tensor, image_mask: Tensor) -> Tensor:
        features = self.keeper.forward_features(
            images,
            bbox_token_prior=bbox,
            image_valid_mask=image_mask,
        )
        features["bbox"] = bbox
        raw_logits = classification_logits_from_features(self.keeper, features)
        patches, patch_indices, token_valid = _feature_tensors(features)
        return self.adapter.forward_sparse(
            patches, patch_indices, token_valid, raw_logits
        )["logits"]


def _export_diagnostics(
    *,
    keeper: nn.Module,
    candidate: SupportConstrainedCapsuleAdapter,
    cache: Mapping[str, Tensor],
    resource_batch: tuple[Tensor, Tensor, Mapping[str, object]],
    output_dir: Path,
) -> Dict[str, object]:
    isolated_path = output_dir / "support_constrained_capsule_adapter.onnx"
    try:
        isolated = _onnx_compare(
            wrapper=_IsolatedAdapterExport(candidate),
            inputs=(
                cache["patches"][:1],
                cache["patch_indices"][:1],
                cache["token_valid"][:1],
                cache["raw_logits"][:1],
            ),
            input_names=("patches", "patch_indices", "token_valid", "raw_logits"),
            path=isolated_path,
        )
    except Exception as error:
        isolated = _failed_export(isolated_path, error)
    images, _targets, metadata = resource_batch
    images = images[:STATIC_EXPORT_BATCH_SIZE].float().cpu()
    bbox = metadata.get("bbox")
    if not torch.is_tensor(bbox):
        bbox = torch.zeros(STATIC_EXPORT_BATCH_SIZE, 4)
    bbox = bbox[:STATIC_EXPORT_BATCH_SIZE].float().cpu()
    image_mask = metadata.get("image_mask")
    if not torch.is_tensor(image_mask):
        image_mask = torch.ones(
            STATIC_EXPORT_BATCH_SIZE,
            int(images.size(-2)),
            int(images.size(-1)),
            dtype=torch.bool,
        )
    image_mask = image_mask[:STATIC_EXPORT_BATCH_SIZE].bool().cpu()
    full_path = output_dir / "support_constrained_capsule_candidate.onnx"
    try:
        full = _onnx_compare(
            wrapper=_FullCandidateExport(keeper, candidate),
            inputs=(images, bbox, image_mask),
            input_names=("images", "bbox", "image_mask"),
            path=full_path,
        )
    except Exception as error:
        full = _failed_export(full_path, error)
    return {
        "isolated": isolated,
        "full": full,
        "routing_iterations": ROUTING_ITERATIONS,
        "python_fallback": False,
    }


def _benchmark_inference(
    *,
    keeper: nn.Module,
    adapter: Optional[SupportConstrainedCapsuleAdapter],
    images_cpu: Tensor,
    metadata_cpu: Mapping[str, object],
    device: torch.device,
    amp_dtype: torch.dtype,
    repeats: int,
) -> Dict[str, object]:
    gc.collect()
    torch.cuda.empty_cache()
    keeper = keeper.to(device).eval()
    adapter_device = (
        copy.deepcopy(adapter).to(device).eval() if adapter is not None else None
    )
    count = min(NATURAL_BATCH_SIZE, int(images_cpu.size(0)))
    if count != NATURAL_BATCH_SIZE:
        raise ValueError("CapsALM benchmark requires a complete batch of 32.")
    images = images_cpu[:count].to(device)
    metadata = _metadata_to_device(metadata_cpu, device=device, count=count)

    def iteration() -> Tensor:
        with torch.inference_mode(), torch.autocast(
            device_type="cuda", dtype=amp_dtype, enabled=True
        ):
            raw_logits, features = _forward_classification_with_metadata(
                keeper, images, metadata, device=device
            )
        if adapter_device is None:
            return raw_logits.float()
        if not isinstance(features, Mapping):
            raise ValueError("Benchmark keeper forward returned no features.")
        patches, patch_indices, token_valid = _feature_tensors(features)
        return adapter_device.forward_sparse(
            patches.float(), patch_indices, token_valid, raw_logits.float()
        )["logits"]

    for _ in range(3):
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
        "logits_finite": bool(torch.isfinite(logits).all()),
        "batch_size": count,
    }
    del adapter_device, images, logits
    gc.collect()
    torch.cuda.empty_cache()
    return result


def _dense_patch_map(
    values: Tensor, patch_indices: Tensor, token_valid: Tensor
) -> Tensor:
    if values.ndim != 1 or patch_indices.ndim != 1 or token_valid.ndim != 1:
        raise ValueError("Dense patch map inputs must be one-dimensional.")
    if not (values.shape == patch_indices.shape == token_valid.shape):
        raise ValueError("Dense patch map inputs must align.")
    output = values.new_zeros(GRID_HEIGHT * GRID_WIDTH)
    valid = token_valid.to(device=values.device, dtype=torch.bool)
    output.scatter_(
        0,
        patch_indices.to(device=values.device, dtype=torch.long),
        values * valid.to(dtype=values.dtype),
    )
    return output.reshape(GRID_HEIGHT, GRID_WIDTH)


def _bbox_region_maps(
    bbox_prior: Tensor, patch_indices: Tensor, token_valid: Tensor
) -> Dict[str, Tensor]:
    valid = _dense_patch_map(
        token_valid.to(dtype=torch.float32), patch_indices, token_valid
    ) > 0.5
    dense_prior = _dense_patch_map(
        bbox_prior.float().clamp(0.0, 1.0), patch_indices, token_valid
    )
    foreground = (dense_prior > 0.05) & valid
    if not bool(foreground.any().item()):
        foreground = valid.clone()
    foreground_float = foreground.float().reshape(1, 1, GRID_HEIGHT, GRID_WIDTH)
    eroded = 1.0 - F.max_pool2d(
        1.0 - foreground_float, kernel_size=3, stride=1, padding=1
    )
    interior = (eroded[0, 0] > 0.5) & foreground
    border = foreground & ~interior
    background = valid & ~foreground
    return {
        "valid": valid,
        "bbox_prior": dense_prior,
        "foreground": foreground,
        "border": border,
        "background": background,
    }


def _spatial_mass(values: Tensor, masks: Mapping[str, Tensor]) -> Dict[str, float]:
    mass = values.float().clamp_min(0.0)
    mass = mass / mass.sum().clamp_min(1e-12)
    return {
        f"{name}_mass": float(mass[mask].sum().item())
        for name, mask in masks.items()
        if name in {"foreground", "border", "background"}
    }


def _signed_normalize_map(value: Tensor) -> np.ndarray:
    maximum = float(value.detach().abs().amax().item())
    if maximum <= 0.0:
        return np.full(tuple(value.shape), 0.5, dtype=np.float32)
    return (
        0.5 + 0.5 * value.detach().float().cpu().numpy() / maximum
    ).clip(0.0, 1.0).astype(np.float32)


def _deterministic_occlusion_saliency(
    *,
    keeper: nn.Module,
    candidate: SupportConstrainedCapsuleAdapter,
    image: Tensor,
    metadata: Mapping[str, object],
    base_margin: Tensor,
    device: torch.device,
    batch_size: int,
    grid: int = 4,
) -> Tensor:
    if image.shape[0] != 1 or image.ndim != 4:
        raise ValueError("Occlusion saliency requires one BCHW input.")
    height, width = int(image.size(-2)), int(image.size(-1))
    occluded = image.detach().repeat(int(grid * grid), 1, 1, 1)
    for row in range(int(grid)):
        y0 = row * height // int(grid)
        y1 = (row + 1) * height // int(grid)
        for column in range(int(grid)):
            x0 = column * width // int(grid)
            x1 = (column + 1) * width // int(grid)
            occluded[row * int(grid) + column, :, y0:y1, x0:x1] = 0.0
    margins = []
    with torch.inference_mode():
        for start in range(0, int(occluded.size(0)), max(1, int(batch_size))):
            stop = min(start + max(1, int(batch_size)), int(occluded.size(0)))
            metadata_batch = _repeat_metadata(metadata, stop - start)
            raw_logits, features = _forward_classification_with_metadata(
                keeper, occluded[start:stop], metadata_batch, device=device
            )
            if not isinstance(features, Mapping):
                raise ValueError("Occlusion keeper forward returned no features.")
            patches, patch_indices, token_valid = _feature_tensors(features)
            result = candidate.forward_sparse(
                patches.float(), patch_indices, token_valid, raw_logits.float()
            )
            margins.append(class1_margin(result["logits"]))
    coarse = (
        base_margin.detach().reshape(1) - torch.cat(margins)
    ).abs().reshape(1, 1, int(grid), int(grid))
    return F.interpolate(
        coarse,
        size=(GRID_HEIGHT, GRID_WIDTH),
        mode="bilinear",
        align_corners=False,
    )[0, 0]


def _render_xai_pages(
    records: Sequence[Mapping[str, object]], output_dir: Path
) -> list[str]:
    columns = (
        "image",
        "valid",
        "bbox",
        "control_c1",
        "candidate_c1",
        "candidate_minus_control",
        "dynamic_minus_uniform",
        "occlusion",
    )
    panel_width = 150
    row_height = 190
    pages = []
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
            maps = record["display_maps"]
            for column, name in enumerate(columns):
                panel = (
                    Image.fromarray(rgb)
                    if name == "image"
                    else _heat_overlay(rgb, np.asarray(maps[name]), alpha=0.55)
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
                    f"idx={record['sample_index']} y={record['target']} "
                    f"pred raw/ctl/cand={record['raw_prediction']}/"
                    f"{record['control_prediction']}/{record['candidate_prediction']}"
                ),
                fill="black",
            )
        path = output_dir / f"capsalm_xai_page_{page_index:03d}.png"
        canvas.save(path)
        pages.append(str(path.resolve()))
    return pages


def _xai_audit(
    *,
    keeper: nn.Module,
    control: SupportConstrainedCapsuleAdapter,
    candidate: SupportConstrainedCapsuleAdapter,
    dataset: MangoYOLOCropDataset,
    transform,
    raw_clean: Sequence[Mapping[str, object]],
    control_clean: Sequence[Mapping[str, object]],
    candidate_clean: Sequence[Mapping[str, object]],
    mean: Sequence[float],
    std: Sequence[float],
    args: argparse.Namespace,
    device: torch.device,
    output_dir: Path,
) -> Dict[str, object]:
    selected, categories, required = _required_xai_indices(
        raw_clean, control_clean, candidate_clean, minimum_rows=12
    )
    row_lookup = {
        name: {int(row["sample_index"]): row for row in values}
        for name, values in (
            ("raw", raw_clean),
            ("control", control_clean),
            ("candidate", candidate_clean),
        )
    }
    keeper = keeper.to(device).eval()
    control_device = copy.deepcopy(control).to(device).eval()
    candidate_device = copy.deepcopy(candidate).to(device).eval()
    for module in (control_device, candidate_device):
        for parameter in module.parameters():
            parameter.requires_grad_(False)
    records = []
    arrays: Dict[str, np.ndarray] = {}
    all_maps_finite = True
    all_defined_nonzero = True
    coupling_pruned_mass = 0.0
    mass_records = []
    for position, sample_index in enumerate(selected):
        loader, _summary = _make_loader(
            base_dataset=dataset,
            transform=transform,
            indices=[sample_index],
            batch_size=1,
            num_workers=0,
            context=f"capsalm_xai_{sample_index}",
            seed=SEED + 500 + position,
        )
        images_cpu, targets_cpu, metadata_cpu = next(iter(loader))
        images = images_cpu.to(device)
        with torch.inference_mode():
            raw_logits, features = _forward_classification_with_metadata(
                keeper, images, metadata_cpu, device=device
            )
        if not isinstance(features, Mapping):
            raise ValueError("CapsALM XAI keeper forward returned no features.")
        patches, patch_indices, token_valid = _feature_tensors(features)
        bbox_prior = features.get("patch_bbox_prior")
        if not torch.is_tensor(bbox_prior) or bbox_prior.shape != token_valid.shape:
            raise ValueError("CapsALM XAI requires aligned patch_bbox_prior.")
        control_result = control_device.forward_sparse(
            patches.float(), patch_indices, token_valid, raw_logits.float()
        )
        candidate_result = candidate_device.forward_sparse(
            patches.float(), patch_indices, token_valid, raw_logits.float()
        )
        uniform_result = candidate_device.forward_sparse(
            patches.float(),
            patch_indices,
            token_valid,
            raw_logits.float(),
            routing_iterations=1,
        )
        control_spatial = _routing_statistics(control_result, token_valid)[
            "class1_spatial"
        ][0]
        candidate_spatial = _routing_statistics(candidate_result, token_valid)[
            "class1_spatial"
        ][0]
        uniform_spatial = _routing_statistics(uniform_result, token_valid)[
            "class1_spatial"
        ][0]
        control_dense = _dense_patch_map(
            control_spatial, patch_indices[0], token_valid[0]
        )
        candidate_dense = _dense_patch_map(
            candidate_spatial, patch_indices[0], token_valid[0]
        )
        uniform_dense = _dense_patch_map(
            uniform_spatial, patch_indices[0], token_valid[0]
        )
        regions = _bbox_region_maps(
            bbox_prior[0], patch_indices[0], token_valid[0]
        )
        base_margin = class1_margin(candidate_result["logits"])
        occlusion = _deterministic_occlusion_saliency(
            keeper=keeper,
            candidate=candidate_device,
            image=images,
            metadata=metadata_cpu,
            base_margin=base_margin,
            device=device,
            batch_size=int(args.xai_batch_size),
        )
        difference = candidate_dense - control_dense
        dynamic_difference = candidate_dense - uniform_dense
        raw_maps = {
            "valid": regions["valid"].float(),
            "bbox": regions["bbox_prior"],
            "control_c1": control_dense,
            "candidate_c1": candidate_dense,
            "candidate_minus_control": difference,
            "dynamic_minus_uniform": dynamic_difference,
            "occlusion": occlusion,
        }
        display_maps = {
            "valid": _normalize_map(raw_maps["valid"]),
            "bbox": _normalize_map(raw_maps["bbox"]),
            "control_c1": _normalize_map(raw_maps["control_c1"]),
            "candidate_c1": _normalize_map(raw_maps["candidate_c1"]),
            "candidate_minus_control": _signed_normalize_map(difference),
            "dynamic_minus_uniform": _signed_normalize_map(dynamic_difference),
            "occlusion": _normalize_map(occlusion),
        }
        finite = all(bool(torch.isfinite(value).all()) for value in raw_maps.values())
        nonzero = all(
            bool(torch.count_nonzero(value).item())
            for key, value in raw_maps.items()
            if key not in {"candidate_minus_control", "dynamic_minus_uniform"}
        ) and bool(torch.count_nonzero(difference).item()) and bool(
            torch.count_nonzero(dynamic_difference).item()
        )
        all_maps_finite = all_maps_finite and finite
        all_defined_nonzero = all_defined_nonzero and nonzero
        invalid = ~regions["valid"]
        coupling_pruned_mass += float(control_dense[invalid].abs().sum().item())
        coupling_pruned_mass += float(candidate_dense[invalid].abs().sum().item())
        coupling_pruned_mass += float(uniform_dense[invalid].abs().sum().item())
        control_mass = _spatial_mass(control_dense, regions)
        candidate_mass = _spatial_mass(candidate_dense, regions)
        mass_records.append(
            {
                "sample_index": int(sample_index),
                "control": control_mass,
                "candidate": candidate_mass,
            }
        )
        rgb = _rgb_from_tensor(images_cpu[0], mean=mean, std=std)
        raw_row = row_lookup["raw"][sample_index]
        control_row = row_lookup["control"][sample_index]
        candidate_row = row_lookup["candidate"][sample_index]
        probabilities = {
            name: [float(row_lookup[name][sample_index][f"prob_{index}"]) for index in range(NUM_CLASSES)]
            for name in ("raw", "control", "candidate")
        }
        records.append(
            {
                "sample_index": int(sample_index),
                "target": int(targets_cpu[0].item()),
                "categories": list(categories.get(sample_index, ())),
                "raw_prediction": int(raw_row["prediction"]),
                "control_prediction": int(control_row["prediction"]),
                "candidate_prediction": int(candidate_row["prediction"]),
                "probabilities": probabilities,
                "control_mass": control_mass,
                "candidate_mass": candidate_mass,
                "rgb": rgb,
                "display_maps": display_maps,
            }
        )
        arrays[f"sample_{sample_index}_rgb"] = rgb
        for key, value in raw_maps.items():
            arrays[f"sample_{sample_index}_{key}"] = (
                value.detach().float().cpu().numpy().astype(np.float32)
            )
        for key in ("foreground", "border", "background"):
            arrays[f"sample_{sample_index}_{key}_mask"] = (
                regions[key].detach().cpu().numpy().astype(np.uint8)
            )
    tensor_path = output_dir / "support_constrained_capsule_xai_tensors.npz"
    np.savez_compressed(tensor_path, **arrays)
    pages = _render_xai_pages(records, output_dir)
    serializable_records = [
        {key: value for key, value in record.items() if key not in {"rgb", "display_maps"}}
        for record in records
    ]
    manifest = {
        "selected_indices": selected,
        "required_indices": required,
        "categories": {str(key): list(value) for key, value in categories.items()},
        "required_coverage_exact": set(required).issubset(set(selected)),
        "minimum_twelve_when_available": len(selected) >= min(12, len(raw_clean)),
        "all_maps_finite": all_maps_finite,
        "all_defined_maps_nonzero": all_defined_nonzero,
        "coupling_pruned_invalid_mass": coupling_pruned_mass,
        "coupling_pruned_invalid_mass_exact_zero": coupling_pruned_mass == 0.0,
        "selection_replay_exact": _required_xai_indices(
            raw_clean, control_clean, candidate_clean, minimum_rows=12
        )[0]
        == selected,
        "input_saliency_method": (
            "deterministic_4x4_zero_occlusion_absolute_deployed_margin_drop"
        ),
        "bbox_region_contract": (
            "patch_bbox_prior_gt_0p05; one_patch_erosion; retained_valid_background"
        ),
        "mass_records": mass_records,
        "tensor_path": str(tensor_path.resolve()),
        "tensor_sha256": _sha256(tensor_path),
        "pages": pages,
        "records": serializable_records,
    }
    manifest_path = output_dir / "support_constrained_capsule_xai_manifest.json"
    manifest_path.write_text(
        json.dumps(manifest, indent=2, sort_keys=True, ensure_ascii=True) + "\n",
        encoding="utf-8",
    )
    del control_device, candidate_device
    gc.collect()
    torch.cuda.empty_cache()
    return {
        **manifest,
        "manifest": str(manifest_path.resolve()),
        "manifest_sha256": _sha256(manifest_path),
    }


def assess_stage_a(
    *,
    structural_checks: Mapping[str, bool],
    comparisons: Mapping[str, Mapping[str, Mapping[str, object]]],
    mechanisms: Mapping[str, Mapping[str, Mapping[str, object]]],
    mcl_comparisons: Mapping[str, Mapping[str, Mapping[str, object]]],
    dual: Mapping[str, object],
) -> Dict[str, object]:
    clean = comparisons["clean"]
    raw = clean["raw_candidate"]
    control = clean["control_candidate"]
    raw_delta = raw["delta"]
    control_delta = control["delta"]
    raw_transition = raw["transitions"]
    control_transition = control["transitions"]
    decision_checks = {
        "macro_f1_delta_vs_raw_gte_minus_0p002": float(raw_delta["macro_f1"])
        >= -0.002,
        "class1_f1_gain_vs_raw_gte_0p005": float(raw_delta["class1_f1"])
        >= 0.005,
        "class1_precision_gain_vs_raw_gte_0p010": float(
            raw_delta["class1_precision"]
        )
        >= 0.010,
        "class1_recall_delta_vs_raw_gte_minus_0p005": float(
            raw_delta["class1_recall"]
        )
        >= -0.005,
        "class1_tp_break_vs_raw_lte_1": int(raw_transition["focus_tp_break"]) <= 1,
        "class1_rescues_gte_breaks_vs_raw": int(raw_transition["focus_fn_rescue"])
        >= int(raw_transition["focus_tp_break"]),
        "restricted_fp_reduction_vs_raw_gte_3": int(
            raw_transition["restricted_focus_fp_reduction"]
        )
        >= 3,
        "corrections_gt_harms_vs_raw": int(raw_transition["candidate_correction"])
        > int(raw_transition["candidate_harm"]),
        "maximum_nonfocus_f1_drop_vs_raw_lte_0p010": float(
            raw["maximum_nonfocus_f1_drop"]
        )
        <= 0.010,
        "macro_f1_delta_vs_control_nonnegative": float(control_delta["macro_f1"])
        >= 0.0,
        "class1_f1_gain_vs_control_gte_0p003": float(control_delta["class1_f1"])
        >= 0.003,
        "class1_precision_gain_vs_control_gte_0p003": float(
            control_delta["class1_precision"]
        )
        >= 0.003,
        "class1_recall_delta_vs_control_gte_minus_0p005": float(
            control_delta["class1_recall"]
        )
        >= -0.005,
        "restricted_fp_reduction_vs_control_gte_1": int(
            control_transition["restricted_focus_fp_reduction"]
        )
        >= 1,
        "corrections_gt_harms_vs_control": int(
            control_transition["candidate_correction"]
        )
        > int(control_transition["candidate_harm"]),
    }
    shifted = [comparisons[name] for name, _, _ in LIGHTING_CONDITIONS]
    per_condition_illumination: Dict[str, bool] = {}
    for condition, _brightness, _contrast in LIGHTING_CONDITIONS:
        item = comparisons[condition]["raw_candidate"]
        delta = item["delta"]
        transition = item["transitions"]
        per_condition_illumination[f"{condition}_class1_f1_nonnegative"] = float(
            delta["class1_f1"]
        ) >= 0.0
        per_condition_illumination[
            f"{condition}_class1_precision_nonnegative"
        ] = float(delta["class1_precision"]) >= 0.0
        per_condition_illumination[
            f"{condition}_class1_recall_gte_minus_0p005"
        ] = float(delta["class1_recall"]) >= -0.005
        per_condition_illumination[f"{condition}_class1_tp_break_lte_1"] = int(
            transition["focus_tp_break"]
        ) <= 1
        per_condition_illumination[
            f"{condition}_class1_rescues_gte_breaks"
        ] = int(transition["focus_fn_rescue"]) >= int(transition["focus_tp_break"])
        per_condition_illumination[
            f"{condition}_restricted_fp_nonincrease"
        ] = int(transition["restricted_focus_fp_reduction"]) >= 0
        per_condition_illumination[
            f"{condition}_maximum_nonfocus_drop_lte_0p010"
        ] = float(item["maximum_nonfocus_f1_drop"]) <= 0.010
    shifted_f1 = [float(item["raw_candidate"]["delta"]["class1_f1"]) for item in shifted]
    shifted_precision = [
        float(item["raw_candidate"]["delta"]["class1_precision"]) for item in shifted
    ]
    shifted_removals = [
        int(item["raw_candidate"]["transitions"]["restricted_focus_fp_reduction"])
        for item in shifted
    ]
    illumination_checks = {
        **per_condition_illumination,
        "mean_shifted_class1_f1_gain_gte_0p003": statistics.mean(shifted_f1)
        >= 0.003,
        "mean_shifted_class1_precision_gain_gte_0p005": statistics.mean(
            shifted_precision
        )
        >= 0.005,
        "restricted_fp_removed_in_at_least_2_shifted_conditions": sum(
            value >= 1 for value in shifted_removals
        )
        >= 2,
    }
    mcl_checks: Dict[str, bool] = {}
    for condition in CONDITIONS:
        candidate_metrics = comparisons[condition]["raw_candidate"]["candidate"]
        mcl_metrics = mcl_comparisons[condition]["raw_candidate"]["candidate"]
        for metric, label in (
            ("per_class_f1", "f1"),
            ("per_class_precision", "precision"),
            ("per_class_recall", "recall"),
        ):
            mcl_checks[f"{condition}_class1_{label}_strictly_exceeds_mcl"] = float(
                candidate_metrics[metric][FOCUS_CLASS]
            ) > float(mcl_metrics[metric][FOCUS_CLASS])
    mechanism_checks: Dict[str, bool] = {}
    for condition in CONDITIONS:
        control_mechanism = mechanisms[condition]["control"]
        candidate_mechanism = mechanisms[condition]["candidate"]
        candidate_auc = candidate_mechanism[
            "class1_capsule_residual_margin_auroc_vs_0_2_4"
        ]
        control_auc = control_mechanism[
            "class1_capsule_residual_margin_auroc_vs_0_2_4"
        ]
        mechanism_checks[f"{condition}_rank_violation_lte_0p90x_control"] = float(
            candidate_mechanism["rank_violation_mean"]
        ) <= 0.90 * float(control_mechanism["rank_violation_mean"])
        mechanism_checks[f"{condition}_support_mean_lte_control"] = float(
            candidate_mechanism["support_violation_mean"]
        ) <= float(control_mechanism["support_violation_mean"])
        mechanism_checks[f"{condition}_support_p95_lte_control"] = float(
            candidate_mechanism["support_violation_p95"]
        ) <= float(control_mechanism["support_violation_p95"])
        mechanism_checks[f"{condition}_candidate_auroc_gte_0p70"] = (
            candidate_auc is not None and float(candidate_auc) >= 0.70
        )
        mechanism_checks[f"{condition}_candidate_auroc_gte_control"] = (
            candidate_auc is not None
            and control_auc is not None
            and float(candidate_auc) >= float(control_auc)
        )
        entropy = float(
            candidate_mechanism["normalized_class1_routing_entropy_mean"]
        )
        effective = float(
            candidate_mechanism["effective_class1_routed_patch_count_mean"]
        )
        mechanism_checks[f"{condition}_routing_entropy_in_range"] = (
            0.20 <= entropy <= 0.995
        )
        mechanism_checks[f"{condition}_effective_patch_count_in_range"] = (
            4.0 <= effective <= 160.0
        )
        mechanism_checks[f"{condition}_coupling_contract_exact"] = bool(
            float(candidate_mechanism["coupling_parent_sum_maximum_error"]) <= 1e-6
            and float(candidate_mechanism["invalid_coupling_mass"]) == 0.0
        )
        errors = candidate_mechanism["dynamic_uniform_maximum_errors"]
        mechanism_checks[f"{condition}_dynamic_differs_on_all_audited_paths"] = all(
            float(value) > 0.0 for value in errors.values()
        )
        mechanism_checks[f"{condition}_dynamic_changes_holdout_logit"] = int(
            candidate_mechanism["dynamic_uniform_logit_changed_rows"]
        ) > 0
    mechanism_checks.update(
        {
            "dual_finite_nonnegative": bool(dual["finite_nonnegative"]),
            "rank_multiplier_positive": int(dual["rank_nonzero"]) > 0,
            "rank_multiplier_nonzero_spread": float(dual["rank_std"]) > 0.0,
            "largest_dual_source_share_lte_0p02": float(
                dual["largest_source_share"]
            )
            <= 0.02,
        }
    )
    all_checks = {
        **{str(key): bool(value) for key, value in structural_checks.items()},
        **decision_checks,
        **illumination_checks,
        **mcl_checks,
        **mechanism_checks,
    }
    failed = [key for key, passed in all_checks.items() if not passed]
    clean_behavior_passed = all(decision_checks.values())
    return {
        "structural_checks": dict(structural_checks),
        "decision_checks": decision_checks,
        "illumination_checks": illumination_checks,
        "mcl_comparator_checks": mcl_checks,
        "mechanism_checks": mechanism_checks,
        "clean_behavior_passed": clean_behavior_passed,
        "constraint_reduction_claim_authorized": bool(
            clean_behavior_passed and all(mechanism_checks.values())
        ),
        "failed_checks": failed,
        "all_gates_passed": not failed,
        "stage_b_authorized": not failed,
        "validation_authorized": not failed,
        "test_authorized": False,
        "full_train_authorized": False,
    }


def _training_serializable(training: Mapping[str, object]) -> Dict[str, object]:
    return {key: value for key, value in training.items() if key != "dual_tensors"}


def _write_training_curve(
    path: Path,
    control: Mapping[str, object],
    candidate: Mapping[str, object],
) -> None:
    rows = [*control["history"], *candidate["history"]]
    fields = (
        "variant",
        "epoch",
        "natural_rows",
        "natural_batches",
        "boundary_events",
        "boundary_positive_rows",
        "loss",
        "ce",
        "rank_penalty",
        "rank_dual",
        "support_penalty",
        "support_dual",
        "rank_violation",
        "support_violation",
        "lambda_rank_mean",
        "lambda_rank_max",
        "lambda_support_mean",
        "lambda_support_max",
    )
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields)
        writer.writeheader()
        writer.writerows(rows)


def _write_mechanism_csv(
    path: Path, mechanisms: Mapping[str, Mapping[str, Mapping[str, object]]]
) -> None:
    fields = (
        "condition",
        "variant",
        "rows",
        "positive_rows",
        "hard_negative_rows",
        "rank_violation_mean",
        "rank_violation_p95",
        "support_violation_mean",
        "support_violation_p95",
        "class1_capsule_residual_margin_auroc_vs_0_2_4",
        "normalized_class1_routing_entropy_mean",
        "effective_class1_routed_patch_count_mean",
        "coupling_parent_sum_maximum_error",
        "invalid_coupling_mass",
        "dynamic_uniform_logit_changed_rows",
    )
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields)
        writer.writeheader()
        for condition in CONDITIONS:
            for variant in ("control", "candidate"):
                source = mechanisms[condition][variant]
                writer.writerow(
                    {
                        "condition": condition,
                        "variant": variant,
                        **{key: source.get(key) for key in fields[2:]},
                    }
                )


def _write_dual_csv(
    path: Path,
    *,
    candidate_training: Mapping[str, object],
    reference_indices: Sequence[int],
    rows: Sequence[CleanTrainRow],
) -> None:
    dual = candidate_training["dual_tensors"]
    with path.open("w", encoding="utf-8", newline="") as handle:
        fields = (
            "condition",
            "sample_index",
            "source_stem",
            "rank_multiplier",
            "support_multiplier",
        )
        writer = csv.DictWriter(handle, fieldnames=fields)
        writer.writeheader()
        for condition_index, condition in enumerate(CONDITIONS):
            for position, sample_index in enumerate(reference_indices):
                writer.writerow(
                    {
                        "condition": condition,
                        "sample_index": int(sample_index),
                        "source_stem": rows[int(sample_index)].source_stem,
                        "rank_multiplier": float(
                            dual["rank"][condition_index, position].item()
                        ),
                        "support_multiplier": float(
                            dual["support"][condition_index, position].item()
                        ),
                    }
                )


def _write_report(path: Path, summary: Mapping[str, object]) -> None:
    raw = summary["comparisons"]["clean"]["raw_candidate"]
    control = summary["comparisons"]["clean"]["control_candidate"]
    mechanism = summary["mechanism"]["holdout"]["clean"]
    gate = summary["gate"]
    lines = [
        "# Support-Constrained Capsule ALM A0 Result",
        "",
        f"- Status: `{summary['status']}`",
        f"- Stage B authorized: `{gate['stage_b_authorized']}`",
        f"- Failed checks: `{', '.join(gate['failed_checks']) or 'none'}`",
        (
            "- Clean macro/class1 F1 delta versus raw: "
            f"`{raw['delta']['macro_f1']:+.6f}/"
            f"{raw['delta']['class1_f1']:+.6f}`"
        ),
        (
            "- Clean class1 precision/recall delta versus raw: "
            f"`{raw['delta']['class1_precision']:+.6f}/"
            f"{raw['delta']['class1_recall']:+.6f}`"
        ),
        (
            "- Clean class1 F1/precision delta versus CE control: "
            f"`{control['delta']['class1_f1']:+.6f}/"
            f"{control['delta']['class1_precision']:+.6f}`"
        ),
        (
            "- Restricted FP reduction / class1 TP breaks versus raw: "
            f"`{raw['transitions']['restricted_focus_fp_reduction']}/"
            f"{raw['transitions']['focus_tp_break']}`"
        ),
        (
            "- Clean rank violation control/candidate: "
            f"`{mechanism['control']['rank_violation_mean']:.6f}/"
            f"{mechanism['candidate']['rank_violation_mean']:.6f}`"
        ),
        (
            "- Runtime ratio / full peak VRAM GiB: "
            f"`{summary['resources']['runtime_ratio']:.6f}/"
            f"{summary['resources']['candidate']['peak_vram_gib']:.6f}`"
        ),
        f"- XAI rows/pages: `{len(summary['xai']['selected_indices'])}/{len(summary['xai']['pages'])}`",
        "",
        "Only source-disjoint yolo_f/train rows were used. Validation and test were not accessed.",
        "Current-best commands remain unchanged.",
    ]
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def _write_manifest(output_dir: Path) -> Dict[str, object]:
    artifacts = []
    for path in sorted(output_dir.rglob("*")):
        if path.is_file() and path.name != "artifact_manifest.json":
            artifacts.append(
                {
                    "path": str(path.relative_to(output_dir)).replace("\\", "/"),
                    "bytes": path.stat().st_size,
                    "sha256": _sha256(path),
                }
            )
    payload = {
        "artifacts": artifacts,
        "artifact_count": len(artifacts),
        "total_bytes": sum(int(row["bytes"]) for row in artifacts),
        "forbidden_model_binary_count": sum(
            Path(str(row["path"])).suffix.lower()
            in {".pt", ".pth", ".ckpt", ".engine"}
            for row in artifacts
        ),
    }
    path = output_dir / "artifact_manifest.json"
    path.write_text(
        json.dumps(payload, indent=2, sort_keys=True, ensure_ascii=True) + "\n",
        encoding="utf-8",
    )
    return {**payload, "path": str(path.resolve()), "sha256": _sha256(path)}


def run_audit(args: argparse.Namespace) -> Dict[str, object]:
    (
        provenance,
        rows,
        cohorts,
        natural_order,
        boundary_schedule,
    ) = _load_locked_inputs(args)
    if bool(args.preflight_only):
        return {
            "status": "preflight_passed",
            "method": METHOD,
            "provenance": provenance,
            "cohort": _cohort_serializable(cohorts),
            "output_directory_created": False,
            "validation_predictions_used": False,
            "test_data_used": False,
        }
    if not torch.cuda.is_available():
        raise RuntimeError("Locked CapsALM A0 requires CUDA.")
    paths = _source_paths(args)
    output_path = Path(args.output_dir).resolve()
    raw_root = paths["data"].parent.parent.resolve()
    try:
        output_path.relative_to(raw_root)
    except ValueError:
        pass
    else:
        raise ValueError("CapsALM output cannot be written under the raw dataset tree.")
    output_dir = _prepare_output_dir(output_path)

    set_seed(SEED, deterministic=True)
    torch.set_float32_matmul_precision("highest")
    torch.backends.cuda.matmul.allow_tf32 = False
    torch.backends.cudnn.allow_tf32 = False
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False
    torch.use_deterministic_algorithms(True)
    device = torch.device("cuda")
    amp_dtype = _amp_dtype(device)

    checkpoint = load_checkpoint(paths["checkpoint"], map_location="cpu")
    keeper, control_prototype, candidate_prototype, construction = (
        _construct_keeper_and_adapters(checkpoint)
    )
    frozen_keeper_before = _state_sha256(keeper)
    dataset, transform, dataset_summary = _build_dataset(
        checkpoint, rows, paths["data"]
    )
    natural_cache, natural_cache_summary, resource_batch = _extract_fit_cache(
        keeper=keeper,
        dataset=dataset,
        transform=transform,
        indices=cohorts["fit_indices"],
        args=args,
        device=device,
        amp_dtype=amp_dtype,
    )
    boundary_indices = [
        *[int(value) for value in cohorts["reference_indices"]],
        *[int(value) for value in cohorts["hard_indices"]],
    ]
    boundary_caches: Dict[str, Dict[str, Tensor]] = {
        "clean": _subset_cache(natural_cache, boundary_indices)
    }
    boundary_cache_summary: Dict[str, object] = {
        "clean": {
            "condition": "clean",
            "rows": len(boundary_indices),
            "sample_index_sha256": _ordered_index_sha256(boundary_indices),
            "source": "exact_subset_of_natural_fit_cache",
            "all_finite": True,
        }
    }
    for condition, _brightness, _contrast in LIGHTING_CONDITIONS:
        cache, cache_summary = _extract_condition_cache(
            keeper=keeper,
            dataset=dataset,
            transform=transform,
            indices=boundary_indices,
            condition=condition,
            args=args,
            device=device,
            amp_dtype=amp_dtype,
        )
        boundary_caches[condition] = cache
        boundary_cache_summary[condition] = cache_summary

    first = _batch_from_cache(
        natural_cache,
        _cache_positions(natural_cache),
        natural_order[:NATURAL_BATCH_SIZE],
        device=device,
    )
    with torch.inference_mode():
        initial_result = candidate_prototype.to(device).forward_sparse(
            first["patches"],
            first["patch_indices"],
            first["token_valid"],
            first["raw_logits"],
        )
    initial_raw_error = float(
        (initial_result["logits"] - first["raw_logits"]).abs().amax().item()
    )
    candidate_prototype = candidate_prototype.cpu()
    equation = _equation_diagnostics()
    initial_gradient = _initial_gradient_audit(
        prototype=candidate_prototype,
        natural_cache=natural_cache,
        boundary_caches=boundary_caches,
        natural_order=natural_order,
        boundary_schedule=boundary_schedule,
        reference_indices=cohorts["reference_indices"],
        device=device,
    )
    control, control_training = _train_variant(
        name="control",
        prototype=control_prototype,
        natural_cache=natural_cache,
        boundary_caches=boundary_caches,
        natural_order=natural_order,
        boundary_schedule=boundary_schedule,
        reference_indices=cohorts["reference_indices"],
        candidate=False,
        args=args,
        device=device,
    )
    candidate, candidate_training = _train_variant(
        name="candidate",
        prototype=candidate_prototype,
        natural_cache=natural_cache,
        boundary_caches=boundary_caches,
        natural_order=natural_order,
        boundary_schedule=boundary_schedule,
        reference_indices=cohorts["reference_indices"],
        candidate=True,
        args=args,
        device=device,
    )
    dual = _dual_diagnostics(
        candidate_training,
        reference_indices=cohorts["reference_indices"],
        rows=rows,
    )
    hard_holdout_indices = [
        int(sample_index)
        for sample_index in cohorts["holdout_indices"]
        if rows[int(sample_index)].target in RESTRICTED_NEGATIVE_CLASSES
        and rows[int(sample_index)].keeper_prediction == FOCUS_CLASS
    ]
    if len(hard_holdout_indices) != 37:
        raise ValueError("CapsALM holdout restricted-hard cohort differs from keeper.")
    (
        raw_conditions,
        control_conditions,
        candidate_conditions,
        mechanisms,
        evaluation_summary,
    ) = _evaluate_conditions(
        keeper=keeper,
        control=control,
        candidate=candidate,
        dataset=dataset,
        transform=transform,
        indices=cohorts["holdout_indices"],
        rows=rows,
        hard_holdout_indices=hard_holdout_indices,
        args=args,
        device=device,
        amp_dtype=amp_dtype,
    )
    comparisons = _build_comparisons(
        raw_conditions=raw_conditions,
        control_conditions=control_conditions,
        candidate_conditions=candidate_conditions,
    )
    (
        mcl_raw_conditions,
        _mcl_candidate_conditions,
        mcl_comparisons,
        mcl_replay,
    ) = _load_mcl_comparator(paths=paths)
    raw_mcl_replay = _compare_raw_to_mcl(raw_conditions, mcl_raw_conditions)

    raw_benchmark = _benchmark_inference(
        keeper=keeper,
        adapter=None,
        images_cpu=resource_batch[0],
        metadata_cpu=resource_batch[2],
        device=device,
        amp_dtype=amp_dtype,
        repeats=int(args.benchmark_repeats),
    )
    candidate_benchmark = _benchmark_inference(
        keeper=keeper,
        adapter=candidate,
        images_cpu=resource_batch[0],
        metadata_cpu=resource_batch[2],
        device=device,
        amp_dtype=amp_dtype,
        repeats=int(args.benchmark_repeats),
    )
    runtime_ratio = float(candidate_benchmark["median_seconds"]) / max(
        float(raw_benchmark["median_seconds"]), 1e-12
    )
    incremental_vram = max(
        0.0,
        float(candidate_benchmark["peak_vram_gib"])
        - float(raw_benchmark["peak_vram_gib"]),
    )
    export = _export_diagnostics(
        keeper=keeper,
        candidate=candidate,
        cache=natural_cache,
        resource_batch=resource_batch,
        output_dir=output_dir,
    )
    mean, std = checkpoint_input_normalization(checkpoint)
    xai = _xai_audit(
        keeper=keeper,
        control=control,
        candidate=candidate,
        dataset=dataset,
        transform=transform,
        raw_clean=raw_conditions["clean"],
        control_clean=control_conditions["clean"],
        candidate_clean=candidate_conditions["clean"],
        mean=mean,
        std=std,
        args=args,
        device=device,
        output_dir=output_dir,
    )
    frozen_keeper_after = _state_sha256(keeper.cpu())
    structural_checks = {
        "locked_arguments_exact": _locked_args_exact(args),
        "official_capsule_source_commit_clean_license_verified": bool(
            provenance["capsule_worktree_clean"]
            and provenance["capsule_license_present"]
            and provenance["capsule_commit"] == LOCKED_CAPSULE_COMMIT
        ),
        "official_alm_source_commit_clean_no_license_disclosed": bool(
            provenance["alm_worktree_clean"]
            and not provenance["alm_license_present"]
            and provenance["alm_commit"] == LOCKED_ALM_COMMIT
            and provenance["alm_github_last_positive_loop_discrepancy_disclosed"]
        ),
        "source_disjoint_cohorts_and_hashes_exact": bool(
            len(cohorts["fit_indices"]) == 7372
            and len(cohorts["holdout_indices"]) == 1843
            and len(cohorts["reference_indices"]) == 432
            and len(cohorts["hard_indices"]) == 186
            and not cohorts["source_overlap"]
            and cohorts["fit_index_sha256"] == LOCKED_FIT_INDEX_SHA256
            and cohorts["holdout_index_sha256"] == LOCKED_HOLDOUT_INDEX_SHA256
            and cohorts["reference_index_sha256"] == LOCKED_REFERENCE_INDEX_SHA256
            and cohorts["hard_index_sha256"] == LOCKED_HARD_INDEX_SHA256
        ),
        "natural_and_boundary_schedule_hashes_exact": bool(
            provenance["natural_order_sha256"] == LOCKED_NATURAL_ORDER_SHA256
            and provenance["boundary_schedule_sha256"]
            == LOCKED_BOUNDARY_SCHEDULE_SHA256
        ),
        "keeper_parameter_count_exact": int(construction["keeper_parameters"])
        == EXPECTED_KEEPER_PARAMETERS,
        "adapter_parameter_schema_count_exact": int(
            construction["adapter_parameters"]
        )
        == EXPECTED_ADAPTER_PARAMETERS,
        "candidate_control_initial_state_optimizer_bit_exact": bool(
            construction["candidate_control_bit_exact"]
            and construction["candidate_control_initial_optimizer_bit_exact"]
        ),
        "isolated_initialization_rng_restored": bool(
            construction["isolated_initialization_rng_restored"]
        ),
        "initial_residual_head_zero_and_raw_equivalence_exact": bool(
            construction["residual_head_weight_zero"]
            and construction["residual_head_bias_zero"]
            and initial_raw_error == 0.0
        ),
        "routing_and_alm_equations_match_independent_reference": bool(
            equation["routing_matches_reference_lte_1e6"]
            and equation["constraints_match_reference_lte_1e7"]
            and equation["coupling_contract_exact"]
        ),
        "control_detached_gradient_matches_pure_ce_bit_exact": bool(
            initial_gradient[
                "control_detached_gradient_matches_pure_ce_bit_exact"
            ]
        ),
        "candidate_rank_gradient_nonzero_and_differs_from_ce": bool(
            initial_gradient["rank_gradient_nonzero"]
            and initial_gradient["candidate_gradient_differs_from_ce"]
        ),
        "initial_support_violation_exact_zero": bool(
            initial_gradient["initial_support_violation_exact_zero"]
        ),
        "both_execute_exact_epochs_occurrences_and_schedules": bool(
            control_training["epochs"] == EPOCHS
            and candidate_training["epochs"] == EPOCHS
            and control_training["natural_rows"] == 221160
            and candidate_training["natural_rows"] == 221160
            and control_training["boundary_events"] == 3240
            and candidate_training["boundary_events"] == 3240
            and control_training["natural_order_sha256"]
            == LOCKED_NATURAL_ORDER_SHA256
            and candidate_training["boundary_schedule_sha256"]
            == LOCKED_BOUNDARY_SCHEDULE_SHA256
        ),
        "both_training_curves_and_gradients_finite": bool(
            control_training["all_losses_finite"]
            and candidate_training["all_losses_finite"]
            and control_training["all_gradients_finite"]
            and candidate_training["all_gradients_finite"]
        ),
        "both_all_trainable_parameter_families_receive_gradient": bool(
            control_training["all_trainable_gradients_seen"]
            and candidate_training["all_trainable_gradients_seen"]
        ),
        "dual_values_finite_nonnegative_rank_positive": bool(
            dual["finite_nonnegative"] and int(dual["rank_nonzero"]) > 0
        ),
        "keeper_frozen_bit_exact": frozen_keeper_before == frozen_keeper_after,
        "holdout_mechanisms_finite": all(
            values["all_finite"]
            for condition in mechanisms.values()
            for values in condition.values()
        ),
        "clean_raw_replays_cidt_argmax": int(
            evaluation_summary["clean_raw_cidt_argmax_mismatches"]
        )
        == 0,
        "mcl_locked_metrics_order_probabilities_replay_exact": bool(
            mcl_replay["summary_comparisons_exact"]
            and mcl_replay["sample_order_exact"]
            and mcl_replay["probabilities_locked_by_prediction_sha256"]
            and raw_mcl_replay["sample_order_exact"]
            and raw_mcl_replay["argmax_mismatches"] == 0
            and raw_mcl_replay["probabilities_bit_exact"]
        ),
        "xai_required_selection_replays_exact": bool(
            xai["required_coverage_exact"] and xai["selection_replay_exact"]
        ),
        "xai_minimum_twelve_when_available": bool(
            xai["minimum_twelve_when_available"]
        ),
        "xai_maps_finite_nonzero_pruned_mass_zero": bool(
            xai["all_maps_finite"]
            and xai["all_defined_maps_nonzero"]
            and xai["coupling_pruned_invalid_mass_exact_zero"]
        ),
        "runtime_ratio_lte_1p15_batch32": runtime_ratio <= MAX_RUNTIME_RATIO,
        "full_peak_vram_lte_3p25_gib": float(
            candidate_benchmark["peak_vram_gib"]
        )
        <= MAX_FULL_VRAM_GIB,
        "adapter_incremental_peak_vram_lte_0p75_gib": incremental_vram
        <= MAX_INCREMENTAL_VRAM_GIB,
        "isolated_onnx_error_lte_1e5_argmax_exact": bool(
            export["isolated"]["succeeded"]
            and export["isolated"]["finite"]
            and export["isolated"]["argmax_match"]
            and float(export["isolated"]["maximum_absolute_error"])
            <= MAX_ONNX_ERROR
        ),
        "full_onnx_error_lte_1e5_argmax_exact": bool(
            export["full"]["succeeded"]
            and export["full"]["finite"]
            and export["full"]["argmax_match"]
            and float(export["full"]["maximum_absolute_error"])
            <= MAX_ONNX_ERROR
        ),
        "onnx_routes_exactly_three_iterations_without_fallback": bool(
            export["routing_iterations"] == ROUTING_ITERATIONS
            and not export["python_fallback"]
        ),
        "current_best_commands_and_history_hashes_unchanged": bool(
            provenance["sha256"]["current_commands"]
            == LOCKED_CURRENT_COMMAND_SHA256
            and provenance["sha256"]["command_history"]
            == LOCKED_COMMAND_HISTORY_SHA256
        ),
        "validation_not_used": not bool(provenance["validation_predictions_used"]),
        "test_not_used": not bool(provenance["test_data_used"]),
    }
    gate = assess_stage_a(
        structural_checks=structural_checks,
        comparisons=comparisons,
        mechanisms=mechanisms,
        mcl_comparisons=mcl_comparisons,
        dual=dual,
    )

    predictions_path = output_dir / "predictions_all_conditions.csv"
    training_path = output_dir / "training_curve.csv"
    mechanism_csv_path = output_dir / "constraint_routing_mechanism.csv"
    mechanism_json_path = output_dir / "constraint_routing_mechanism.json"
    dual_path = output_dir / "dual_multipliers.csv"
    replay_path = output_dir / "independent_replay.json"
    summary_path = output_dir / "summary.json"
    report_path = output_dir / "report.md"
    _write_predictions(
        predictions_path,
        rows=rows,
        raw_conditions=raw_conditions,
        control_conditions=control_conditions,
        candidate_conditions=candidate_conditions,
    )
    _write_training_curve(training_path, control_training, candidate_training)
    _write_mechanism_csv(mechanism_csv_path, mechanisms)
    mechanism_json_path.write_text(
        json.dumps(mechanisms, indent=2, sort_keys=True, ensure_ascii=True) + "\n",
        encoding="utf-8",
    )
    _write_dual_csv(
        dual_path,
        candidate_training=candidate_training,
        reference_indices=cohorts["reference_indices"],
        rows=rows,
    )
    replay = _replay_predictions(predictions_path, expected=comparisons)
    if not bool(replay["comparisons_exact"]):
        raise RuntimeError("Independent CapsALM prediction replay differs.")
    replay_payload = {
        "candidate_predictions": replay,
        "mcl_comparator": mcl_replay,
        "raw_keeper_vs_mcl_prior": raw_mcl_replay,
    }
    replay_path.write_text(
        json.dumps(replay_payload, indent=2, sort_keys=True, ensure_ascii=True)
        + "\n",
        encoding="utf-8",
    )
    resources = {
        "raw": raw_benchmark,
        "candidate": candidate_benchmark,
        "runtime_ratio": runtime_ratio,
        "adapter_incremental_peak_vram_gib": incremental_vram,
    }
    summary = {
        "method": METHOD,
        "status": "passed" if gate["all_gates_passed"] else "rejected",
        "decision": (
            "Authorize one locked full validation adapter evaluation."
            if gate["all_gates_passed"]
            else "Reject exact CapsALM A0; no neighboring sweep."
        ),
        "provenance": provenance,
        "cohort": _cohort_serializable(cohorts),
        "dataset": dataset_summary,
        "feature_cache": {
            "natural": natural_cache_summary,
            "boundary": boundary_cache_summary,
        },
        "construction": construction,
        "equation_diagnostics": equation,
        "initial_gradient_audit": initial_gradient,
        "training": {
            "control": _training_serializable(control_training),
            "candidate": _training_serializable(candidate_training),
        },
        "dual": dual,
        "mechanism": {"holdout": mechanisms},
        "comparisons": comparisons,
        "mcl_comparisons": mcl_comparisons,
        "evaluation": evaluation_summary,
        "independent_replay": replay_payload,
        "resources": resources,
        "onnx": export,
        "xai": xai,
        "gate": gate,
        "validation_predictions_used": False,
        "test_data_used": False,
        "current_best_commands_updated": False,
        "model_binary_written": False,
    }
    summary_path.write_text(
        json.dumps(summary, indent=2, sort_keys=True, ensure_ascii=True) + "\n",
        encoding="utf-8",
    )
    _write_report(report_path, summary)
    manifest = _write_manifest(output_dir)
    if int(manifest["forbidden_model_binary_count"]) != 0:
        raise RuntimeError("CapsALM artifact set contains a forbidden model binary.")
    return {**summary, "artifact_manifest": manifest}


def main(argv: Optional[Sequence[str]] = None) -> None:
    args = parse_args(argv)
    summary = run_audit(args)
    print(json.dumps(summary, indent=2, sort_keys=True, ensure_ascii=True))


if __name__ == "__main__":
    main()
