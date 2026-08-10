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
from typing import Dict, Mapping, Optional, Sequence

import numpy as np

os.environ.setdefault("CUBLAS_WORKSPACE_CONFIG", ":4096:8")

import torch
from PIL import Image, ImageDraw
from torch import Tensor, nn
import torch.nn.functional as F
from torch.utils.data import DataLoader
import torchvision.transforms.functional as TVF

from trkh.core.utils import set_seed
from trkh.data.dataset import MangoYOLOCropDataset
from trkh.evaluation.input_normalization import checkpoint_input_normalization
from trkh.evaluation.robustness_eval import _forward_classification_with_metadata
from trkh.inference.inference import load_checkpoint
from trkh.models.model import (
    classification_logits_from_features,
    create_model,
    extract_head_input_from_features,
    load_model_state,
)
from trkh.tools.audit_ceconv_stem_residual_readiness import (
    _FullCandidateExport,
    _build_comparisons,
    _build_dataset,
    _collect_xai_maps,
    _failed_export,
    _full_worktree_clean,
    _heat_overlay,
    _onnx_compare,
    _predict_paired_conditions,
    _replay_predictions,
    _rgb_from_tensor,
    _tracked_worktree_clean,
    _write_predictions,
    balanced_boundary_order,
)
from trkh.tools.audit_class1_boundary_cagrad_readiness import (
    CONDITIONS,
    _read_prior_predictions,
)
from trkh.tools.audit_class1_reference_agem_readiness import (
    EXPECTED_HARD_ROWS,
    EXPECTED_HOLDOUT_ROWS,
    EXPECTED_REFERENCE_ROWS,
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
from trkh.tools.audit_visual_contrast_attention_readiness import _state_sha256
from trkh.tools.audit_xca_dual_axis_readiness import (
    _amp_dtype,
    _comparison,
    _forward_logits,
    _git_commit,
    _make_loader,
    _metadata_to_device,
    _prepare_output_dir,
    _sha256,
    _tensor_sha256,
    _verify_sha256,
)


METHOD = "augself_color_adapter_a0"
BATCH_SIZE = 32
HALF_BATCH = 16
TRAIN_STEPS = 60
LEARNING_RATE = 5e-4
WEIGHT_DECAY = 0.01
AUXILIARY_WEIGHT = 1.0
SEED = 42
FOLD = 0
EMBED_DIM = 256
ADAPTER_HIDDEN_DIM = 64
PREDICTOR_HIDDEN_DIM = 512
EXPECTED_KEEPER_PARAMETERS = 7_245_590
EXPECTED_ADAPTER_PARAMETERS = 32_768
EXPECTED_PREDICTOR_PARAMETERS = 528_388
EXPECTED_TRAIN_ROWS = BATCH_SIZE * TRAIN_STEPS
JITTER_PROBABILITY = 0.8
JITTER_FACTOR_MIN = 0.6
JITTER_FACTOR_MAX = 1.4
JITTER_HUE_MIN = -0.1
JITTER_HUE_MAX = 0.1
MAX_RUNTIME_RATIO = 1.10
MAX_PEAK_VRAM_GIB = 0.75
MAX_ONNX_ERROR = 1e-5
STATIC_EXPORT_BATCH_SIZE = 1
XAI_OCCLUSION_GRID = 4

LOCKED_KEEPER_SHA256 = "1f49d577240c69dc63c30af70db52ec2aa9da65a17aef1c4b1c09ece6c482677"
LOCKED_LAUNCHER_ARGS_SHA256 = "908a05cf66b2a01162cae62e4ff2251eaae1297d31e70510144e4954159b7eff"
LOCKED_DATA_SHA256 = "716e33df24c63a9e9920f97b685199707fb84ab4c7154544f5dd9a3e00d884ef"
LOCKED_CIDT_SUMMARY_SHA256 = "d4891edf2963ab12385b7ce5bdc812ec3e19c5c098acd25c66eb557af541d7ad"
LOCKED_CIDT_PREDICTIONS_SHA256 = "2e0993752d58d99ea429bfefe1e2bfe6fa949e45aea1a26cc4bdfee97d4db21c"
LOCKED_CAGRAD_SUMMARY_SHA256 = "dab24c6dd7dcf873e523915aadbdd1f8ef5337801278d9ffc55ce8c867a37750"
LOCKED_CAGRAD_PREDICTIONS_SHA256 = "92dc296345e34d3f0ac026db2dcb2f982e40af79e1179136dfbb17f1cc2b9d9f"
LOCKED_CAGRAD_MANIFEST_SHA256 = "7f1f2a18846cd54fb7a7eafe2964a17681b99d009a70aadfaa09be54d43d1ced"
LOCKED_PROTOCOL_SHA256 = "248f1ade029bf641cbaef0be3125c7390d4670ce0b78100d2d1867839f3203ae"
LOCKED_PAPER_SHA256 = "1289ef0c57bd9fb1bf29db51540a944ba1435f080483de5b59263cc1f44eb3a1"
LOCKED_SUPPLEMENT_SHA256 = "c0355e117d20975e5da896acf601c0162796f27b932c7d3210447f4f09e58c03"
LOCKED_OFFICIAL_COMMIT = "c131db66b5ade96af86774bc43a2cb797390bba5"
LOCKED_README_SHA256 = "0d86607921f9d506c316dfb48a993a65147dd811b99436ecca97a1aa0f5d7088"
LOCKED_MODELS_SHA256 = "860b86ea0a8c03f0b6d3c2d4b53e149408f22f40c981f4dc75f2b59e7e3ee014"
LOCKED_TRAINERS_SHA256 = "bed42e208164afdef32a0cb4f063ee2e1867b9bd6eade70107e6e7e3e65b9e96"
LOCKED_TRANSFORMS_SHA256 = "eb394ae72fa0276f854c0f51127b3a61da812144545f9023bc4c3a4f02dbb62d"
LOCKED_CURRENT_COMMAND_SHA256 = "36b9aa1a21b765829acf4c8321be147bd76297de4ccdb8a40e6dee8e37940faf"
LOCKED_COMMAND_HISTORY_SHA256 = "39bd2879ce66fddf36a953021ea1e40f8d9de6cb4334b9b825011b2b8dc98f53"
LOCKED_FIT_INDEX_SHA256 = "22edca99022fe2dcd0287a08b5f6b7c0d699603b904f65c82b51fdea7f31ce5d"
LOCKED_HOLDOUT_INDEX_SHA256 = "a628686b491c8b8f10bbf1782e6c84a923f21c8b53617cf0b0325260e97e97ae"
LOCKED_REFERENCE_INDEX_SHA256 = "1474f2d8e8fe83767e3c250fd89e827f1544fb058c762bc9c114b150a18d2574"
LOCKED_HARD_INDEX_SHA256 = "56dfdbc6180d99b80605799bcd8e2ef412f0abd5c0191054b4fe7d6554d7f020"
LOCKED_TRAIN_ORDER_SHA256 = "8dd3b2b74b5975b2ce430f85b833cd75e98e3f96ae518850737aef80df31fe5a"


class HeadResidualAdapter(nn.Module):
    def __init__(self, dim: int = EMBED_DIM, hidden_dim: int = ADAPTER_HIDDEN_DIM) -> None:
        super().__init__()
        self.dim = int(dim)
        self.hidden_dim = int(hidden_dim)
        self.norm = nn.LayerNorm(self.dim, elementwise_affine=False)
        self.down = nn.Linear(self.dim, self.hidden_dim, bias=False)
        self.up = nn.Linear(self.hidden_dim, self.dim, bias=False)
        nn.init.zeros_(self.up.weight)

    def forward(self, embedding: Tensor) -> Tensor:
        residual = self.up(F.gelu(self.down(self.norm(embedding))))
        return embedding + residual


class AdaptedLinearHead(nn.Module):
    def __init__(self, classifier: nn.Linear) -> None:
        super().__init__()
        if not isinstance(classifier, nn.Linear):
            raise TypeError("AugSelf A0 requires the keeper's linear classification head.")
        self.adapter = HeadResidualAdapter(classifier.in_features, ADAPTER_HIDDEN_DIM)
        self.classifier = copy.deepcopy(classifier)

    @property
    def in_features(self) -> int:
        return int(self.classifier.in_features)

    @property
    def out_features(self) -> int:
        return int(self.classifier.out_features)

    def forward(self, embedding: Tensor) -> Tensor:
        return self.classifier(self.adapter(embedding))


class AugSelfColorPredictor(nn.Module):
    def __init__(
        self,
        embedding_dim: int = EMBED_DIM,
        hidden_dim: int = PREDICTOR_HIDDEN_DIM,
    ) -> None:
        super().__init__()
        pair_dim = 2 * int(embedding_dim)
        hidden_dim = int(hidden_dim)
        self.network = nn.Sequential(
            nn.Linear(pair_dim, hidden_dim, bias=False),
            nn.BatchNorm1d(hidden_dim),
            nn.ReLU(inplace=False),
            nn.Linear(hidden_dim, hidden_dim, bias=False),
            nn.BatchNorm1d(hidden_dim),
            nn.ReLU(inplace=False),
            nn.Linear(hidden_dim, 4, bias=True),
        )

    def forward(self, first: Tensor, second: Tensor) -> Tensor:
        return torch.tanh(self.network(torch.cat((first, second), dim=1)))


class _IsolatedAdapterExport(nn.Module):
    def __init__(self, adapter: HeadResidualAdapter) -> None:
        super().__init__()
        self.adapter = copy.deepcopy(adapter).cpu().eval()

    def forward(self, embedding: Tensor) -> Tensor:
        return self.adapter(embedding)


def parse_args(argv: Optional[Sequence[str]] = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Locked train-only AugSelf color-adapter information gate. "
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
        "--cagrad-summary",
        type=Path,
        default=Path("runs/audit_class1_boundary_cagrad_readiness_20260715/summary.json"),
    )
    parser.add_argument(
        "--cagrad-predictions",
        type=Path,
        default=Path(
            "runs/audit_class1_boundary_cagrad_readiness_20260715/"
            "predictions_all_conditions.csv"
        ),
    )
    parser.add_argument(
        "--cagrad-manifest",
        type=Path,
        default=Path(
            "runs/audit_class1_boundary_cagrad_readiness_20260715/"
            "artifact_manifest.json"
        ),
    )
    parser.add_argument(
        "--protocol",
        type=Path,
        default=Path(
            "docs/TRKH_5CLASS_AUGSELF_COLOR_ADAPTER_READINESS_PROTOCOL_20260716.md"
        ),
    )
    parser.add_argument(
        "--paper",
        type=Path,
        default=Path(r"D:\DataAI\external_sources\papers\augself_neurips2021.pdf"),
    )
    parser.add_argument(
        "--supplement",
        type=Path,
        default=Path(
            r"D:\DataAI\external_sources\papers\augself_neurips2021_supplemental.pdf"
        ),
    )
    parser.add_argument(
        "--official-root",
        type=Path,
        default=Path(r"D:\DataAI\external_sources\official\augself"),
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=Path("runs/audit_augself_color_adapter_readiness_20260716"),
    )
    parser.add_argument("--preflight-only", action="store_true", default=False)
    parser.add_argument("--device", choices=("cuda",), default="cuda")
    parser.add_argument("--batch-size", type=int, default=BATCH_SIZE)
    parser.add_argument("--num-workers", type=int, default=4)
    parser.add_argument("--seed", type=int, default=SEED)
    parser.add_argument("--fold", type=int, default=FOLD)
    parser.add_argument("--max-train-batches", type=int, default=TRAIN_STEPS)
    parser.add_argument("--learning-rate", type=float, default=LEARNING_RATE)
    parser.add_argument("--weight-decay", type=float, default=WEIGHT_DECAY)
    parser.add_argument("--auxiliary-weight", type=float, default=AUXILIARY_WEIGHT)
    parser.add_argument("--benchmark-repeats", type=int, default=7)
    parser.add_argument("--xai-batch-size", type=int, default=4)
    return parser.parse_args(argv)


def _locked_args_exact(args: argparse.Namespace) -> bool:
    return bool(
        str(args.device) == "cuda"
        and int(args.batch_size) == BATCH_SIZE
        and int(args.num_workers) == 4
        and int(args.seed) == SEED
        and int(args.fold) == FOLD
        and int(args.max_train_batches) == TRAIN_STEPS
        and math.isclose(float(args.learning_rate), LEARNING_RATE, abs_tol=0.0)
        and math.isclose(float(args.weight_decay), WEIGHT_DECAY, abs_tol=0.0)
        and math.isclose(float(args.auxiliary_weight), AUXILIARY_WEIGHT, abs_tol=0.0)
        and int(args.benchmark_repeats) >= 3
        and int(args.xai_batch_size) >= 1
    )


def _source_paths(args: argparse.Namespace) -> Dict[str, Path]:
    official = Path(args.official_root).resolve()
    return {
        "checkpoint": Path(args.checkpoint).resolve(),
        "launcher_args": Path(args.launcher_args).resolve(),
        "data": Path(args.data).resolve(),
        "cidt_summary": Path(args.cidt_summary).resolve(),
        "cidt_predictions": Path(args.cidt_predictions).resolve(),
        "cagrad_summary": Path(args.cagrad_summary).resolve(),
        "cagrad_predictions": Path(args.cagrad_predictions).resolve(),
        "cagrad_manifest": Path(args.cagrad_manifest).resolve(),
        "protocol": Path(args.protocol).resolve(),
        "paper": Path(args.paper).resolve(),
        "supplement": Path(args.supplement).resolve(),
        "official_root": official,
        "official_readme": official / "README.md",
        "official_models": official / "models.py",
        "official_trainers": official / "trainers.py",
        "official_transforms": official / "transforms.py",
        "current_commands": Path(
            "docs/TRKH_CURRENT_BEST_FULL_TRAIN_COMMANDS_20260706.txt"
        ).resolve(),
        "command_history": Path(
            "docs/TRKH_CURRENT_BEST_COMMAND_UPDATE_HISTORY.txt"
        ).resolve(),
    }


def _load_locked_inputs(
    args: argparse.Namespace,
) -> tuple[
    Dict[str, object],
    list[CleanTrainRow],
    Dict[str, object],
    Dict[str, Dict[str, list[Dict[str, object]]]],
]:
    if not _locked_args_exact(args):
        raise ValueError("Arguments differ from the precommitted AugSelf protocol.")
    paths = _source_paths(args)
    hashes = {
        "checkpoint": _verify_sha256(paths["checkpoint"], LOCKED_KEEPER_SHA256, "keeper"),
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
        "cagrad_summary": _verify_sha256(
            paths["cagrad_summary"], LOCKED_CAGRAD_SUMMARY_SHA256, "CAGrad summary"
        ),
        "cagrad_predictions": _verify_sha256(
            paths["cagrad_predictions"],
            LOCKED_CAGRAD_PREDICTIONS_SHA256,
            "CAGrad predictions",
        ),
        "cagrad_manifest": _verify_sha256(
            paths["cagrad_manifest"], LOCKED_CAGRAD_MANIFEST_SHA256, "CAGrad manifest"
        ),
        "protocol": _verify_sha256(
            paths["protocol"], LOCKED_PROTOCOL_SHA256, "AugSelf protocol"
        ),
        "paper": _verify_sha256(paths["paper"], LOCKED_PAPER_SHA256, "AugSelf paper"),
        "supplement": _verify_sha256(
            paths["supplement"], LOCKED_SUPPLEMENT_SHA256, "AugSelf supplement"
        ),
        "official_readme": _verify_sha256(
            paths["official_readme"], LOCKED_README_SHA256, "official README"
        ),
        "official_models": _verify_sha256(
            paths["official_models"], LOCKED_MODELS_SHA256, "official models.py"
        ),
        "official_trainers": _verify_sha256(
            paths["official_trainers"], LOCKED_TRAINERS_SHA256, "official trainers.py"
        ),
        "official_transforms": _verify_sha256(
            paths["official_transforms"],
            LOCKED_TRANSFORMS_SHA256,
            "official transforms.py",
        ),
        "current_commands": _verify_sha256(
            paths["current_commands"],
            LOCKED_CURRENT_COMMAND_SHA256,
            "current-best command file",
        ),
        "command_history": _verify_sha256(
            paths["command_history"],
            LOCKED_COMMAND_HISTORY_SHA256,
            "current-best command history",
        ),
    }
    official_commit = _git_commit(paths["official_root"])
    official_clean = _full_worktree_clean(paths["official_root"])
    license_matches = sorted(
        str(path.name)
        for pattern in ("LICENSE*", "COPYING*")
        for path in paths["official_root"].glob(pattern)
        if path.is_file()
    )
    if official_commit != LOCKED_OFFICIAL_COMMIT:
        raise ValueError(
            f"Official AugSelf commit differs: {official_commit} != {LOCKED_OFFICIAL_COMMIT}"
        )
    if not official_clean:
        raise ValueError("Official AugSelf worktree is not clean.")
    if license_matches:
        raise ValueError(f"Official AugSelf license state changed: {license_matches}")
    for name in ("cidt_summary", "cagrad_summary"):
        payload = json.loads(paths[name].read_text(encoding="utf-8"))
        if bool(payload.get("test_data_used", True)):
            raise ValueError(f"{name} provenance indicates test data use.")
        if bool(
            payload.get(
                "validation_predictions_used",
                payload.get("validation_data_used", True),
            )
        ):
            raise ValueError(f"{name} provenance indicates validation use.")
    rows = _read_clean_train_rows(paths["cidt_predictions"])
    cohorts = _locked_cohort_summary(rows, fold=int(args.fold))
    observed_hashes = {
        "fit": str(cohorts["fit_index_sha256"]),
        "holdout": str(cohorts["holdout_index_sha256"]),
        "reference": str(cohorts["reference_index_sha256"]),
        "hard": str(cohorts["hard_index_sha256"]),
    }
    expected_hashes = {
        "fit": LOCKED_FIT_INDEX_SHA256,
        "holdout": LOCKED_HOLDOUT_INDEX_SHA256,
        "reference": LOCKED_REFERENCE_INDEX_SHA256,
        "hard": LOCKED_HARD_INDEX_SHA256,
    }
    if observed_hashes != expected_hashes:
        raise ValueError(
            f"AugSelf cohort hashes differ: {observed_hashes} != {expected_hashes}"
        )
    prior, argmax_mismatches = _read_prior_predictions(
        paths["cagrad_predictions"],
        rows=rows,
        holdout_indices=cohorts["holdout_indices"],
    )
    repo_root = Path.cwd().resolve()
    tracked_clean = _tracked_worktree_clean(repo_root)
    if not tracked_clean:
        raise ValueError("Tracked TRKH worktree must be clean for the formal AugSelf gate.")
    return (
        {
            "paths": {key: str(value) for key, value in paths.items()},
            "sha256": hashes,
            "official_commit": official_commit,
            "official_worktree_clean": official_clean,
            "official_license_absent": not license_matches,
            "repository_commit": _git_commit(repo_root),
            "tracked_worktree_clean": tracked_clean,
            "prior_argmax_mismatches": argmax_mismatches,
            "validation_predictions_used": False,
            "test_data_used": False,
        },
        rows,
        cohorts,
        prior,
    )


def private_color_seed(
    *, phase: str, seed: int, step: int, sample_index: int, view: int
) -> int:
    key = f"augself-color|{phase}|{seed}|{step}|{sample_index}|{view}"
    digest = hashlib.sha256(key.encode("utf-8")).digest()
    return int.from_bytes(digest[:8], byteorder="little", signed=False) % (2**63 - 1)


def _normalization_tensors(
    mean: Sequence[float], std: Sequence[float], *, dtype: torch.dtype
) -> tuple[Tensor, Tensor]:
    mean_tensor = torch.tensor(mean, dtype=dtype).view(3, 1, 1)
    std_tensor = torch.tensor(std, dtype=dtype).view(3, 1, 1)
    return mean_tensor, std_tensor


def make_color_view(
    image: Tensor,
    *,
    phase: str,
    seed: int,
    step: int,
    sample_index: int,
    view: int,
    mean: Sequence[float],
    std: Sequence[float],
) -> tuple[Tensor, Dict[str, object]]:
    if image.device.type != "cpu" or image.ndim != 3 or int(image.size(0)) != 3:
        raise ValueError("Color views require one CPU CHW RGB tensor.")
    private_seed = private_color_seed(
        phase=phase,
        seed=seed,
        step=step,
        sample_index=sample_index,
        view=view,
    )
    generator = torch.Generator(device="cpu")
    generator.manual_seed(private_seed)
    mean_tensor, std_tensor = _normalization_tensors(mean, std, dtype=image.dtype)
    rgb = (image.detach() * std_tensor + mean_tensor).clamp(0.0, 1.0)
    active = bool(torch.rand((), generator=generator).item() < JITTER_PROBABILITY)
    operation_names = ("brightness", "contrast", "saturation", "hue")
    if active:
        factors = torch.empty(4, dtype=torch.float64)
        factors[:3].uniform_(
            JITTER_FACTOR_MIN,
            JITTER_FACTOR_MAX,
            generator=generator,
        )
        factors[3].uniform_(JITTER_HUE_MIN, JITTER_HUE_MAX, generator=generator)
        order_indices = torch.randperm(4, generator=generator).tolist()
        ordered_names = [operation_names[index] for index in order_indices]
        brightness, contrast, saturation, hue = [float(value) for value in factors]
        for operation in ordered_names:
            if operation == "brightness":
                rgb = TVF.adjust_brightness(rgb, brightness)
            elif operation == "contrast":
                rgb = TVF.adjust_contrast(rgb, contrast)
            elif operation == "saturation":
                rgb = TVF.adjust_saturation(rgb, saturation)
            elif operation == "hue":
                rgb = TVF.adjust_hue(rgb, hue)
            else:
                raise AssertionError(operation)
            rgb = rgb.clamp(0.0, 1.0)
        centered = (
            (brightness - 1.0) / 0.8,
            (contrast - 1.0) / 0.8,
            (saturation - 1.0) / 0.8,
            hue / 0.2,
        )
        raw_parameters = (brightness, contrast, saturation, hue)
    else:
        ordered_names = []
        centered = (0.0, 0.0, 0.0, 0.0)
        raw_parameters = (1.0, 1.0, 1.0, 0.0)
    normalized = ((rgb - mean_tensor) / std_tensor).contiguous()
    record = {
        "phase": str(phase),
        "seed": int(seed),
        "private_seed": int(private_seed),
        "step": int(step),
        "sample_index": int(sample_index),
        "view": int(view),
        "active": active,
        "operation_order": ordered_names,
        "raw_parameters": [float(value) for value in raw_parameters],
        "centered_parameters": [float(value) for value in centered],
        "tensor_sha256": _tensor_sha256(normalized),
    }
    return normalized, record


def make_color_pair_batch(
    images: Tensor,
    sample_indices: Sequence[int],
    *,
    phase: str,
    seed: int,
    step: int,
    mean: Sequence[float],
    std: Sequence[float],
) -> tuple[Tensor, Tensor, Tensor, list[Dict[str, object]]]:
    if images.device.type != "cpu" or int(images.size(0)) != len(sample_indices):
        raise ValueError("Color pair batch/image index contract differs.")
    first_views = []
    second_views = []
    targets = []
    records: list[Dict[str, object]] = []
    for position, raw_index in enumerate(sample_indices):
        sample_index = int(raw_index)
        first, first_record = make_color_view(
            images[position],
            phase=phase,
            seed=seed,
            step=step,
            sample_index=sample_index,
            view=1,
            mean=mean,
            std=std,
        )
        second, second_record = make_color_view(
            images[position],
            phase=phase,
            seed=seed,
            step=step,
            sample_index=sample_index,
            view=2,
            mean=mean,
            std=std,
        )
        first_parameters = torch.tensor(
            first_record["centered_parameters"], dtype=torch.float32
        )
        second_parameters = torch.tensor(
            second_record["centered_parameters"], dtype=torch.float32
        )
        target = first_parameters - second_parameters
        first_views.append(first)
        second_views.append(second)
        targets.append(target)
        records.append(
            {
                "phase": str(phase),
                "step": int(step),
                "sample_index": sample_index,
                "view_1": first_record,
                "view_2": second_record,
                "target_12": [float(value) for value in target.tolist()],
                "target_21": [float(value) for value in (-target).tolist()],
                "reversal_exact": bool(torch.equal(-target, second_parameters - first_parameters)),
            }
        )
    return (
        torch.stack(first_views),
        torch.stack(second_views),
        torch.stack(targets),
        records,
    )


def _canonical_sha256(value: object) -> str:
    payload = json.dumps(value, sort_keys=True, separators=(",", ":"), ensure_ascii=True)
    return hashlib.sha256(payload.encode("utf-8")).hexdigest()


def symmetric_color_loss(
    predictor: AugSelfColorPredictor,
    first: Tensor,
    second: Tensor,
    target_12: Tensor,
    *,
    detach_embeddings: bool,
) -> tuple[Tensor, Tensor, Tensor]:
    if detach_embeddings:
        first = first.detach()
        second = second.detach()
    prediction_12 = predictor(first, second)
    prediction_21 = predictor(second, first)
    loss = 0.5 * (
        F.mse_loss(prediction_12.float(), target_12.float())
        + F.mse_loss(prediction_21.float(), -target_12.float())
    )
    return loss, prediction_12, prediction_21


def _construct_models(
    checkpoint: Mapping[str, object],
) -> tuple[nn.Module, nn.Module, nn.Module, AugSelfColorPredictor, AugSelfColorPredictor, Dict[str, object]]:
    model_config = checkpoint.get("model_config")
    model_state = checkpoint.get("model_state")
    class_names = checkpoint.get("class_names")
    if not isinstance(model_config, Mapping) or not isinstance(model_state, Mapping):
        raise ValueError("Keeper model config/state is invalid.")
    if not isinstance(class_names, list) or len(class_names) != 5:
        raise ValueError("Keeper class order is invalid.")
    set_seed(SEED, deterministic=True)
    raw = create_model(num_classes=5, model_config=model_config).eval()
    load_model_state(raw, dict(model_state), strict=True)
    if not isinstance(raw.head, nn.Linear):
        raise TypeError(f"Keeper head is not linear: {type(raw.head).__name__}")
    if int(raw.head.in_features) != EMBED_DIM or int(raw.head.out_features) != 5:
        raise ValueError("Keeper head dimensions differ from the locked protocol.")
    prototype = copy.deepcopy(raw)
    prototype.head = AdaptedLinearHead(prototype.head)
    predictor_prototype = AugSelfColorPredictor()
    control = copy.deepcopy(prototype).eval()
    candidate = copy.deepcopy(prototype).eval()
    control_predictor = copy.deepcopy(predictor_prototype).eval()
    candidate_predictor = copy.deepcopy(predictor_prototype).eval()
    raw_parameters = sum(parameter.numel() for parameter in raw.parameters())
    adapter_parameters = sum(
        parameter.numel() for parameter in candidate.head.adapter.parameters()
    )
    predictor_parameters = sum(
        parameter.numel() for parameter in candidate_predictor.parameters()
    )
    return raw, control, candidate, control_predictor, candidate_predictor, {
        "raw_state_sha256": _state_sha256(raw),
        "control_state_sha256": _state_sha256(control),
        "candidate_state_sha256": _state_sha256(candidate),
        "control_predictor_state_sha256": _state_sha256(control_predictor),
        "candidate_predictor_state_sha256": _state_sha256(candidate_predictor),
        "candidate_control_model_bit_exact": _state_sha256(control)
        == _state_sha256(candidate),
        "candidate_control_predictor_bit_exact": _state_sha256(control_predictor)
        == _state_sha256(candidate_predictor),
        "raw_parameters": raw_parameters,
        "adapter_parameters": adapter_parameters,
        "predictor_parameters": predictor_parameters,
        "total_trainable_parameters": adapter_parameters + predictor_parameters,
        "adapter_up_zero_exact": bool(
            torch.count_nonzero(candidate.head.adapter.up.weight).item() == 0
        ),
    }


def _configure_trainability(model: nn.Module) -> list[str]:
    allowed = {"head.adapter.down.weight", "head.adapter.up.weight"}
    trainable = []
    for name, parameter in model.named_parameters():
        parameter.requires_grad_(name in allowed)
        if parameter.requires_grad:
            trainable.append(name)
    if set(trainable) != allowed:
        raise ValueError(f"AugSelf adapter trainable schema differs: {trainable}")
    model.eval()
    model.head.adapter.train()
    return trainable


def _frozen_model_sha256(model: nn.Module) -> str:
    digest = hashlib.sha256()
    for name, value in sorted(model.state_dict().items()):
        if name.startswith("head.adapter."):
            continue
        digest.update(name.encode("utf-8"))
        digest.update(value.detach().cpu().contiguous().numpy().tobytes())
    return digest.hexdigest()


def _trainable_snapshot(
    model: nn.Module, predictor: nn.Module
) -> Dict[str, Tensor]:
    result = {
        f"model.{name}": parameter.detach().cpu().clone()
        for name, parameter in model.named_parameters()
        if parameter.requires_grad
    }
    result.update(
        {
            f"predictor.{name}": parameter.detach().cpu().clone()
            for name, parameter in predictor.named_parameters()
            if parameter.requires_grad
        }
    )
    return result


def _forward_augmented(
    model: nn.Module,
    images: Tensor,
    metadata: Mapping[str, object],
    *,
    device: torch.device,
) -> tuple[Tensor, Tensor]:
    logits, features = _forward_classification_with_metadata(
        model,
        images,
        metadata,
        device=device,
    )
    if not isinstance(features, Mapping):
        raise TypeError("Keeper forward did not return a feature mapping.")
    head_input = extract_head_input_from_features(model, dict(features))
    embedding = model.head.adapter(head_input)
    return logits, embedding


def _initial_exactness(
    *,
    raw: nn.Module,
    control: nn.Module,
    candidate: nn.Module,
    images_cpu: Tensor,
    metadata_cpu: Mapping[str, object],
    prior_clean: Sequence[Mapping[str, object]],
    device: torch.device,
) -> Dict[str, object]:
    count = min(BATCH_SIZE, int(images_cpu.size(0)))
    images = images_cpu[:count].to(device)
    metadata = _metadata_to_device(metadata_cpu, device=device, count=count)
    outputs: Dict[str, Tensor] = {}
    for name, prototype in (("raw", raw), ("control", control), ("candidate", candidate)):
        model = copy.deepcopy(prototype).to(device).eval()
        with torch.inference_mode():
            outputs[name] = _forward_logits(
                model,
                images,
                metadata,
                device=device,
            ).detach().cpu()
        del model
        gc.collect()
        torch.cuda.empty_cache()
    sample_indices = metadata_cpu.get("sample_index")
    if not torch.is_tensor(sample_indices):
        raise ValueError("Initial exactness metadata is missing sample_index.")
    prior_by_index = {int(row["sample_index"]): row for row in prior_clean}
    prior_probabilities = torch.tensor(
        [
            [
                float(prior_by_index[int(index)][f"prob_{class_index}"])
                for class_index in range(5)
            ]
            for index in sample_indices[:count].tolist()
        ],
        dtype=torch.float32,
    )
    raw_probabilities = outputs["raw"].softmax(dim=1)
    return {
        "rows": count,
        "raw_control_logits_bit_exact": torch.equal(outputs["raw"], outputs["control"]),
        "raw_candidate_logits_bit_exact": torch.equal(outputs["raw"], outputs["candidate"]),
        "control_candidate_logits_bit_exact": torch.equal(
            outputs["control"], outputs["candidate"]
        ),
        "raw_control_max_abs_error": float(
            (outputs["raw"] - outputs["control"]).abs().amax().item()
        ),
        "raw_candidate_max_abs_error": float(
            (outputs["raw"] - outputs["candidate"]).abs().amax().item()
        ),
        "raw_prior_argmax_match": torch.equal(
            raw_probabilities.argmax(dim=1), prior_probabilities.argmax(dim=1)
        ),
        "raw_prior_probability_max_abs_error": float(
            (raw_probabilities - prior_probabilities).abs().amax().item()
        ),
    }


def _gradient_map(module: nn.Module) -> Dict[str, Optional[Tensor]]:
    return {
        name: None if parameter.grad is None else parameter.grad.detach().cpu().clone()
        for name, parameter in module.named_parameters()
    }


def _gradient_norm(values: Mapping[str, Optional[Tensor]]) -> float:
    squared = 0.0
    for value in values.values():
        if value is not None:
            squared += float(value.double().square().sum().item())
    return math.sqrt(squared)


def _initial_gradient_audit(
    *,
    control: nn.Module,
    candidate: nn.Module,
    control_predictor: AugSelfColorPredictor,
    candidate_predictor: AugSelfColorPredictor,
    images_cpu: Tensor,
    metadata_cpu: Mapping[str, object],
    sample_indices: Sequence[int],
    mean: Sequence[float],
    std: Sequence[float],
    device: torch.device,
) -> Dict[str, object]:
    first_cpu, second_cpu, target_cpu, records = make_color_pair_batch(
        images_cpu,
        sample_indices,
        phase="gradient_audit",
        seed=SEED,
        step=0,
        mean=mean,
        std=std,
    )
    outputs: Dict[str, Dict[str, object]] = {}
    predictor_gradients: Dict[str, Dict[str, Optional[Tensor]]] = {}
    for name, model_prototype, predictor_prototype, detach in (
        ("control", control, control_predictor, True),
        ("candidate", candidate, candidate_predictor, False),
    ):
        set_seed(SEED, deterministic=True)
        model = copy.deepcopy(model_prototype).to(device)
        predictor = copy.deepcopy(predictor_prototype).to(device)
        _configure_trainability(model)
        predictor.train()
        model.zero_grad(set_to_none=True)
        predictor.zero_grad(set_to_none=True)
        first = first_cpu.to(device)
        second = second_cpu.to(device)
        target = target_cpu.to(device)
        _, first_embedding = _forward_augmented(
            model, first, metadata_cpu, device=device
        )
        _, second_embedding = _forward_augmented(
            model, second, metadata_cpu, device=device
        )
        auxiliary, prediction_12, prediction_21 = symmetric_color_loss(
            predictor,
            first_embedding,
            second_embedding,
            target,
            detach_embeddings=detach,
        )
        auxiliary.backward()
        adapter_gradients = _gradient_map(model.head.adapter)
        predictor_gradient = _gradient_map(predictor)
        predictor_gradients[name] = predictor_gradient
        outputs[name] = {
            "loss": float(auxiliary.detach().item()),
            "prediction_12_sha256": _tensor_sha256(prediction_12),
            "prediction_21_sha256": _tensor_sha256(prediction_21),
            "adapter_gradient_norm": _gradient_norm(adapter_gradients),
            "adapter_gradient_finite": all(
                value is None or bool(torch.isfinite(value).all())
                for value in adapter_gradients.values()
            ),
            "adapter_gradient_presence": {
                key: value is not None for key, value in adapter_gradients.items()
            },
            "predictor_gradient_norm": _gradient_norm(predictor_gradient),
            "predictor_gradient_finite": all(
                value is not None and bool(torch.isfinite(value).all())
                for value in predictor_gradient.values()
            ),
        }
        del model, predictor, first, second, target
        gc.collect()
        torch.cuda.empty_cache()
    control_gradients = predictor_gradients["control"]
    candidate_gradients = predictor_gradients["candidate"]
    same_keys = control_gradients.keys() == candidate_gradients.keys()
    predictor_gradients_bit_exact = bool(
        same_keys
        and all(
            control_gradients[name] is not None
            and candidate_gradients[name] is not None
            and torch.equal(control_gradients[name], candidate_gradients[name])
            for name in control_gradients
        )
    )
    return {
        "rows": len(sample_indices),
        "view_record_sha256": _canonical_sha256(records),
        "target_reversal_exact": all(bool(row["reversal_exact"]) for row in records),
        "control": outputs["control"],
        "candidate": outputs["candidate"],
        "candidate_adapter_aux_gradient_nonzero": float(
            outputs["candidate"]["adapter_gradient_norm"]
        )
        > 0.0,
        "control_adapter_aux_gradient_exact_zero": float(
            outputs["control"]["adapter_gradient_norm"]
        )
        == 0.0,
        "predictor_gradients_bit_exact": predictor_gradients_bit_exact,
        "predictor_gradient_max_abs_error": max(
            (
                float(
                    (control_gradients[name] - candidate_gradients[name])
                    .abs()
                    .amax()
                    .item()
                )
                for name in control_gradients
                if control_gradients[name] is not None
                and candidate_gradients[name] is not None
            ),
            default=math.inf,
        ),
    }


def _movement_summary(
    initial: Mapping[str, Tensor], model: nn.Module, predictor: nn.Module
) -> Dict[str, object]:
    final = _trainable_snapshot(model, predictor)
    return {
        name: {
            "initial_sha256": _tensor_sha256(value),
            "final_sha256": _tensor_sha256(final[name]),
            "changed": not torch.equal(value, final[name]),
            "maximum_absolute_delta": float((value - final[name]).abs().amax().item()),
        }
        for name, value in initial.items()
    }


def _train_variant(
    *,
    name: str,
    model_prototype: nn.Module,
    predictor_prototype: AugSelfColorPredictor,
    detach_embeddings: bool,
    base_dataset: MangoYOLOCropDataset,
    transform,
    train_indices: Sequence[int],
    mean: Sequence[float],
    std: Sequence[float],
    args: argparse.Namespace,
    device: torch.device,
) -> tuple[nn.Module, AugSelfColorPredictor, Dict[str, object], list[Dict[str, object]]]:
    started = time.perf_counter()
    set_seed(int(args.seed), deterministic=True)
    model = copy.deepcopy(model_prototype).to(device)
    predictor = copy.deepcopy(predictor_prototype).to(device)
    model_trainable = _configure_trainability(model)
    predictor.train()
    for parameter in predictor.parameters():
        parameter.requires_grad_(True)
    predictor_trainable = [name for name, _ in predictor.named_parameters()]
    initial_frozen_sha = _frozen_model_sha256(model)
    initial_trainable = _trainable_snapshot(model, predictor)
    optimizer = torch.optim.AdamW(
        [
            *[parameter for parameter in model.parameters() if parameter.requires_grad],
            *predictor.parameters(),
        ],
        lr=float(args.learning_rate),
        weight_decay=float(args.weight_decay),
    )
    optimizer_initial_state_empty = len(optimizer.state) == 0
    loader, loader_summary = _make_loader(
        base_dataset=base_dataset,
        transform=transform,
        indices=train_indices,
        batch_size=int(args.batch_size),
        num_workers=int(args.num_workers),
        context=f"augself_{name}_train",
        seed=int(args.seed) + 300,
    )
    gradient_seen = {key: False for key in initial_trainable}
    gradient_max_norm = {key: 0.0 for key in initial_trainable}
    all_gradients_finite = True
    history = []
    view_records: list[Dict[str, object]] = []
    observed_indices: list[int] = []
    for step, (images_cpu, targets_cpu, metadata_cpu) in enumerate(loader):
        if step >= TRAIN_STEPS:
            raise RuntimeError("AugSelf loader exceeded the locked 60 steps.")
        sample_index_tensor = metadata_cpu.get("sample_index")
        if not torch.is_tensor(sample_index_tensor):
            raise ValueError("AugSelf training metadata is missing sample_index.")
        sample_indices = [int(value) for value in sample_index_tensor.tolist()]
        first_cpu, second_cpu, target_cpu, records = make_color_pair_batch(
            images_cpu,
            sample_indices,
            phase="train",
            seed=int(args.seed),
            step=step,
            mean=mean,
            std=std,
        )
        observed_indices.extend(sample_indices)
        view_records.extend(records)
        first = first_cpu.to(device=device, non_blocking=True)
        second = second_cpu.to(device=device, non_blocking=True)
        targets = targets_cpu.to(device=device, dtype=torch.long, non_blocking=True)
        color_target = target_cpu.to(device=device, non_blocking=True)
        optimizer.zero_grad(set_to_none=True)
        logits_1, embedding_1 = _forward_augmented(
            model, first, metadata_cpu, device=device
        )
        logits_2, embedding_2 = _forward_augmented(
            model, second, metadata_cpu, device=device
        )
        classification = 0.5 * (
            F.cross_entropy(logits_1.float(), targets)
            + F.cross_entropy(logits_2.float(), targets)
        )
        auxiliary, prediction_12, prediction_21 = symmetric_color_loss(
            predictor,
            embedding_1,
            embedding_2,
            color_target,
            detach_embeddings=detach_embeddings,
        )
        loss = classification + float(args.auxiliary_weight) * auxiliary
        if not bool(torch.isfinite(loss)):
            raise FloatingPointError(f"Non-finite AugSelf loss at step {step}.")
        loss.backward()
        squared_gradient_norm = 0.0
        for prefix, module in (("model", model), ("predictor", predictor)):
            for parameter_name, parameter in module.named_parameters():
                if not parameter.requires_grad:
                    continue
                key = f"{prefix}.{parameter_name}"
                gradient = parameter.grad
                if gradient is None:
                    continue
                finite = bool(torch.isfinite(gradient).all())
                all_gradients_finite = all_gradients_finite and finite
                norm = float(gradient.detach().double().norm().item())
                squared_gradient_norm += norm * norm
                gradient_max_norm[key] = max(gradient_max_norm[key], norm)
                if finite and norm > 0.0:
                    gradient_seen[key] = True
        optimizer.step()
        history.append(
            {
                "variant": name,
                "step": step,
                "loss": float(loss.detach().item()),
                "classification_loss": float(classification.detach().item()),
                "color_loss": float(auxiliary.detach().item()),
                "gradient_norm": math.sqrt(squared_gradient_norm),
                "prediction_12_mean_abs": float(prediction_12.detach().abs().mean().item()),
                "prediction_21_mean_abs": float(prediction_21.detach().abs().mean().item()),
            }
        )
    model.eval()
    predictor.eval()
    frozen_final_sha = _frozen_model_sha256(model)
    movement = _movement_summary(initial_trainable, model, predictor)
    return model.cpu(), predictor.cpu(), {
        "variant": name,
        "detach_embeddings": bool(detach_embeddings),
        "steps": len(history),
        "rows": len(observed_indices),
        "train_order_sha256": _ordered_index_sha256(observed_indices),
        "view_record_sha256": _canonical_sha256(view_records),
        "all_view_targets_reversal_exact": all(
            bool(record["reversal_exact"]) for record in view_records
        ),
        "history": history,
        "loader": loader_summary,
        "model_trainable_parameters": model_trainable,
        "predictor_trainable_parameters": predictor_trainable,
        "optimizer_initial_state_empty": optimizer_initial_state_empty,
        "optimizer_final_state_entries": len(optimizer.state),
        "gradient_seen": gradient_seen,
        "gradient_max_norm": gradient_max_norm,
        "all_gradients_finite": all_gradients_finite,
        "all_trainable_gradients_seen": all(gradient_seen.values()),
        "frozen_state_initial_sha256": initial_frozen_sha,
        "frozen_state_final_sha256": frozen_final_sha,
        "frozen_state_bit_exact": initial_frozen_sha == frozen_final_sha,
        "state_movement": movement,
        "all_trainable_parameters_changed": all(
            bool(value["changed"]) for value in movement.values()
        ),
        "elapsed_seconds": float(time.perf_counter() - started),
    }, view_records


def _replay_color_loader(
    *,
    loader: DataLoader,
    expected_records: Sequence[Mapping[str, object]],
    phase: str,
    seed: int,
    mean: Sequence[float],
    std: Sequence[float],
    constant_step: Optional[int] = None,
) -> Dict[str, object]:
    replayed: list[Dict[str, object]] = []
    for loader_step, (images_cpu, _targets_cpu, metadata_cpu) in enumerate(loader):
        sample_index_tensor = metadata_cpu.get("sample_index")
        if not torch.is_tensor(sample_index_tensor):
            raise ValueError("Color replay metadata is missing sample_index.")
        sample_indices = [int(value) for value in sample_index_tensor.tolist()]
        step = int(constant_step) if constant_step is not None else loader_step
        _first, _second, _target, records = make_color_pair_batch(
            images_cpu,
            sample_indices,
            phase=phase,
            seed=seed,
            step=step,
            mean=mean,
            std=std,
        )
        replayed.extend(records)
    expected_hash = _canonical_sha256(expected_records)
    replay_hash = _canonical_sha256(replayed)
    return {
        "rows": len(replayed),
        "expected_sha256": expected_hash,
        "replayed_sha256": replay_hash,
        "records_exact": list(expected_records) == replayed,
        "sha256_exact": expected_hash == replay_hash,
        "all_reversals_exact": all(bool(row["reversal_exact"]) for row in replayed),
    }


def _prediction_row(
    *, sample_index: int, target: int, probabilities: Tensor
) -> Dict[str, object]:
    values = probabilities.detach().float().cpu()
    row: Dict[str, object] = {
        "sample_index": int(sample_index),
        "target": int(target),
        "prediction": int(values.argmax().item()),
    }
    for class_index in range(5):
        row[f"prob_{class_index}"] = float(values[class_index].item())
    return row


def _focused_holdout_indices(
    *,
    rows: Sequence[CleanTrainRow],
    holdout_indices: Sequence[int],
    raw_clean: Sequence[Mapping[str, object]],
) -> tuple[list[int], Dict[str, object]]:
    raw_by_index = {int(row["sample_index"]): row for row in raw_clean}
    class1 = [index for index in holdout_indices if rows[index].target == FOCUS_CLASS]
    restricted_fp = [
        index
        for index in holdout_indices
        if rows[index].target in RESTRICTED_NEGATIVE_CLASSES
        and int(raw_by_index[index]["prediction"]) == FOCUS_CLASS
    ]
    selected_set = set(class1).union(restricted_fp)
    selected = [int(index) for index in holdout_indices if int(index) in selected_set]
    return selected, {
        "rows": len(selected),
        "class1_rows": len(class1),
        "restricted_raw_fp_rows": len(restricted_fp),
        "ordered_index_sha256": _ordered_index_sha256(selected),
        "class1_index_sha256": _ordered_index_sha256(class1),
        "restricted_fp_index_sha256": _ordered_index_sha256(restricted_fp),
    }


def _representation_evaluation(
    *,
    control: nn.Module,
    candidate: nn.Module,
    control_predictor: AugSelfColorPredictor,
    candidate_predictor: AugSelfColorPredictor,
    base_dataset: MangoYOLOCropDataset,
    transform,
    focused_indices: Sequence[int],
    mean: Sequence[float],
    std: Sequence[float],
    args: argparse.Namespace,
    device: torch.device,
) -> tuple[Dict[str, object], list[Dict[str, object]], DataLoader]:
    loader, loader_summary = _make_loader(
        base_dataset=base_dataset,
        transform=transform,
        indices=focused_indices,
        batch_size=int(args.batch_size),
        num_workers=int(args.num_workers),
        context="augself_focused_holdout",
        seed=int(args.seed) + 600,
    )
    models = {
        "control": copy.deepcopy(control).to(device).eval(),
        "candidate": copy.deepcopy(candidate).to(device).eval(),
    }
    predictors = {
        "control": copy.deepcopy(control_predictor).to(device).eval(),
        "candidate": copy.deepcopy(candidate_predictor).to(device).eval(),
    }
    total_squared = {name: 0.0 for name in models}
    channel_squared = {
        name: torch.zeros(4, dtype=torch.float64) for name in models
    }
    symmetry_absolute = {name: 0.0 for name in models}
    scalar_count = 0
    vector_count = 0
    zero_squared = 0.0
    control_rows: list[Dict[str, object]] = []
    candidate_rows: list[Dict[str, object]] = []
    focused_records: list[Dict[str, object]] = []
    for _loader_step, (images_cpu, targets_cpu, metadata_cpu) in enumerate(loader):
        sample_index_tensor = metadata_cpu.get("sample_index")
        if not torch.is_tensor(sample_index_tensor):
            raise ValueError("Focused holdout metadata is missing sample_index.")
        sample_indices = [int(value) for value in sample_index_tensor.tolist()]
        first_cpu, second_cpu, target_cpu, color_records = make_color_pair_batch(
            images_cpu,
            sample_indices,
            phase="holdout",
            seed=int(args.seed),
            step=0,
            mean=mean,
            std=std,
        )
        first = first_cpu.to(device)
        second = second_cpu.to(device)
        color_target = target_cpu.to(device)
        batch_predictions: Dict[str, tuple[Tensor, Tensor]] = {}
        batch_probabilities: Dict[str, tuple[Tensor, Tensor]] = {}
        with torch.inference_mode():
            for name in ("control", "candidate"):
                logits_1, embedding_1 = _forward_augmented(
                    models[name], first, metadata_cpu, device=device
                )
                logits_2, embedding_2 = _forward_augmented(
                    models[name], second, metadata_cpu, device=device
                )
                prediction_12 = predictors[name](embedding_1, embedding_2)
                prediction_21 = predictors[name](embedding_2, embedding_1)
                batch_predictions[name] = (
                    prediction_12.detach().cpu(),
                    prediction_21.detach().cpu(),
                )
                batch_probabilities[name] = (
                    logits_1.float().softmax(dim=1).detach().cpu(),
                    logits_2.float().softmax(dim=1).detach().cpu(),
                )
                errors_12 = (prediction_12.float() - color_target.float()).square()
                errors_21 = (prediction_21.float() + color_target.float()).square()
                total_squared[name] += float(errors_12.sum().item() + errors_21.sum().item())
                channel_squared[name] += (
                    errors_12.sum(dim=0).double().cpu()
                    + errors_21.sum(dim=0).double().cpu()
                )
                symmetry_absolute[name] += float(
                    (prediction_12.float() + prediction_21.float()).abs().sum().item()
                )
        batch_size = int(first.size(0))
        scalar_count += 2 * batch_size * 4
        vector_count += 2 * batch_size
        zero_squared += 2.0 * float(target_cpu.double().square().sum().item())
        for position, (sample_index, target_value) in enumerate(
            zip(sample_indices, targets_cpu.tolist())
        ):
            control_rows.extend(
                (
                    _prediction_row(
                        sample_index=sample_index * 2,
                        target=int(target_value),
                        probabilities=batch_probabilities["control"][0][position],
                    ),
                    _prediction_row(
                        sample_index=sample_index * 2 + 1,
                        target=int(target_value),
                        probabilities=batch_probabilities["control"][1][position],
                    ),
                )
            )
            candidate_rows.extend(
                (
                    _prediction_row(
                        sample_index=sample_index * 2,
                        target=int(target_value),
                        probabilities=batch_probabilities["candidate"][0][position],
                    ),
                    _prediction_row(
                        sample_index=sample_index * 2 + 1,
                        target=int(target_value),
                        probabilities=batch_probabilities["candidate"][1][position],
                    ),
                )
            )
            focused_records.append(
                {
                    **color_records[position],
                    "target_class": int(target_value),
                    "control_prediction_12": batch_predictions["control"][0][position].tolist(),
                    "control_prediction_21": batch_predictions["control"][1][position].tolist(),
                    "candidate_prediction_12": batch_predictions["candidate"][0][position].tolist(),
                    "candidate_prediction_21": batch_predictions["candidate"][1][position].tolist(),
                }
            )
    zero_mse = zero_squared / max(1, scalar_count)
    variant_metrics: Dict[str, object] = {}
    for name in ("control", "candidate"):
        mse = total_squared[name] / max(1, scalar_count)
        variant_metrics[name] = {
            "mse": mse,
            "per_channel_mse": (
                channel_squared[name] / max(1, vector_count)
            ).tolist(),
            "explained_variance_vs_zero": 1.0 - mse / max(zero_mse, 1e-12),
            "ordered_pair_antisymmetry_mean_abs": symmetry_absolute[name]
            / max(1, scalar_count),
        }
    classification = _comparison(
        control_rows=control_rows,
        candidate_rows=candidate_rows,
        num_classes=5,
        focus_class=FOCUS_CLASS,
    )
    result = {
        "rows": len(focused_records),
        "ordered_pairs": scalar_count // 4,
        "zero_predictor_mse": zero_mse,
        "control": variant_metrics["control"],
        "candidate": variant_metrics["candidate"],
        "candidate_control_mse_ratio": float(variant_metrics["candidate"]["mse"])
        / max(float(variant_metrics["control"]["mse"]), 1e-12),
        "candidate_zero_mse_ratio": float(variant_metrics["candidate"]["mse"])
        / max(zero_mse, 1e-12),
        "two_view_classification": classification,
        "view_record_sha256": _canonical_sha256(
            [
                {
                    key: value
                    for key, value in record.items()
                    if not key.startswith("control_prediction_")
                    and not key.startswith("candidate_prediction_")
                    and key != "target_class"
                }
                for record in focused_records
            ]
        ),
        "all_target_reversals_exact": all(
            bool(record["reversal_exact"]) for record in focused_records
        ),
        "all_outputs_finite": all(
            math.isfinite(float(value))
            for name in ("control", "candidate")
            for value in (
                variant_metrics[name]["mse"],
                variant_metrics[name]["explained_variance_vs_zero"],
                *variant_metrics[name]["per_channel_mse"],
            )
        ),
        "loader": loader_summary,
    }
    del models, predictors
    gc.collect()
    torch.cuda.empty_cache()
    return result, focused_records, loader


def assess_stage_a(
    *,
    structural_checks: Mapping[str, bool],
    representation: Mapping[str, object],
    comparisons: Mapping[str, Mapping[str, Mapping[str, object]]],
) -> Dict[str, object]:
    candidate_representation = representation["candidate"]
    control_representation = representation["control"]
    pair_classification = representation["two_view_classification"]
    representation_checks = {
        "candidate_color_mse_lte_0p95_zero": float(candidate_representation["mse"])
        <= 0.95 * float(representation["zero_predictor_mse"]),
        "candidate_color_mse_lte_0p995_control": float(candidate_representation["mse"])
        <= 0.995 * float(control_representation["mse"]),
        "candidate_color_explained_variance_positive": float(
            candidate_representation["explained_variance_vs_zero"]
        )
        > 0.0,
        "candidate_two_view_accuracy_gte_control": float(
            pair_classification["candidate"]["accuracy"]
        )
        >= float(pair_classification["control"]["accuracy"]),
        "candidate_two_view_class1_f1_gte_control": float(
            pair_classification["candidate"]["per_class_f1"][FOCUS_CLASS]
        )
        >= float(pair_classification["control"]["per_class_f1"][FOCUS_CLASS]),
    }
    clean = comparisons["clean"]
    raw_candidate = clean["raw_candidate"]
    control_candidate = clean["control_candidate"]
    raw_delta = raw_candidate["delta"]
    control_delta = control_candidate["delta"]
    raw_transitions = raw_candidate["transitions"]
    control_transitions = control_candidate["transitions"]
    decision_checks = {
        "macro_f1_gain_vs_raw_gte_0p001": float(raw_delta["macro_f1"]) >= 0.001,
        "class1_f1_gain_vs_raw_gte_0p003": float(raw_delta["class1_f1"]) >= 0.003,
        "class1_precision_gain_vs_raw_gte_0p005": float(
            raw_delta["class1_precision"]
        )
        >= 0.005,
        "class1_recall_delta_vs_raw_gte_minus_0p005": float(
            raw_delta["class1_recall"]
        )
        >= -0.005,
        "zero_net_raw_class1_tp_break": int(raw_transitions["focus_tp_break"])
        <= int(raw_transitions["focus_fn_rescue"]),
        "restricted_fp_reduction_vs_raw_gte_2": int(
            raw_transitions["restricted_focus_fp_reduction"]
        )
        >= 2,
        "corrections_gt_harms_vs_raw": int(raw_transitions["candidate_correction"])
        > int(raw_transitions["candidate_harm"]),
        "maximum_nonfocus_f1_drop_vs_raw_lte_0p010": float(
            raw_candidate["maximum_nonfocus_f1_drop"]
        )
        <= 0.010,
        "new_3_to_2_harms_lte_2": int(
            raw_transitions["new_nonfocus_3_to_2_harms"]
        )
        <= 2,
        "macro_f1_delta_vs_control_nonnegative": float(control_delta["macro_f1"])
        >= 0.0,
        "class1_f1_gain_vs_control_gte_0p002": float(control_delta["class1_f1"])
        >= 0.002,
        "class1_precision_gain_vs_control_gte_0p003": float(
            control_delta["class1_precision"]
        )
        >= 0.003,
        "class1_recall_delta_vs_control_gte_minus_0p005": float(
            control_delta["class1_recall"]
        )
        >= -0.005,
        "restricted_fp_reduction_vs_control_gte_1": int(
            control_transitions["restricted_focus_fp_reduction"]
        )
        >= 1,
        "corrections_gt_harms_vs_control": int(
            control_transitions["candidate_correction"]
        )
        > int(control_transitions["candidate_harm"]),
    }
    lighting = [comparisons[name] for name, _, _ in LIGHTING_CONDITIONS]
    raw_macro = [float(row["raw_candidate"]["delta"]["macro_f1"]) for row in lighting]
    raw_precision = [
        float(row["raw_candidate"]["delta"]["class1_precision"]) for row in lighting
    ]
    control_macro = [
        float(row["control_candidate"]["delta"]["macro_f1"]) for row in lighting
    ]
    control_precision = [
        float(row["control_candidate"]["delta"]["class1_precision"])
        for row in lighting
    ]
    raw_recall = [
        float(row["raw_candidate"]["delta"]["class1_recall"]) for row in lighting
    ]
    control_recall = [
        float(row["control_candidate"]["delta"]["class1_recall"])
        for row in lighting
    ]
    illumination_checks = {
        "three_lighting_conditions_exact": len(lighting) == 3,
        "candidate_raw_macro_nonnegative_at_least_2_of_3": sum(
            value >= 0.0 for value in raw_macro
        )
        >= 2,
        "candidate_raw_precision_nonnegative_at_least_2_of_3": sum(
            value >= 0.0 for value in raw_precision
        )
        >= 2,
        "candidate_control_macro_nonnegative_at_least_2_of_3": sum(
            value >= 0.0 for value in control_macro
        )
        >= 2,
        "candidate_control_precision_nonnegative_at_least_2_of_3": sum(
            value >= 0.0 for value in control_precision
        )
        >= 2,
        "worst_class1_recall_delta_vs_raw_gte_minus_0p020": min(
            raw_recall, default=-math.inf
        )
        >= -0.020,
        "worst_class1_recall_delta_vs_control_gte_minus_0p020": min(
            control_recall, default=-math.inf
        )
        >= -0.020,
        "net_class1_tp_loss_lte_2_each_vs_raw_and_control": all(
            int(row[comparator]["transitions"]["focus_tp_break"])
            - int(row[comparator]["transitions"]["focus_fn_rescue"])
            <= 2
            for row in lighting
            for comparator in ("raw_candidate", "control_candidate")
        ),
        "aggregate_restricted_fp_removals_gt_creations_vs_raw": sum(
            int(row["raw_candidate"]["transitions"]["restricted_focus_fp_reduction"])
            for row in lighting
        )
        > 0,
        "aggregate_restricted_fp_removals_gt_creations_vs_control": sum(
            int(
                row["control_candidate"]["transitions"][
                    "restricted_focus_fp_reduction"
                ]
            )
            for row in lighting
        )
        > 0,
    }
    all_checks = {
        **{str(key): bool(value) for key, value in structural_checks.items()},
        **representation_checks,
        **decision_checks,
        **illumination_checks,
    }
    failed = [key for key, passed in all_checks.items() if not passed]
    return {
        "structural_checks": dict(structural_checks),
        "representation_checks": representation_checks,
        "decision_checks": decision_checks,
        "illumination_checks": illumination_checks,
        "failed_checks": failed,
        "all_gates_passed": not failed,
        "stage_b_authorized": not failed,
        "validation_authorized": not failed,
        "test_authorized": False,
        "full_train_authorized": False,
    }


def _benchmark_inference(
    *,
    prototype: nn.Module,
    images_cpu: Tensor,
    metadata_cpu: Mapping[str, object],
    device: torch.device,
    amp_dtype: torch.dtype,
    repeats: int,
) -> Dict[str, object]:
    set_seed(SEED, deterministic=True)
    gc.collect()
    torch.cuda.empty_cache()
    model = copy.deepcopy(prototype).to(device).eval()
    for parameter in model.parameters():
        parameter.requires_grad_(False)
    images = images_cpu.to(device)
    metadata = _metadata_to_device(
        metadata_cpu, device=device, count=int(images_cpu.size(0))
    )

    def iteration() -> Tensor:
        with torch.inference_mode(), torch.autocast(device_type="cuda", dtype=amp_dtype):
            return _forward_logits(model, images, metadata, device=device)

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
        "batch_size": int(images.size(0)),
        "mode": "deployment_inference_auxiliary_predictor_absent",
    }
    del model, images, logits
    gc.collect()
    torch.cuda.empty_cache()
    return result


def _export_diagnostics(
    *,
    candidate: nn.Module,
    images_cpu: Tensor,
    metadata_cpu: Mapping[str, object],
    output_dir: Path,
) -> Dict[str, object]:
    isolated_path = output_dir / "augself_color_adapter.onnx"
    embedding = torch.linspace(
        -1.0,
        1.0,
        steps=STATIC_EXPORT_BATCH_SIZE * EMBED_DIM,
        dtype=torch.float32,
    ).reshape(STATIC_EXPORT_BATCH_SIZE, EMBED_DIM)
    try:
        isolated = _onnx_compare(
            wrapper=_IsolatedAdapterExport(candidate.head.adapter),
            inputs=(embedding,),
            input_names=("embedding",),
            path=isolated_path,
        )
    except Exception as error:
        isolated = _failed_export(isolated_path, error)
    images = images_cpu[:STATIC_EXPORT_BATCH_SIZE].detach().float().cpu()
    bbox = metadata_cpu.get("bbox")
    if not torch.is_tensor(bbox):
        bbox = torch.zeros(STATIC_EXPORT_BATCH_SIZE, 8)
    bbox = bbox[:STATIC_EXPORT_BATCH_SIZE].detach().float().cpu()
    image_mask = metadata_cpu.get("image_mask")
    if not torch.is_tensor(image_mask):
        image_mask = torch.ones(
            STATIC_EXPORT_BATCH_SIZE,
            int(images.size(-2)),
            int(images.size(-1)),
            dtype=torch.bool,
        )
    image_mask = image_mask[:STATIC_EXPORT_BATCH_SIZE].detach().bool().cpu()
    full_path = output_dir / "augself_color_adapter_candidate.onnx"
    try:
        full = _onnx_compare(
            wrapper=_FullCandidateExport(candidate),
            inputs=(images, bbox, image_mask),
            input_names=("images", "bbox", "image_mask"),
            path=full_path,
        )
    except Exception as error:
        full = _failed_export(full_path, error)
    return {"isolated": isolated, "full": full, "predictor_exported": False}


def _required_xai_indices(
    raw_rows: Sequence[Mapping[str, object]],
    control_rows: Sequence[Mapping[str, object]],
    candidate_rows: Sequence[Mapping[str, object]],
    *,
    minimum_rows: int = 12,
) -> tuple[list[int], Dict[int, list[str]], list[int]]:
    if not (len(raw_rows) == len(control_rows) == len(candidate_rows)):
        raise ValueError("XAI comparator row counts differ.")
    categories: Dict[int, list[str]] = {}
    required: set[int] = set()
    changed_rank: list[tuple[float, int]] = []
    boundary_rank: list[tuple[float, int]] = []
    order: Dict[int, int] = {}
    for position, (raw, control, candidate) in enumerate(
        zip(raw_rows, control_rows, candidate_rows)
    ):
        sample_index = int(raw["sample_index"])
        order[sample_index] = position
        target = int(raw["target"])
        candidate_prediction = int(candidate["prediction"])
        labels: list[str] = []
        special = False
        for comparator_name, comparator in (("raw", raw), ("control", control)):
            comparator_prediction = int(comparator["prediction"])
            if (
                target == FOCUS_CLASS
                and comparator_prediction == FOCUS_CLASS
                and candidate_prediction != FOCUS_CLASS
            ):
                labels.append(f"class1_tp_break_vs_{comparator_name}")
                special = True
            if (
                target == FOCUS_CLASS
                and comparator_prediction != FOCUS_CLASS
                and candidate_prediction == FOCUS_CLASS
            ):
                labels.append(f"class1_tp_rescue_vs_{comparator_name}")
                special = True
            if target in RESTRICTED_NEGATIVE_CLASSES:
                if (
                    comparator_prediction == FOCUS_CLASS
                    and candidate_prediction != FOCUS_CLASS
                ):
                    labels.append(f"restricted_fp_remove_vs_{comparator_name}")
                    special = True
                if (
                    comparator_prediction != FOCUS_CLASS
                    and candidate_prediction == FOCUS_CLASS
                ):
                    labels.append(f"restricted_fp_create_vs_{comparator_name}")
                    special = True
            if comparator_prediction != target and candidate_prediction == target:
                labels.append(f"correction_vs_{comparator_name}")
            if comparator_prediction == target and candidate_prediction != target:
                labels.append(f"harm_vs_{comparator_name}")
        if special:
            required.add(sample_index)
        if labels:
            categories[sample_index] = list(dict.fromkeys(labels))
        raw_delta = abs(
            float(candidate[f"prob_{FOCUS_CLASS}"])
            - float(raw[f"prob_{FOCUS_CLASS}"])
        )
        control_delta = abs(
            float(candidate[f"prob_{FOCUS_CLASS}"])
            - float(control[f"prob_{FOCUS_CLASS}"])
        )
        rank = max(raw_delta, control_delta)
        if any(label.startswith(("correction_", "harm_")) for label in labels):
            changed_rank.append((-rank, sample_index))
        boundary_rank.append((-rank, sample_index))
    selected = set(required)
    for _negative_delta, sample_index in sorted(changed_rank):
        if len(selected) >= int(minimum_rows):
            break
        selected.add(sample_index)
    for _negative_delta, sample_index in sorted(boundary_rank):
        if len(selected) >= int(minimum_rows):
            break
        if sample_index not in selected:
            categories.setdefault(sample_index, []).append("diagnostic_boundary")
            selected.add(sample_index)
    selected_ordered = sorted(selected, key=order.__getitem__)
    required_ordered = sorted(required, key=order.__getitem__)
    return selected_ordered, categories, required_ordered


def _repeat_metadata(metadata: Mapping[str, object], count: int) -> Dict[str, object]:
    result: Dict[str, object] = {}
    for key, value in metadata.items():
        if torch.is_tensor(value):
            if int(value.size(0)) != 1:
                raise ValueError(f"XAI metadata {key} is not batch-one.")
            repeats = (int(count), *(1 for _ in range(value.ndim - 1)))
            result[key] = value.repeat(repeats)
        else:
            result[key] = value
    return result


def _color_loss_per_row(
    *,
    model: nn.Module,
    predictor: AugSelfColorPredictor,
    first: Tensor,
    second: Tensor,
    target: Tensor,
    metadata: Mapping[str, object],
    device: torch.device,
) -> Tensor:
    with torch.inference_mode():
        _logits_1, embedding_1 = _forward_augmented(
            model, first, metadata, device=device
        )
        _logits_2, embedding_2 = _forward_augmented(
            model, second, metadata, device=device
        )
        prediction_12 = predictor(embedding_1, embedding_2)
        prediction_21 = predictor(embedding_2, embedding_1)
        return 0.5 * (
            (prediction_12.float() - target.float()).square().mean(dim=1)
            + (prediction_21.float() + target.float()).square().mean(dim=1)
        )


def _array_sha256(value: np.ndarray) -> str:
    array = np.ascontiguousarray(value)
    digest = hashlib.sha256()
    digest.update(str(array.dtype).encode("ascii"))
    digest.update(json.dumps(list(array.shape)).encode("ascii"))
    digest.update(array.tobytes())
    return digest.hexdigest()


def _collect_color_xai(
    *,
    candidate: nn.Module,
    predictor: AugSelfColorPredictor,
    base_dataset: MangoYOLOCropDataset,
    transform,
    selected_indices: Sequence[int],
    mean: Sequence[float],
    std: Sequence[float],
    args: argparse.Namespace,
    device: torch.device,
) -> Dict[str, object]:
    loader, loader_summary = _make_loader(
        base_dataset=base_dataset,
        transform=transform,
        indices=selected_indices,
        batch_size=1,
        num_workers=int(args.num_workers),
        context="augself_color_loss_xai",
        seed=int(args.seed) + 760,
    )
    model = copy.deepcopy(candidate).to(device).eval()
    auxiliary = copy.deepcopy(predictor).to(device).eval()
    for parameter in model.parameters():
        parameter.requires_grad_(False)
    for parameter in auxiliary.parameters():
        parameter.requires_grad_(False)
    records: Dict[int, Dict[str, object]] = {}
    grid = XAI_OCCLUSION_GRID
    for images_cpu, _targets_cpu, metadata_cpu in loader:
        sample_index_tensor = metadata_cpu.get("sample_index")
        if not torch.is_tensor(sample_index_tensor):
            raise ValueError("Color XAI metadata is missing sample_index.")
        sample_index = int(sample_index_tensor.item())
        first_cpu, second_cpu, target_cpu, color_records = make_color_pair_batch(
            images_cpu,
            [sample_index],
            phase="xai",
            seed=int(args.seed),
            step=0,
            mean=mean,
            std=std,
        )
        first = first_cpu.to(device)
        second = second_cpu.to(device)
        target = target_cpu.to(device)
        baseline = _color_loss_per_row(
            model=model,
            predictor=auxiliary,
            first=first,
            second=second,
            target=target,
            metadata=metadata_cpu,
            device=device,
        )[0]
        occluded_first = first.repeat(grid * grid, 1, 1, 1)
        occluded_second = second.repeat(grid * grid, 1, 1, 1)
        height, width = int(first.size(-2)), int(first.size(-1))
        for cell in range(grid * grid):
            row = cell // grid
            column = cell % grid
            y0 = row * height // grid
            y1 = (row + 1) * height // grid
            x0 = column * width // grid
            x1 = (column + 1) * width // grid
            occluded_first[cell, :, y0:y1, x0:x1] = 0.0
            occluded_second[cell, :, y0:y1, x0:x1] = 0.0
        repeated_metadata = _repeat_metadata(metadata_cpu, grid * grid)
        repeated_target = target.repeat(grid * grid, 1)
        occluded_loss = _color_loss_per_row(
            model=model,
            predictor=auxiliary,
            first=occluded_first,
            second=occluded_second,
            target=repeated_target,
            metadata=repeated_metadata,
            device=device,
        )
        coarse = (occluded_loss - baseline).abs().reshape(1, 1, grid, grid)
        saliency = F.interpolate(
            coarse,
            size=(height, width),
            mode="bilinear",
            align_corners=False,
        )[0, 0]
        minimum = saliency.amin()
        maximum = saliency.amax()
        normalized = (
            (saliency - minimum) / (maximum - minimum).clamp_min(1e-12)
        ).detach().cpu().numpy()
        first_rgb = _rgb_from_tensor(first_cpu[0], mean=mean, std=std)
        second_rgb = _rgb_from_tensor(second_cpu[0], mean=mean, std=std)
        difference_rgb = np.abs(
            first_rgb.astype(np.int16) - second_rgb.astype(np.int16)
        ).clip(0, 255).astype(np.uint8)
        records[sample_index] = {
            "view_1_rgb": first_rgb,
            "view_2_rgb": second_rgb,
            "absolute_difference_rgb": difference_rgb,
            "color_loss_saliency": normalized,
            "baseline_color_loss": float(baseline.item()),
            "maximum_occlusion_delta": float((occluded_loss - baseline).abs().amax().item()),
            "color_record": color_records[0],
        }
    del model, auxiliary
    gc.collect()
    torch.cuda.empty_cache()
    if list(records) != list(selected_indices):
        raise ValueError("Color XAI row order differs from selected indices.")
    finite = all(
        np.isfinite(np.asarray(row["color_loss_saliency"])).all()
        and np.isfinite(np.asarray(row["view_1_rgb"])).all()
        and np.isfinite(np.asarray(row["view_2_rgb"])).all()
        for row in records.values()
    )
    nonzero = any(
        float(row["maximum_occlusion_delta"]) > 0.0 for row in records.values()
    )
    return {
        "records": records,
        "loader": loader_summary,
        "finite": bool(finite),
        "nonzero": bool(nonzero),
        "method": f"paired-input {grid}x{grid} neutral-mean occlusion",
    }


def _render_xai_pages(
    *,
    output_dir: Path,
    selected_indices: Sequence[int],
    categories: Mapping[int, Sequence[str]],
    rows: Sequence[CleanTrainRow],
    raw_clean: Sequence[Mapping[str, object]],
    control_clean: Sequence[Mapping[str, object]],
    candidate_clean: Sequence[Mapping[str, object]],
    raw_maps: Mapping[int, Mapping[str, object]],
    control_maps: Mapping[int, Mapping[str, object]],
    candidate_maps: Mapping[int, Mapping[str, object]],
    color_maps: Mapping[int, Mapping[str, object]],
) -> list[str]:
    raw_by_index = {int(row["sample_index"]): row for row in raw_clean}
    control_by_index = {int(row["sample_index"]): row for row in control_clean}
    candidate_by_index = {int(row["sample_index"]): row for row in candidate_clean}
    columns = (
        "input",
        "raw Grad-CAM",
        "control Grad-CAM",
        "candidate Grad-CAM",
        "color view 1",
        "color view 2",
        "absolute difference",
        "color-loss saliency",
    )
    tile = 144
    header = 28
    label_height = 50
    page_size = 12
    pages = []
    for page_number, start in enumerate(
        range(0, len(selected_indices), page_size), start=1
    ):
        page_indices = list(selected_indices[start : start + page_size])
        canvas = Image.new(
            "RGB",
            (len(columns) * tile, header + len(page_indices) * (tile + label_height)),
            "white",
        )
        draw = ImageDraw.Draw(canvas)
        for column, title in enumerate(columns):
            draw.text((column * tile + 4, 7), title, fill="black")
        for row_position, sample_index in enumerate(page_indices):
            y = header + row_position * (tile + label_height)
            rgb = np.asarray(raw_maps[sample_index]["rgb"], dtype=np.uint8)
            color = color_maps[sample_index]
            views = [
                Image.fromarray(rgb),
                _heat_overlay(rgb, np.asarray(raw_maps[sample_index]["gradcam"])),
                _heat_overlay(rgb, np.asarray(control_maps[sample_index]["gradcam"])),
                _heat_overlay(rgb, np.asarray(candidate_maps[sample_index]["gradcam"])),
                Image.fromarray(np.asarray(color["view_1_rgb"], dtype=np.uint8)),
                Image.fromarray(np.asarray(color["view_2_rgb"], dtype=np.uint8)),
                Image.fromarray(
                    np.asarray(color["absolute_difference_rgb"], dtype=np.uint8)
                ),
                _heat_overlay(rgb, np.asarray(color["color_loss_saliency"])),
            ]
            for column, view in enumerate(views):
                canvas.paste(
                    view.resize((tile, tile), Image.Resampling.BILINEAR),
                    (column * tile, y),
                )
            source = rows[sample_index]
            raw_row = raw_by_index[sample_index]
            control_row = control_by_index[sample_index]
            candidate_row = candidate_by_index[sample_index]
            line_one = (
                f"idx={sample_index} y={source.target} raw/ctl/cand="
                f"{raw_row['prediction']}/{control_row['prediction']}/"
                f"{candidate_row['prediction']}"
            )
            line_two = ",".join(categories.get(sample_index, ("diagnostic",)))
            draw.text((4, y + tile + 3), line_one, fill="black")
            draw.text((4, y + tile + 23), line_two[:180], fill="black")
        path = output_dir / f"augself_xai_contact_sheet_{page_number:02d}.png"
        canvas.save(path)
        pages.append(str(path.resolve()))
    return pages


def _xai_audit(
    *,
    output_dir: Path,
    raw: nn.Module,
    control: nn.Module,
    candidate: nn.Module,
    candidate_predictor: AugSelfColorPredictor,
    base_dataset: MangoYOLOCropDataset,
    transform,
    rows: Sequence[CleanTrainRow],
    raw_conditions: Mapping[str, Sequence[Mapping[str, object]]],
    control_conditions: Mapping[str, Sequence[Mapping[str, object]]],
    candidate_conditions: Mapping[str, Sequence[Mapping[str, object]]],
    mean: Sequence[float],
    std: Sequence[float],
    args: argparse.Namespace,
    device: torch.device,
) -> Dict[str, object]:
    selected, categories, required = _required_xai_indices(
        raw_conditions["clean"],
        control_conditions["clean"],
        candidate_conditions["clean"],
    )
    replay_selected, replay_categories, replay_required = _required_xai_indices(
        raw_conditions["clean"],
        control_conditions["clean"],
        candidate_conditions["clean"],
    )
    xai_device = torch.device("cpu")
    raw_result = _collect_xai_maps(
        name="raw",
        prototype=raw,
        base_dataset=base_dataset,
        transform=transform,
        selected_indices=selected,
        args=args,
        device=xai_device,
        mean=mean,
        std=std,
        collect_groups=False,
    )
    control_result = _collect_xai_maps(
        name="control",
        prototype=control,
        base_dataset=base_dataset,
        transform=transform,
        selected_indices=selected,
        args=args,
        device=xai_device,
        mean=mean,
        std=std,
        collect_groups=False,
    )
    candidate_result = _collect_xai_maps(
        name="candidate",
        prototype=candidate,
        base_dataset=base_dataset,
        transform=transform,
        selected_indices=selected,
        args=args,
        device=xai_device,
        mean=mean,
        std=std,
        collect_groups=False,
    )
    color_result = _collect_color_xai(
        candidate=candidate,
        predictor=candidate_predictor,
        base_dataset=base_dataset,
        transform=transform,
        selected_indices=selected,
        mean=mean,
        std=std,
        args=args,
        device=device,
    )
    maps_path = output_dir / "augself_xai_maps.npz"
    arrays: Dict[str, np.ndarray] = {}
    map_hashes: Dict[str, Dict[str, str]] = {}
    for sample_index in selected:
        sample_hashes = {}
        for name, value in (
            ("raw_gradcam", raw_result["records"][sample_index]["gradcam"]),
            ("control_gradcam", control_result["records"][sample_index]["gradcam"]),
            ("candidate_gradcam", candidate_result["records"][sample_index]["gradcam"]),
            ("view_1_rgb", color_result["records"][sample_index]["view_1_rgb"]),
            ("view_2_rgb", color_result["records"][sample_index]["view_2_rgb"]),
            (
                "absolute_difference_rgb",
                color_result["records"][sample_index]["absolute_difference_rgb"],
            ),
            (
                "color_loss_saliency",
                color_result["records"][sample_index]["color_loss_saliency"],
            ),
        ):
            key = f"sample_{sample_index}_{name}"
            array = np.asarray(value)
            arrays[key] = array
            sample_hashes[name] = _array_sha256(array)
        map_hashes[str(sample_index)] = sample_hashes
    np.savez_compressed(maps_path, **arrays)
    pages = _render_xai_pages(
        output_dir=output_dir,
        selected_indices=selected,
        categories=categories,
        rows=rows,
        raw_clean=raw_conditions["clean"],
        control_clean=control_conditions["clean"],
        candidate_clean=candidate_conditions["clean"],
        raw_maps=raw_result["records"],
        control_maps=control_result["records"],
        candidate_maps=candidate_result["records"],
        color_maps=color_result["records"],
    )
    color_records = {
        str(index): color_result["records"][index]["color_record"] for index in selected
    }
    manifest = {
        "selected_indices": selected,
        "categories": {str(key): list(value) for key, value in categories.items()},
        "required_indices": required,
        "required_coverage_exact": set(required).issubset(set(selected)),
        "selection_replay_exact": selected == replay_selected
        and required == replay_required
        and categories == replay_categories,
        "minimum_twelve_when_available": len(selected) >= min(12, len(raw_conditions["clean"])),
        "raw_maps_finite": bool(raw_result["finite"]),
        "control_maps_finite": bool(control_result["finite"]),
        "candidate_maps_finite": bool(candidate_result["finite"]),
        "color_maps_finite": bool(color_result["finite"]),
        "color_loss_saliency_nonzero": bool(color_result["nonzero"]),
        "color_loss_saliency_method": color_result["method"],
        "color_view_record_sha256": _canonical_sha256(color_records),
        "map_hashes": map_hashes,
        "maps_npz": str(maps_path.resolve()),
        "maps_npz_sha256": _sha256(maps_path),
        "page_count": len(pages),
        "pages": pages,
        "loaders": {
            "raw": raw_result["loader"],
            "control": control_result["loader"],
            "candidate": candidate_result["loader"],
            "color": color_result["loader"],
        },
        "target_class": FOCUS_CLASS,
        "gradcam_source": "stem_output",
        "device": "cpu_fp32_gradcam_and_cuda_fp32_occlusion",
    }
    manifest_path = output_dir / "augself_xai_manifest.json"
    manifest_path.write_text(
        json.dumps(manifest, indent=2, sort_keys=True, ensure_ascii=True) + "\n",
        encoding="utf-8",
    )
    return {**manifest, "manifest": str(manifest_path.resolve())}


def _write_training_curve(
    path: Path,
    control: Mapping[str, object],
    candidate: Mapping[str, object],
) -> None:
    rows = [*control["history"], *candidate["history"]]
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(
            handle,
            fieldnames=(
                "variant",
                "step",
                "loss",
                "classification_loss",
                "color_loss",
                "gradient_norm",
                "prediction_12_mean_abs",
                "prediction_21_mean_abs",
            ),
        )
        writer.writeheader()
        writer.writerows(rows)


def _write_report(path: Path, summary: Mapping[str, object]) -> None:
    clean = summary["comparisons"]["clean"]
    raw = clean["raw_candidate"]
    control = clean["control_candidate"]
    representation = summary["representation"]
    gate = summary["gate"]
    lines = [
        "# AugSelf Color-Adapter A0 Result",
        "",
        f"- Status: `{summary['status']}`",
        f"- Stage B authorized: `{gate['stage_b_authorized']}`",
        f"- Failed checks: `{', '.join(gate['failed_checks']) or 'none'}`",
        f"- Candidate/control/zero color MSE: `{representation['candidate']['mse']:.6f}/{representation['control']['mse']:.6f}/{representation['zero_predictor_mse']:.6f}`",
        f"- Candidate explained variance: `{representation['candidate']['explained_variance_vs_zero']:+.6f}`",
        f"- Clean macro/class1 F1 delta vs raw: `{raw['delta']['macro_f1']:+.6f}/{raw['delta']['class1_f1']:+.6f}`",
        f"- Clean class1 precision/recall delta vs raw: `{raw['delta']['class1_precision']:+.6f}/{raw['delta']['class1_recall']:+.6f}`",
        f"- Clean class1 F1/precision delta vs detached control: `{control['delta']['class1_f1']:+.6f}/{control['delta']['class1_precision']:+.6f}`",
        f"- Restricted FP reduction / TP breaks vs raw: `{raw['transitions']['restricted_focus_fp_reduction']}/{raw['transitions']['focus_tp_break']}`",
        f"- Candidate runtime ratio / peak VRAM GiB: `{summary['resources']['runtime_ratio']:.6f}/{summary['resources']['candidate_benchmark']['peak_vram_gib']:.6f}`",
        f"- XAI rows/pages: `{len(summary['xai']['selected_indices'])}/{summary['xai']['page_count']}`",
        "",
        "Only source-disjoint yolo_f/train rows were used. Validation and test were not accessed.",
    ]
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def _write_manifest(output_dir: Path) -> Dict[str, object]:
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
        "method": METHOD,
        "artifact_count": len(artifacts),
        "total_bytes": sum(int(row["bytes"]) for row in artifacts),
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


def run_audit(args: argparse.Namespace) -> Dict[str, object]:
    provenance, rows, cohorts, prior = _load_locked_inputs(args)
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
        raise RuntimeError("Locked AugSelf A0 requires CUDA.")
    paths = _source_paths(args)
    output_path = Path(args.output_dir).resolve()
    raw_root = paths["data"].parent.parent.resolve()
    try:
        output_path.relative_to(raw_root)
    except ValueError:
        pass
    else:
        raise ValueError("AugSelf output cannot be written under the raw dataset tree.")
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
    (
        raw,
        control,
        candidate,
        control_predictor,
        candidate_predictor,
        construction,
    ) = _construct_models(checkpoint)
    dataset, transform, dataset_summary = _build_dataset(
        checkpoint, rows, paths["data"]
    )
    mean, std = checkpoint_input_normalization(checkpoint)
    train_indices = balanced_boundary_order(
        cohorts["reference_indices"],
        cohorts["hard_indices"],
        steps=TRAIN_STEPS,
        half_batch=HALF_BATCH,
        seed=SEED,
    )
    train_order_sha = _ordered_index_sha256(train_indices)
    if train_order_sha != LOCKED_TRAIN_ORDER_SHA256:
        raise ValueError(
            f"AugSelf train order differs: {train_order_sha} != {LOCKED_TRAIN_ORDER_SHA256}"
        )
    reference_set = set(int(value) for value in cohorts["reference_indices"])
    hard_set = set(int(value) for value in cohorts["hard_indices"])
    balanced_batches_exact = all(
        sum(index in reference_set for index in train_indices[start : start + BATCH_SIZE])
        == HALF_BATCH
        and sum(index in hard_set for index in train_indices[start : start + BATCH_SIZE])
        == HALF_BATCH
        for start in range(0, len(train_indices), BATCH_SIZE)
    )

    resource_loader, resource_loader_summary = _make_loader(
        base_dataset=dataset,
        transform=transform,
        indices=cohorts["holdout_indices"][:BATCH_SIZE],
        batch_size=BATCH_SIZE,
        num_workers=int(args.num_workers),
        context="augself_resource_batch",
        seed=SEED + 400,
    )
    images_cpu, targets_cpu, metadata_cpu = next(iter(resource_loader))
    initial = _initial_exactness(
        raw=raw,
        control=control,
        candidate=candidate,
        images_cpu=images_cpu,
        metadata_cpu=metadata_cpu,
        prior_clean=prior["clean"]["raw"],
        device=device,
    )
    gradient_loader, gradient_loader_summary = _make_loader(
        base_dataset=dataset,
        transform=transform,
        indices=train_indices[:BATCH_SIZE],
        batch_size=BATCH_SIZE,
        num_workers=int(args.num_workers),
        context="augself_initial_gradient_batch",
        seed=SEED + 410,
    )
    gradient_images, _gradient_targets, gradient_metadata = next(iter(gradient_loader))
    gradient_indices_tensor = gradient_metadata.get("sample_index")
    if not torch.is_tensor(gradient_indices_tensor):
        raise ValueError("Initial gradient batch is missing sample_index.")
    gradient_audit = _initial_gradient_audit(
        control=control,
        candidate=candidate,
        control_predictor=control_predictor,
        candidate_predictor=candidate_predictor,
        images_cpu=gradient_images,
        metadata_cpu=gradient_metadata,
        sample_indices=[int(value) for value in gradient_indices_tensor.tolist()],
        mean=mean,
        std=std,
        device=device,
    )

    control_model, trained_control_predictor, control_training, control_views = (
        _train_variant(
            name="detached_control",
            model_prototype=control,
            predictor_prototype=control_predictor,
            detach_embeddings=True,
            base_dataset=dataset,
            transform=transform,
            train_indices=train_indices,
            mean=mean,
            std=std,
            args=args,
            device=device,
        )
    )
    candidate_model, trained_candidate_predictor, candidate_training, candidate_views = (
        _train_variant(
            name="augself_candidate",
            model_prototype=candidate,
            predictor_prototype=candidate_predictor,
            detach_embeddings=False,
            base_dataset=dataset,
            transform=transform,
            train_indices=train_indices,
            mean=mean,
            std=std,
            args=args,
            device=device,
        )
    )
    train_views_bit_exact = control_views == candidate_views
    train_replay_loader, train_replay_loader_summary = _make_loader(
        base_dataset=dataset,
        transform=transform,
        indices=train_indices,
        batch_size=BATCH_SIZE,
        num_workers=int(args.num_workers),
        context="augself_train_view_replay",
        seed=SEED + 420,
    )
    train_view_replay = _replay_color_loader(
        loader=train_replay_loader,
        expected_records=candidate_views,
        phase="train",
        seed=SEED,
        mean=mean,
        std=std,
    )

    focused_indices, focused_cohort = _focused_holdout_indices(
        rows=rows,
        holdout_indices=cohorts["holdout_indices"],
        raw_clean=prior["clean"]["raw"],
    )
    representation, focused_records, focused_loader = _representation_evaluation(
        control=control_model,
        candidate=candidate_model,
        control_predictor=trained_control_predictor,
        candidate_predictor=trained_candidate_predictor,
        base_dataset=dataset,
        transform=transform,
        focused_indices=focused_indices,
        mean=mean,
        std=std,
        args=args,
        device=device,
    )
    focused_color_records = [
        {
            key: value
            for key, value in record.items()
            if not key.startswith("control_prediction_")
            and not key.startswith("candidate_prediction_")
            and key != "target_class"
        }
        for record in focused_records
    ]
    focused_view_replay = _replay_color_loader(
        loader=focused_loader,
        expected_records=focused_color_records,
        phase="holdout",
        seed=SEED,
        mean=mean,
        std=std,
        constant_step=0,
    )

    control_conditions, candidate_conditions, paired_loaders = _predict_paired_conditions(
        control_prototype=control_model,
        candidate_prototype=candidate_model,
        base_dataset=dataset,
        transform=transform,
        holdout_indices=cohorts["holdout_indices"],
        args=args,
        device=device,
        amp_dtype=amp_dtype,
    )
    raw_conditions = {condition: prior[condition]["raw"] for condition in CONDITIONS}
    comparisons = _build_comparisons(
        raw_conditions=raw_conditions,
        control_conditions=control_conditions,
        candidate_conditions=candidate_conditions,
    )

    raw_benchmark = _benchmark_inference(
        prototype=raw,
        images_cpu=images_cpu,
        metadata_cpu=metadata_cpu,
        device=device,
        amp_dtype=amp_dtype,
        repeats=int(args.benchmark_repeats),
    )
    candidate_benchmark = _benchmark_inference(
        prototype=candidate_model,
        images_cpu=images_cpu,
        metadata_cpu=metadata_cpu,
        device=device,
        amp_dtype=amp_dtype,
        repeats=int(args.benchmark_repeats),
    )
    runtime_ratio = float(
        candidate_benchmark["median_seconds"]
        / max(float(raw_benchmark["median_seconds"]), 1e-12)
    )
    export = _export_diagnostics(
        candidate=candidate_model,
        images_cpu=images_cpu,
        metadata_cpu=metadata_cpu,
        output_dir=output_dir,
    )
    xai = _xai_audit(
        output_dir=output_dir,
        raw=raw,
        control=control_model,
        candidate=candidate_model,
        candidate_predictor=trained_candidate_predictor,
        base_dataset=dataset,
        transform=transform,
        rows=rows,
        raw_conditions=raw_conditions,
        control_conditions=control_conditions,
        candidate_conditions=candidate_conditions,
        mean=mean,
        std=std,
        args=args,
        device=device,
    )

    structural_checks = {
        "locked_sources_verified": True,
        "tracked_worktree_clean": bool(provenance["tracked_worktree_clean"]),
        "official_worktree_clean": bool(provenance["official_worktree_clean"]),
        "official_license_absent": bool(provenance["official_license_absent"]),
        "train_only_dataset_contract_exact": bool(
            dataset_summary["paths_exact"] and dataset_summary["train_paths_only"]
        ),
        "source_disjoint_7372_1843_exact": len(cohorts["fit_indices"]) == 7372
        and len(cohorts["holdout_indices"]) == EXPECTED_HOLDOUT_ROWS
        and not cohorts["source_overlap"],
        "fit_cohort_432_186_exact": len(cohorts["reference_indices"])
        == EXPECTED_REFERENCE_ROWS
        and len(cohorts["hard_indices"]) == EXPECTED_HARD_ROWS,
        "balanced_60x32_order_exact": len(train_indices) == EXPECTED_TRAIN_ROWS
        and balanced_batches_exact
        and train_order_sha == LOCKED_TRAIN_ORDER_SHA256,
        "focused_holdout_109_36_exact": int(focused_cohort["class1_rows"]) == 109
        and int(focused_cohort["restricted_raw_fp_rows"]) == 36
        and int(focused_cohort["rows"]) == 145,
        "keeper_parameter_count_exact": int(construction["raw_parameters"])
        == EXPECTED_KEEPER_PARAMETERS,
        "adapter_parameter_count_exact": int(construction["adapter_parameters"])
        == EXPECTED_ADAPTER_PARAMETERS,
        "predictor_parameter_count_exact": int(construction["predictor_parameters"])
        == EXPECTED_PREDICTOR_PARAMETERS,
        "candidate_control_initial_model_bit_exact": bool(
            construction["candidate_control_model_bit_exact"]
        ),
        "candidate_control_initial_predictor_bit_exact": bool(
            construction["candidate_control_predictor_bit_exact"]
        ),
        "adapter_up_zero_exact": bool(construction["adapter_up_zero_exact"]),
        "initial_deployed_logits_bit_exact": bool(
            initial["raw_control_logits_bit_exact"]
            and initial["raw_candidate_logits_bit_exact"]
            and initial["control_candidate_logits_bit_exact"]
        ),
        "initial_raw_prior_argmax_match": bool(initial["raw_prior_argmax_match"]),
        "isolated_candidate_aux_adapter_gradient_nonzero": bool(
            gradient_audit["candidate_adapter_aux_gradient_nonzero"]
        ),
        "isolated_control_aux_adapter_gradient_zero": bool(
            gradient_audit["control_adapter_aux_gradient_exact_zero"]
        ),
        "isolated_predictor_gradients_bit_exact": bool(
            gradient_audit["predictor_gradients_bit_exact"]
        ),
        "candidate_control_train_views_bit_exact": train_views_bit_exact
        and control_training["view_record_sha256"]
        == candidate_training["view_record_sha256"],
        "train_view_independent_replay_exact": bool(
            train_view_replay["records_exact"] and train_view_replay["sha256_exact"]
        ),
        "focused_view_independent_replay_exact": bool(
            focused_view_replay["records_exact"]
            and focused_view_replay["sha256_exact"]
        ),
        "both_execute_60_steps_1920_rows": int(control_training["steps"])
        == int(candidate_training["steps"])
        == TRAIN_STEPS
        and int(control_training["rows"])
        == int(candidate_training["rows"])
        == EXPECTED_TRAIN_ROWS,
        "both_optimizer_initial_states_empty": bool(
            control_training["optimizer_initial_state_empty"]
            and candidate_training["optimizer_initial_state_empty"]
        ),
        "both_all_gradients_finite_nonzero": bool(
            control_training["all_gradients_finite"]
            and candidate_training["all_gradients_finite"]
            and control_training["all_trainable_gradients_seen"]
            and candidate_training["all_trainable_gradients_seen"]
        ),
        "both_all_trainable_parameters_changed": bool(
            control_training["all_trainable_parameters_changed"]
            and candidate_training["all_trainable_parameters_changed"]
        ),
        "both_frozen_states_bit_exact": bool(
            control_training["frozen_state_bit_exact"]
            and candidate_training["frozen_state_bit_exact"]
        ),
        "representation_outputs_finite": bool(representation["all_outputs_finite"]),
        "isolated_onnx_finite_error_lte_1e5_argmax_exact": bool(
            export["isolated"]["succeeded"]
            and export["isolated"]["finite"]
            and export["isolated"]["argmax_match"]
            and float(export["isolated"]["maximum_absolute_error"])
            <= MAX_ONNX_ERROR
        ),
        "full_onnx_finite_error_lte_1e5_argmax_exact": bool(
            export["full"]["succeeded"]
            and export["full"]["finite"]
            and export["full"]["argmax_match"]
            and float(export["full"]["maximum_absolute_error"]) <= MAX_ONNX_ERROR
        ),
        "auxiliary_predictor_not_exported": not bool(export["predictor_exported"]),
        "runtime_ratio_lte_1p10": runtime_ratio <= MAX_RUNTIME_RATIO,
        "candidate_peak_vram_lte_0p75_gib": float(
            candidate_benchmark["peak_vram_gib"]
        )
        <= MAX_PEAK_VRAM_GIB,
        "xai_required_coverage_exact": bool(
            xai["required_coverage_exact"] and xai["selection_replay_exact"]
        ),
        "xai_minimum_twelve_when_available": bool(
            xai["minimum_twelve_when_available"]
        ),
        "xai_all_maps_finite": bool(
            xai["raw_maps_finite"]
            and xai["control_maps_finite"]
            and xai["candidate_maps_finite"]
            and xai["color_maps_finite"]
        ),
        "xai_color_loss_saliency_nonzero": bool(
            xai["color_loss_saliency_nonzero"]
        ),
        "current_best_commands_hash_unchanged": provenance["sha256"][
            "current_commands"
        ]
        == LOCKED_CURRENT_COMMAND_SHA256,
        "command_history_hash_unchanged": provenance["sha256"]["command_history"]
        == LOCKED_COMMAND_HISTORY_SHA256,
        "validation_not_used": not bool(provenance["validation_predictions_used"]),
        "test_not_used": not bool(provenance["test_data_used"]),
    }
    gate = assess_stage_a(
        structural_checks=structural_checks,
        representation=representation,
        comparisons=comparisons,
    )

    predictions_path = output_dir / "predictions_all_conditions.csv"
    training_path = output_dir / "training_curve.csv"
    color_replay_path = output_dir / "focused_color_pair_replay.json"
    independent_replay_path = output_dir / "independent_replay.json"
    summary_path = output_dir / "summary.json"
    report_path = output_dir / "report.md"
    manifest_path = output_dir / "artifact_manifest.json"
    _write_predictions(
        predictions_path,
        rows=rows,
        raw_conditions=raw_conditions,
        control_conditions=control_conditions,
        candidate_conditions=candidate_conditions,
    )
    _write_training_curve(training_path, control_training, candidate_training)
    prediction_replay = _replay_predictions(predictions_path, expected=comparisons)
    if not bool(prediction_replay["comparisons_exact"]):
        raise RuntimeError("Independent prediction replay differs from in-memory results.")
    independent_replay = {
        "predictions": {
            key: value
            for key, value in prediction_replay.items()
            if key != "replayed_comparisons"
        },
        "train_color_views": train_view_replay,
        "focused_color_views": focused_view_replay,
        "xai_selection_exact": bool(xai["selection_replay_exact"]),
    }
    independent_replay_path.write_text(
        json.dumps(independent_replay, indent=2, sort_keys=True, ensure_ascii=True)
        + "\n",
        encoding="utf-8",
    )
    color_replay_payload = {
        "method": METHOD,
        "train_records": candidate_views,
        "focused_records": focused_records,
        "control_candidate_train_records_bit_exact": train_views_bit_exact,
        "train_replay": train_view_replay,
        "focused_replay": focused_view_replay,
    }
    color_replay_path.write_text(
        json.dumps(color_replay_payload, indent=2, sort_keys=True, ensure_ascii=True)
        + "\n",
        encoding="utf-8",
    )

    summary: Dict[str, object] = {
        "status": "authorized" if gate["stage_b_authorized"] else "rejected",
        "method": METHOD,
        "protocol_stage": "A_train_only_matched_detached_gradient_control",
        "protocol": {
            "seed": SEED,
            "fold": FOLD,
            "batch_size": BATCH_SIZE,
            "half_batch": HALF_BATCH,
            "train_steps": TRAIN_STEPS,
            "train_rows": len(train_indices),
            "learning_rate": LEARNING_RATE,
            "weight_decay": WEIGHT_DECAY,
            "auxiliary_weight": AUXILIARY_WEIGHT,
            "jitter_probability": JITTER_PROBABILITY,
            "jitter_factors": [JITTER_FACTOR_MIN, JITTER_FACTOR_MAX],
            "jitter_hue": [JITTER_HUE_MIN, JITTER_HUE_MAX],
            "selection": "single_final_step_60_no_sweep",
        },
        "provenance": provenance,
        "runtime": {
            "torch_version": torch.__version__,
            "cuda_version": torch.version.cuda,
            "device_name": torch.cuda.get_device_name(device),
            "training_dtype": "float32",
            "benchmark_amp_dtype": str(amp_dtype).replace("torch.", ""),
            "cuda_matmul_allow_tf32": torch.backends.cuda.matmul.allow_tf32,
            "cudnn_allow_tf32": torch.backends.cudnn.allow_tf32,
            "cudnn_deterministic": torch.backends.cudnn.deterministic,
            "cudnn_benchmark": torch.backends.cudnn.benchmark,
            "deterministic_algorithms": torch.are_deterministic_algorithms_enabled(),
        },
        "cohort": {
            **_cohort_serializable(cohorts),
            "balanced_train_order_sha256": train_order_sha,
            "balanced_batches_exact": balanced_batches_exact,
            "focused_holdout": focused_cohort,
        },
        "dataset": dataset_summary,
        "construction": construction,
        "initial_exactness": initial,
        "gradient_isolation": gradient_audit,
        "training": {
            "control": control_training,
            "candidate": candidate_training,
            "train_views_bit_exact": train_views_bit_exact,
        },
        "representation": representation,
        "comparisons": comparisons,
        "resources": {
            "raw_benchmark": raw_benchmark,
            "candidate_benchmark": candidate_benchmark,
            "runtime_ratio": runtime_ratio,
            "export": export,
        },
        "xai": xai,
        "gate": gate,
        "replay": independent_replay,
        "loaders": {
            "resource": resource_loader_summary,
            "initial_gradient": gradient_loader_summary,
            "train_replay": train_replay_loader_summary,
            "paired_conditions": paired_loaders,
        },
        "validation_predictions_used": False,
        "test_data_used": False,
        "raw_dataset_touched": False,
        "shared_model_code_touched": False,
        "shared_trainer_code_touched": False,
        "trainable_checkpoint_written": False,
        "binary_model_artifacts_written": False,
        "onnx_export_written": True,
        "current_best_command_revision_created": False,
        "artifacts": {
            "summary": str(summary_path.resolve()),
            "report": str(report_path.resolve()),
            "predictions": str(predictions_path.resolve()),
            "training_curve": str(training_path.resolve()),
            "focused_color_pair_replay": str(color_replay_path.resolve()),
            "independent_replay": str(independent_replay_path.resolve()),
            "xai_manifest": xai["manifest"],
            "manifest": str(manifest_path.resolve()),
        },
    }
    summary_path.write_text(
        json.dumps(summary, indent=2, sort_keys=True, ensure_ascii=True) + "\n",
        encoding="utf-8",
    )
    _write_report(report_path, summary)
    _write_manifest(output_dir)
    return summary


def main(argv: Optional[Sequence[str]] = None) -> None:
    args = parse_args(argv)
    result = run_audit(args)
    if bool(args.preflight_only):
        print(json.dumps(result, indent=2, sort_keys=True, ensure_ascii=True))
        return
    print(json.dumps(result["gate"], indent=2, sort_keys=True, ensure_ascii=True))


if __name__ == "__main__":
    main()
