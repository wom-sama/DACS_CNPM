from __future__ import annotations

import matplotlib
# Force non-interactive backend before importing torch or local modules that use pyplot.
matplotlib.use("Agg")

import argparse
import copy
import csv
import gc
import logging
import math
import os
import subprocess
import sys
import time
from collections import Counter
from dataclasses import fields
from functools import partial
from itertools import count, islice
from pathlib import Path
from typing import Callable, Dict, List, Optional, Sequence, Tuple, Union

import torch
import torch.nn.functional as F
from torch import Tensor, nn, optim
from torch.amp import GradScaler
from torch.utils.data import DataLoader
from tqdm import tqdm

from trkh.core.config import (
    AugmentationConfig,
    IMAGENET_MEAN,
    IMAGENET_STD,
    ModelConfig,
    TrainConfig,
    default_data_yaml,
    load_data_spec,
    to_serializable,
)
from trkh.data.dataset import (
    ClassificationFolderDataset,
    HardSampleRepeatDataset,
    IndexedSampleDataset,
    MangoYOLOCropDataset,
    PseudoVideoAugmenter,
    RareClassRepeatDataset,
    SampleWeightDataset,
    StrictBalancedBatchSampler,
    TeacherProbabilityDataset,
    build_rare_class_repeat_factors,
    build_eval_transform,
    build_train_collate_fn,
    build_train_transform,
)
from trkh.training.debug_and_optimization import (
    GradientCheckpointingEnabler,
    MultiScaleTransform,
    TrainingScheduler,
)
from trkh.evaluation.evaluate import DETECTION_SCORE_MODES, evaluate_model, save_evaluation_artifacts
from trkh.training.loss import HybridDetectionClassificationLoss
from trkh.training.losses import BalancedSoftmaxFocalLoss, LDAMFocalLoss, SupervisedContrastiveLoss
from trkh.models.model import (
    build_model_from_checkpoint,
    classification_logits_from_features,
    create_model,
    extract_bbox_from_model_output,
    extract_head_input_from_features,
    load_model_state,
)
from trkh.core.utils import (
    SAM,
    append_csv_row,
    autocast_context,
    build_safe_dataloader_kwargs,
    build_warmup_decay_scheduler,
    build_optimizer_param_groups,
    compute_class_distribution_skew,
    count_parameters,
    ensure_dir,
    format_seconds,
    json_dump,
    load_checkpoint,
    maybe_enable_dataset_image_cache,
    plot_all_training_metrics,
    plot_detection_training_metrics,
    plot_dataset_overview,
    plot_dataset_color_audit,
    plot_per_class_training_metrics,
    plot_per_class_validation_metric,
    plot_train_val_final_test_metrics,
    plot_training_history,
    plot_validation_convergence,
    save_checkpoint,
    set_seed,
    resolve_amp_dtype,
    summarize_token_norms,
    timestamp_run_name,
)


DEPRECATED_PLATEAU_FLAGS = (
    "--plateau-factor",
    "--plateau-patience",
    "--plateau-threshold",
    "--plateau-threshold-mode",
    "--plateau-cooldown",
    "--cosine-restart-t0",
    "--cosine-restart-t-mult",
)

DETECTION_MODEL_TYPES = {"detr_vit_registers", "vit_registers_hybrid"}
logger = logging.getLogger(__name__)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Train DETR ViT-Registers cho mango multi-object detection.")
    parser.add_argument("--data", type=Path, default=default_data_yaml())
    parser.add_argument(
        "--class-name-mode",
        choices=("auto", "raw", "mango"),
        default=None,
        help=(
            "Cach xu ly names trong data.yaml: auto giu tuong thich mango 4 lop, "
            "raw giu nguyen dataset tuy bien/nhieu lop, mango ep bo 4 lop canonical."
        ),
    )
    parser.add_argument(
        "--expected-num-classes",
        type=int,
        default=0,
        help="Neu > 0, dung de validate so class trong data.yaml truoc khi train.",
    )
    parser.add_argument("--output-dir", type=Path, default=Path(__file__).resolve().parents[2] / "runs")
    parser.add_argument("--run-name", type=str, default=None)
    parser.add_argument(
        "--resume",
        type=Path,
        default=None,
        help="Tiep tuc train tu checkpoint last.pt da luu. Dung cung cau hinh/run-name voi lan train truoc.",
    )
    parser.add_argument(
        "--auto-resume",
        action="store_true",
        default=False,
        help="Tu tim output-dir/run-name/checkpoints/last.pt va tiep tuc neu file ton tai.",
    )
    parser.add_argument(
        "--disable-resume",
        "--fresh-run",
        action="store_true",
        default=False,
        help="Bo qua --resume/--auto-resume va chay moi tu cau hinh CLI hien tai.",
    )
    parser.add_argument(
        "--resume-use-cli-config",
        action="store_true",
        default=False,
        help="Khi resume, dung cau hinh CLI hien tai thay vi train/model config da luu trong checkpoint.",
    )
    parser.add_argument(
        "--resume-reset-optimizer",
        action="store_true",
        default=False,
        help="Khi resume, nap model weights nhung khoi tao lai optimizer.",
    )
    parser.add_argument(
        "--resume-reset-scheduler",
        action="store_true",
        default=False,
        help="Khi resume, nap model weights nhung khoi tao lai scheduler theo cau hinh CLI/checkpoint hien tai.",
    )
    parser.add_argument(
        "--resume-reset-scaler",
        action="store_true",
        default=False,
        help="Khi resume, khong nap AMP GradScaler state tu checkpoint.",
    )
    parser.add_argument(
        "--resume-reset-epoch",
        "--resume-reset-progress",
        action="store_true",
        default=False,
        help=(
            "Khi resume cho phase moi, nap model weights nhung reset epoch/best/early-stopping ve run moi. "
            "Nen dung cung --resume-reset-scheduler de LR warmup/decay khong bi tinh theo epoch cu."
        ),
    )

    parser.add_argument(
        "--model-type",
        choices=("detr_vit_registers", "vit_registers_hybrid", "vit_registers", "resnet50", "mobilenet_v3_large", "vit_b_16"),
        default="vit_registers_hybrid",
    )
    pretrained_group = parser.add_mutually_exclusive_group()
    pretrained_group.add_argument(
        "--pretrained",
        dest="pretrained",
        action="store_true",
        default=False,
        help=(
            "Dung weights ImageNet cua torchvision cho model_type resnet50/mobilenet_v3_large/vit_b_16. "
            "Mac dinh tat de giu TRKH scratch/no-pretrain."
        ),
    )
    pretrained_group.add_argument(
        "--no-pretrained",
        dest="pretrained",
        action="store_false",
        help="Train tu dau; day la mac dinh.",
    )
    parser.add_argument("--image-size", type=int, default=224)
    parser.add_argument("--patch-size", type=int, default=16)
    parser.add_argument("--disable-cnn-stem", action="store_true", default=False)
    parser.add_argument("--stem-channels", type=int, default=32)
    parser.add_argument(
        "--dual-patch-norm",
        action="store_true",
        default=False,
        help="Dung LayerNorm truoc/sau patch embedding de on dinh ViT scratch tren du lieu nho/fine-grained.",
    )
    parser.add_argument(
        "--cnn-feature-fusion",
        action="store_true",
        default=False,
        help="Them nhanh global pooled CNN stem vao logits phan loai; mac dinh tat de giu hanh vi cu.",
    )
    parser.add_argument("--cnn-fusion-dropout", type=float, default=0.1)
    parser.add_argument(
        "--color-stat-fusion",
        action="store_true",
        default=False,
        help=(
            "Them head thong ke mau RGB/center-border vao logits phan loai; "
            "khong dung pretrain va khoi tao residual-zero."
        ),
    )
    parser.add_argument("--color-stat-fusion-dropout", type=float, default=0.1)
    parser.add_argument(
        "--defect-stat-fusion",
        action="store_true",
        default=False,
        help=(
            "Them head thong ke vet toi/dom/local contrast/overexposure vao logits; "
            "nham loi fine-grained do vet hu, bam, qua sang/toi."
        ),
    )
    parser.add_argument("--defect-stat-fusion-dropout", type=float, default=0.1)
    parser.add_argument(
        "--foreground-surface-fusion",
        action="store_true",
        default=False,
        help=(
            "Them head thong ke foreground-only + gray-world color constancy cho maturity/damage boundary; "
            "khong dung pretrain va khoi tao residual-zero."
        ),
    )
    parser.add_argument("--foreground-surface-fusion-dropout", type=float, default=0.1)
    parser.add_argument(
        "--bilinear-patch-fusion",
        action="store_true",
        default=False,
        help=(
            "Them compact bilinear pooling tren patch token de hoc tuong tac texture/mau bac hai; "
            "khong dung pretrain va khoi tao residual-zero."
        ),
    )
    parser.add_argument("--bilinear-patch-rank", type=int, default=32)
    parser.add_argument("--bilinear-patch-dropout", type=float, default=0.1)
    parser.add_argument(
        "--fine-grained-pooling",
        action="store_true",
        default=False,
        help=(
            "Them attention pooling tren patch tokens cho classification-only fine-grained; "
            "mac dinh tat va khoi tao residual-zero."
        ),
    )
    parser.add_argument("--fine-grained-pooling-dropout", type=float, default=0.1)
    parser.add_argument(
        "--multi-branch-fusion",
        action="store_true",
        default=False,
        help=(
            "Them token nhanh mau/edge/CNN-stem vao transformer cho classification fine-grained; "
            "khong dung pretrained va mac dinh tat."
        ),
    )
    parser.add_argument("--branch-color-tokens", type=int, default=1)
    parser.add_argument("--branch-edge-tokens", type=int, default=1)
    parser.add_argument("--branch-cnn-tokens", type=int, default=1)
    parser.add_argument("--branch-token-dropout", type=float, default=0.1)
    parser.add_argument(
        "--detail-patch-enhancement",
        action="store_true",
        default=False,
        help="Them residual mau cuc bo/high-frequency/edge vao tung patch token.",
    )
    parser.add_argument("--detail-patch-dropout", type=float, default=0.05)
    parser.add_argument(
        "--token-pruning",
        action="store_true",
        default=False,
        help="Prune patch token theo attention va foreground prior de giam nhieu nen.",
    )
    parser.add_argument(
        "--token-prune-layers",
        type=str,
        default="2,5",
        help="Danh sach layer 1-based thuc hien pruning, vi du 2,5.",
    )
    parser.add_argument(
        "--token-keep-rates",
        type=str,
        default="0.75,0.50",
        help="Ty le patch goc con lai sau tung prune layer.",
    )
    parser.add_argument("--token-prune-foreground-weight", type=float, default=0.35)
    parser.add_argument(
        "--pairwise-margin-head",
        action="store_true",
        default=False,
        help="Them auxiliary head nhe cho cac cap class de nham nhu 0-1,2-3,4-rest.",
    )
    parser.add_argument("--pairwise-margin-pairs", type=str, default="0-1,2-3,4-rest")
    parser.add_argument("--pairwise-margin-logit-scale", type=float, default=0.35)
    parser.add_argument("--pairwise-margin-dropout", type=float, default=0.05)
    parser.add_argument(
        "--ordinal-maturity-head",
        action="store_true",
        default=False,
        help=(
            "Them mot truc maturity co thu tu cho cac class 0,1,2,3; "
            "khong ep class defect 4 vao cung truc."
        ),
    )
    parser.add_argument("--ordinal-maturity-classes", type=str, default="0,1,2,3")
    parser.add_argument("--ordinal-maturity-logit-scale", type=float, default=0.20)
    parser.add_argument("--ordinal-maturity-dropout", type=float, default=0.05)
    parser.add_argument("--embed-dim", type=int, default=256)
    parser.add_argument("--depth", type=int, default=8)
    parser.add_argument("--num-heads", type=int, default=8)
    parser.add_argument("--mlp-ratio", type=float, default=4.0)
    parser.add_argument("--num-registers", type=int, default=4)
    parser.add_argument("--dropout", type=float, default=0.1)
    parser.add_argument("--attention-dropout", type=float, default=0.0)
    parser.add_argument("--drop-path-rate", type=float, default=0.1)
    parser.add_argument("--register-positional-embedding", action="store_true", default=False)
    parser.add_argument("--gradient-checkpointing", action="store_true", default=False)
    parser.add_argument(
        "--head-pooling",
        choices=("cls", "cls_register_mean", "cls_branch_register_mean"),
        default="cls_register_mean",
    )
    parser.add_argument("--bbox-head-hidden-dim", type=int, default=512)
    parser.add_argument("--num-queries", type=int, default=40)
    parser.add_argument("--decoder-depth", type=int, default=4)
    parser.add_argument("--decoder-num-heads", type=int, default=8)
    parser.add_argument("--decoder-ffn-dim", type=int, default=1024)
    parser.add_argument("--decoder-dropout", type=float, default=0.1)
    parser.add_argument("--decoder-memory-adapter", action="store_true", default=False)
    parser.add_argument("--decoder-memory-adapter-dropout", type=float, default=0.0)
    parser.add_argument(
        "--disable-query-content",
        action="store_true",
        default=False,
        help="Tat learned query content trong DETR decoder; mac dinh bat de giam query collapse.",
    )
    parser.add_argument(
        "--legacy-softmax-background-head",
        action="store_true",
        default=False,
        help="Dung head DETR cu voi background trong softmax; mac dinh la head objectness rieng.",
    )
    parser.add_argument("--objectness-prior-prob", type=float, default=0.125)
    parser.add_argument(
        "--quality-head",
        action="store_true",
        default=False,
        help="Them IoU-quality head de score detection theo chat luong bbox.",
    )
    parser.add_argument("--quality-prior-prob", type=float, default=0.125)
    parser.add_argument(
        "--auxiliary-decoder-loss",
        action="store_true",
        default=False,
        help="Giam sat cac decoder layer trung gian de tang recall va on dinh Hungarian matching.",
    )
    parser.add_argument(
        "--query-denoising-noise",
        type=float,
        default=0.0,
        help="Gaussian noise nhe len learned queries trong train, kieu denoising query khong dung pretrain.",
    )
    parser.add_argument(
        "--count-head",
        action="store_true",
        default=False,
        help="Them count head tu global cls/register feature de uoc luong so object trong anh.",
    )
    parser.add_argument("--count-head-hidden-dim", type=int, default=256)
    parser.add_argument("--count-head-dropout", type=float, default=0.05)
    parser.add_argument("--count-head-prior", type=float, default=1.2)
    parser.add_argument("--temporal-frames", type=int, default=1)
    parser.add_argument("--temporal-num-heads", type=int, default=4)
    parser.add_argument("--temporal-kv-quant-bits", type=int, choices=(0, 4, 8), default=8)

    parser.add_argument("--batch-size", type=int, default=8)
    parser.add_argument("--grad-accum-steps", type=int, default=8)
    parser.add_argument(
        "--epochs",
        type=int,
        default=80,
        help=(
            "So epoch. Dat <= 0 hoac >= 9999 de chay vo han cho den khi early stopping hoac ban tu dung."
        ),
    )
    parser.add_argument("--learning-rate", type=float, default=5e-4)
    parser.add_argument(
        "--backbone-lr-scale",
        type=float,
        default=1.0,
        help="LR multiplier for the image feature extractor; DETR decoder/heads keep the base LR.",
    )
    parser.add_argument("--min-learning-rate", type=float, default=1e-6)
    parser.add_argument("--weight-decay", type=float, default=0.05)
    parser.add_argument("--warmup-epochs", type=int, default=8)
    parser.add_argument("--warmup-start-factor", type=float, default=0.1)
    parser.add_argument("--lr-scheduler", choices=("cosine", "linear"), default="cosine")
    parser.add_argument(
        "--scheduler-total-epochs",
        type=int,
        default=0,
        help="Horizon rieng cho LR scheduler; huu ich khi --epochs <= 0 de train vo han.",
    )
    parser.add_argument("--sam", action="store_true", default=False)
    parser.add_argument("--sam-rho", type=float, default=0.05)
    parser.add_argument("--sam-adaptive", action="store_true", default=False)
    parser.add_argument(
        "--model-ema",
        action="store_true",
        default=False,
        help="Danh gia va luu best.pt bang exponential moving average cua trong so model.",
    )
    parser.add_argument(
        "--model-ema-decay",
        type=float,
        default=0.999,
        help="EMA decay trong (0, 1). EMA khoi dong dong de khong bi tre o cac epoch dau.",
    )
    parser.add_argument("--label-smoothing", type=float, default=0.02)
    parser.add_argument("--grad-clip-norm", type=float, default=1.0)
    parser.add_argument(
        "--max-nonfinite-grad-steps",
        type=int,
        default=8,
        help="Dung train sau N optimizer-step lien tiep co gradient non-finite; dat <=0 de chi skip va canh bao.",
    )
    parser.add_argument("--num-workers", type=int, default=4)
    parser.add_argument(
        "--eval-num-workers",
        type=int,
        default=-1,
        help="So worker rieng cho validation/test. -1 dung theo --num-workers; 0 bat cache anh single-process.",
    )
    parser.add_argument(
        "--train-image-cache-mb",
        type=int,
        default=-1,
        help="Cache anh rieng cho train khi effective workers = 0. -1 dung env/default, 0 tat cache.",
    )
    parser.add_argument(
        "--eval-image-cache-mb",
        type=int,
        default=-1,
        help="Cache anh rieng cho val/test khi effective workers = 0. -1 dung env/default, 0 tat cache.",
    )
    parser.add_argument(
        "--patience",
        type=int,
        default=15,
        help="Early stopping patience. Dat <= 0 de tat early stopping.",
    )
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--deterministic", action="store_true", default=False)
    parser.add_argument("--disable-amp", action="store_true", default=False)
    parser.add_argument("--disable-class-weights", action="store_true", default=False)
    parser.add_argument(
        "--class-weight-mode",
        choices=("uniform", "inverse", "sqrt_inverse", "effective_num"),
        default="sqrt_inverse",
    )
    parser.add_argument("--class-weight-beta", type=float, default=0.999)
    parser.add_argument("--weighted-sampler", "--use-weighted-sampler", action="store_true", default=False)
    parser.add_argument("--weighted-sampler-power", type=float, default=1.75)
    parser.add_argument("--weighted-sampler-epoch-multiplier", type=float, default=2.0)
    parser.add_argument(
        "--disable-balanced-epoch-sampling",
        action="store_true",
        default=False,
        help=(
            "Tat sampler chia deu exposure giua cac class. Mac dinh bat cho classification-only; "
            "sampler chi doc train split va khong thay doi val/test."
        ),
    )
    parser.add_argument("--balanced-epoch-multiplier", type=float, default=1.0)
    parser.add_argument("--balanced-epoch-tolerance", type=float, default=0.10)
    parser.add_argument("--imbalance-auto-tune", action="store_true", default=False)
    parser.add_argument("--disable-imbalance-auto-tune", action="store_true", default=False)
    parser.add_argument("--imbalance-sampler-disable-threshold", type=float, default=0.18)
    parser.add_argument("--max-train-batches", type=int, default=0)
    parser.add_argument("--max-val-batches", type=int, default=0)
    parser.add_argument(
        "--skip-final-test",
        action="store_true",
        default=False,
        help="Bo qua evaluate test cuoi run; chi dung cho smoke/debug nhanh, khong dung cho run chinh.",
    )
    parser.add_argument(
        "--trace-architecture",
        action="store_true",
        default=False,
        help=(
            "Sau khi train/evaluate xong, tao architecture trace 1 anh train ngau nhien moi class "
            "qua cac block bang checkpoints/best.pt."
        ),
    )
    parser.add_argument(
        "--trace-architecture-output-dir",
        type=Path,
        default=None,
        help="Thu muc trace. Mac dinh la <run_dir>/architecture_trace.",
    )
    parser.add_argument("--trace-architecture-seed", type=int, default=42)
    parser.add_argument(
        "--trace-architecture-device",
        choices=("auto", "cpu", "cuda"),
        default="auto",
    )
    parser.add_argument(
        "--full-image-detection",
        action="store_true",
        default=False,
        help="Train detector tren anh goc day du thay vi crop quanh primary object.",
    )
    parser.add_argument(
        "--disable-classification-object-crops",
        action="store_true",
        default=False,
        help=(
            "Classification-only mac dinh bien moi bbox thanh mot crop sample. "
            "Bat co nay de quay ve che do chi dung primary object cua moi anh."
        ),
    )
    parser.add_argument("--focal-loss-gamma", type=float, default=2.0)
    parser.add_argument("--focal-loss-mix", type=float, default=0.35)
    parser.add_argument("--disable-ldam", action="store_true", default=False)
    parser.add_argument("--ldam-max-margin", type=float, default=0.5)
    parser.add_argument("--ldam-scale", type=float, default=30.0)
    parser.add_argument(
        "--classification-loss",
        choices=("ldam_focal", "balanced_softmax"),
        default="ldam_focal",
        help="Loss cho classification-only. balanced_softmax dung thong ke class de giam lech prior.",
    )
    parser.add_argument("--balanced-softmax-tau", type=float, default=1.0)
    parser.add_argument(
        "--metric-learning-loss-weight",
        type=float,
        default=0.0,
        help="Trong so Supervised Contrastive Loss tren embedding phan loai; 0 de tat.",
    )
    parser.add_argument("--metric-learning-temperature", type=float, default=0.12)
    parser.add_argument(
        "--disable-metric-learning-class-balanced",
        action="store_true",
        default=False,
        help="Tat anchor weighting theo tan suat class trong batch cho metric-learning.",
    )
    parser.add_argument(
        "--metric-learning-sources",
        type=str,
        default="head",
        help=(
            "Danh sach nguon embedding cho SupCon, cach nhau boi dau phay. "
            "Ho tro head, cnn, patch, registers, all."
        ),
    )
    parser.add_argument(
        "--boundary-contrastive-loss-weight",
        type=float,
        default=0.0,
        help=(
            "Trong so contrastive loss chi tren cac cap class de nham nhu 0-1,1-2,2-3,4-rest; "
            "0 de tat. Loss chay in-batch va khong tao pair dataset."
        ),
    )
    parser.add_argument(
        "--boundary-contrastive-pairs",
        type=str,
        default="0-1,1-2,2-3,4-rest",
        help="Danh sach cap class cho boundary contrastive, dung cu phap 0-1 hoac 4-rest.",
    )
    parser.add_argument(
        "--boundary-contrastive-sources",
        type=str,
        default="head,patch",
        help="Nguon embedding cho boundary contrastive: head, cnn, patch, registers, all.",
    )
    parser.add_argument("--boundary-contrastive-margin", type=float, default=0.12)
    parser.add_argument("--boundary-contrastive-temperature", type=float, default=0.20)
    parser.add_argument(
        "--boundary-contrastive-max-pairs",
        type=int,
        default=128,
        help="So anchor-loss toi da moi source/pair; lay hard terms de chi phi on dinh.",
    )
    parser.add_argument(
        "--foreground-consistency-loss-weight",
        type=float,
        default=0.0,
        help=(
            "Trong so loss phat patch-token energy nam ngoai pseudo foreground. "
            "Dung cho classification crop de giam hoc nen/ria anh; 0 de tat."
        ),
    )
    parser.add_argument(
        "--foreground-consistency-margin",
        type=float,
        default=0.08,
        help="Nguong mem tao pseudo foreground tu anh da normalize; lon hon se mask chat hon.",
    )
    parser.add_argument(
        "--attention-view-loss-weight",
        type=float,
        default=0.0,
        help=(
            "Trong so CE tren mot view crop/drop dan huong boi patch attention; "
            "0 de tat. View chi dung trong classification training."
        ),
    )
    parser.add_argument("--attention-crop-probability", type=float, default=0.50)
    parser.add_argument("--attention-drop-probability", type=float, default=0.25)
    parser.add_argument(
        "--attention-view-start-epoch",
        type=int,
        default=2,
        help="Epoch 1-based bat dau attention crop/drop, de attention co warm-up.",
    )
    parser.add_argument("--attention-crop-threshold", type=float, default=0.55)
    parser.add_argument("--attention-drop-threshold", type=float, default=0.70)
    parser.add_argument("--attention-crop-padding-ratio", type=float, default=0.08)
    parser.add_argument("--attention-crop-min-area-ratio", type=float, default=0.20)
    parser.add_argument("--attention-view-foreground-weight", type=float, default=0.35)
    parser.add_argument(
        "--attention-view-score-source",
        type=str,
        choices=("learned_attention", "surface_detail", "hybrid"),
        default="learned_attention",
        help=(
            "Nguon score tao attention crop/drop. learned_attention giu hanh vi v8; "
            "surface_detail dung high-frequency/edge gated boi foreground; hybrid tron ca hai."
        ),
    )
    parser.add_argument(
        "--attention-drop-blur-kernel",
        type=int,
        default=15,
        help="Kernel le de blur vung salient trong attention-drop.",
    )
    parser.add_argument(
        "--attention-drop-dilation-kernel",
        type=int,
        default=5,
        help="Kernel le de noi rong vung salient truoc khi ap dung gioi han dien tich.",
    )
    parser.add_argument(
        "--attention-drop-min-area-ratio",
        type=float,
        default=0.06,
        help="Ty le dien tich anh toi thieu bi blur trong mot attention-drop view.",
    )
    parser.add_argument(
        "--attention-drop-max-area-ratio",
        type=float,
        default=0.16,
        help="Ty le dien tich anh toi da bi blur trong mot attention-drop view.",
    )
    parser.add_argument(
        "--register-diversity-loss-weight",
        type=float,
        default=0.0,
        help=(
            "Trong so regularizer giam cosine similarity giua cac register token "
            "trong cung anh; 0 de tat."
        ),
    )
    parser.add_argument(
        "--pairwise-margin-loss-weight",
        type=float,
        default=0.0,
        help="Trong so auxiliary BCE loss cho pairwise margin head; 0 de chi dung logits adjustment neu head ton tai.",
    )
    parser.add_argument(
        "--ordinal-maturity-loss-weight",
        type=float,
        default=0.0,
        help="Trong so SmoothL1 loss cho truc maturity co thu tu; chi tinh tren cac class da khai bao.",
    )
    parser.add_argument(
        "--pretrained-distillation",
        dest="pretrained_distillation",
        action="store_true",
        help="Bat knowledge distillation tu checkpoint pretrained local cho custom TRKH.",
    )
    parser.add_argument(
        "--no-pretrained-distillation",
        dest="pretrained_distillation",
        action="store_false",
        help="Tat knowledge distillation; custom TRKH train/fine-tune khong dung teacher.",
    )
    parser.set_defaults(pretrained_distillation=False)
    parser.add_argument("--distillation-teacher-checkpoint", type=Path, default=None)
    parser.add_argument(
        "--distillation-teacher-csv",
        "--offline-distillation-csv",
        dest="distillation_teacher_csv",
        type=Path,
        default=None,
        help=(
            "CSV soft-label da map theo class order TRKH voi cot path,prob_0..prob_N. "
            "Dung de distill tu ensemble pretrained ma khong forward teacher trong moi batch."
        ),
    )
    parser.add_argument("--distillation-weight", type=float, default=0.10)
    parser.add_argument("--distillation-temperature", type=float, default=2.0)
    parser.add_argument("--distillation-focus-class-index", type=int, default=1)
    parser.add_argument("--distillation-focus-class-weight", type=float, default=1.5)
    parser.add_argument(
        "--elr-loss-weight",
        type=float,
        default=0.0,
        help=(
            "Trong so Early-Learning Regularization cho classification-only; "
            "0 de tat. Dung khi nghi nhan/boundary noise."
        ),
    )
    parser.add_argument(
        "--elr-beta",
        type=float,
        default=0.70,
        help="EMA beta cho target history cua ELR.",
    )
    parser.add_argument(
        "--elr-start-epoch",
        type=int,
        default=2,
        help="Epoch bat dau cong ELR loss; target history van duoc cap nhat tu dau.",
    )
    parser.add_argument(
        "--sample-weight-manifest",
        type=Path,
        default=None,
        help=(
            "CSV train-only co cot image_path/path va tuy chon sample_weight; "
            "dung de tang loss cho hard boundary samples ma khong tang class 1 toan cuc."
        ),
    )
    parser.add_argument(
        "--sample-weight-factor",
        type=float,
        default=1.0,
        help="Weight mac dinh cho dong manifest khong co cot sample_weight.",
    )
    parser.add_argument(
        "--sample-weight-max",
        type=float,
        default=5.0,
        help="Tran sample weight de tranh batch loss bi mot mau chi phoi.",
    )
    parser.add_argument(
        "--balance-auto-max-repeat-factor",
        type=float,
        default=0.0,
        help=(
            "Tran auto repeat/augmentation factor lay tu canbang.yaml. "
            "Dat <=0 de giu auto goc; dung 1.6-2.0 khi class hiem qua giong class khac."
        ),
    )
    parser.add_argument("--batch-mix-probability", type=float, default=0.6)
    parser.add_argument("--mosaic-probability", type=float, default=0.25)
    parser.add_argument("--mosaic-min-split", type=float, default=0.35)
    parser.add_argument("--mosaic-max-split", type=float, default=0.65)
    parser.add_argument("--mixup-probability", type=float, default=0.0)
    parser.add_argument("--mixup-alpha", type=float, default=0.4)
    parser.add_argument("--cutmix-probability", type=float, default=0.45)
    parser.add_argument("--cutmix-alpha", type=float, default=1.0)
    parser.add_argument("--copy-paste-probability", type=float, default=0.0)
    parser.add_argument("--copy-paste-max-objects", type=int, default=2)
    parser.add_argument(
        "--targeted-copy-paste-scale-threshold",
        type=float,
        default=1.5,
        help="Tu dong uu tien paste object cua lop co augmentation/repeat scale >= nguong nay.",
    )
    parser.add_argument(
        "--targeted-copy-paste-probability",
        type=float,
        default=1.0,
        help="Xac suat chon object cua lop hiem khi source image co object du dieu kien.",
    )
    parser.add_argument("--eval-tta", action="store_true", default=False)
    parser.add_argument("--tta-brightness-delta", type=float, default=0.08)
    parser.add_argument("--disable-artifact-logging", action="store_true", default=False)
    parser.add_argument("--debug-loss", action="store_true", default=False)
    parser.add_argument("--debug-iou", action="store_true", default=False)
    parser.add_argument("--optimized-scheduler", action="store_true", default=False)
    parser.add_argument("--stage1-epochs-optimized", type=int, default=5)
    parser.add_argument("--multi-scale-training", action="store_true", default=False)
    parser.add_argument("--multi-scale-epochs", type=int, default=10)
    parser.add_argument("--stage1-epochs", type=int, default=30)
    parser.add_argument(
        "--stage1-auto-advance-macro-f1-threshold",
        type=float,
        default=0.0,
        help="Neu > 0, tu chuyen stage1 sang stage2 khi val_macro_f1 dat nguong nay.",
    )
    parser.add_argument(
        "--stage1-auto-advance-min-epochs",
        type=int,
        default=1,
        help="So epoch stage1 toi thieu truoc khi cho phep tu chuyen sang stage2.",
    )
    parser.add_argument("--stage1-bbox-l1-loss-weight", type=float, default=0.0)
    parser.add_argument("--stage1-bbox-giou-loss-weight", type=float, default=0.0)
    parser.add_argument(
        "--best-metric",
        choices=(
            "composite",
            "macro_f1",
            "balanced_macro_f1",
            "fair_macro_f1",
            "loss_aware_fair_macro_f1",
            "val_loss",
            "detection_f1",
            "macro_detection_hmean",
        ),
        default="composite",
    )
    parser.add_argument("--fair-f1-gap-target", type=float, default=0.05)
    parser.add_argument("--fair-f1-gap-penalty", type=float, default=1.5)
    parser.add_argument("--fair-f1-min-weight", type=float, default=0.25)
    parser.add_argument(
        "--fair-f1-loss-weight",
        type=float,
        default=0.0,
        help="Trong loss_aware_fair_macro_f1, tru bot val_loss * weight de tranh chon checkpoint overfit.",
    )
    parser.add_argument("--cls-loss-weight", type=float, default=1.0)
    parser.add_argument("--disable-classification-overfit-guard", action="store_true", default=False)
    parser.add_argument("--classification-guard-macro-f1-threshold", type=float, default=0.985)
    parser.add_argument("--classification-guard-detection-gap", type=float, default=0.12)
    parser.add_argument("--classification-guard-min-cls-weight", type=float, default=0.25)
    parser.add_argument("--disable-adaptive-detection-loss", action="store_true", default=False)
    parser.add_argument("--adaptive-detection-macro-f1-threshold", type=float, default=0.93)
    parser.add_argument("--adaptive-detection-f1-target", type=float, default=0.90)
    parser.add_argument("--adaptive-detection-gap-threshold", type=float, default=0.20)
    parser.add_argument("--adaptive-detection-bbox-iou-target", type=float, default=0.70)
    parser.add_argument("--adaptive-detection-max-multiplier", type=float, default=2.0)
    parser.add_argument("--disable-rare-class-recall-guard", action="store_true", default=False)
    parser.add_argument("--rare-class-recall-target", type=float, default=0.70)
    parser.add_argument("--rare-class-recall-guard-scale-threshold", type=float, default=1.5)
    parser.add_argument("--rare-class-recall-guard-max-multiplier", type=float, default=2.0)
    parser.add_argument("--rare-class-recall-guard-min-precision", type=float, default=0.35)
    parser.add_argument(
        "--hard-sample-manifest",
        type=Path,
        default=None,
        help="Optional newline/CSV manifest of train image paths to repeat for hard-example fine-tuning.",
    )
    parser.add_argument("--hard-sample-repeat-factor", type=float, default=1.0)
    parser.add_argument("--bbox-l1-loss-weight", type=float, default=1.0)
    parser.add_argument("--bbox-giou-loss-weight", type=float, default=0.5)
    parser.add_argument("--background-loss-weight", type=float, default=0.3)
    parser.add_argument("--objectness-loss-weight", type=float, default=5.0)
    parser.add_argument("--objectness-focal-alpha", type=float, default=0.75)
    parser.add_argument("--objectness-focal-gamma", type=float, default=0.5)
    parser.add_argument(
        "--matcher-class-cost",
        type=float,
        default=1.0,
        help=(
            "Chi phi class trong Hungarian matcher. Dat 0 de matcher gan query theo box/objectness "
            "truoc, roi moi hoc class tren query da match."
        ),
    )
    parser.add_argument("--matcher-objectness-cost", type=float, default=1.0)
    parser.add_argument("--cardinality-loss-weight", type=float, default=0.0)
    parser.add_argument("--count-loss-weight", type=float, default=0.0)
    parser.add_argument("--quality-loss-weight", type=float, default=0.0)
    parser.add_argument("--auxiliary-loss-weight", type=float, default=0.0)
    parser.add_argument("--count-objectness-consistency-weight", type=float, default=0.0)
    parser.add_argument("--eval-detection-nms-iou-threshold", type=float, default=0.5)
    parser.add_argument("--eval-max-detections-per-image", type=int, default=0)
    parser.add_argument(
        "--eval-detection-score-mode",
        choices=DETECTION_SCORE_MODES,
        default="foreground",
    )
    parser.add_argument("--eval-require-foreground-argmax", action="store_true", default=False)
    parser.add_argument(
        "--eval-adaptive-max-detections",
        action="store_true",
        default=False,
        help="Dung count head/objectness de gioi han so detection moi anh trong luc validate/test.",
    )
    parser.add_argument(
        "--eval-adaptive-count-source",
        choices=("auto", "count_head", "objectness"),
        default="auto",
    )
    parser.add_argument("--eval-adaptive-count-margin", type=int, default=1)
    parser.add_argument("--eval-adaptive-min-detections", type=int, default=1)

    parser.add_argument("--crop-margin-ratio", type=float, default=0.05)
    parser.add_argument(
        "--class-crop-margin-scale-threshold",
        type=float,
        default=1.5,
        help="Tu dong mo rong crop cho class co augmentation/repeat scale >= nguong nay.",
    )
    parser.add_argument(
        "--class-crop-margin-max-ratio",
        type=float,
        default=0.16,
        help="Tran crop margin cho class duoc target tu dong; chi ap dung classification crop.",
    )
    parser.add_argument("--resize-mode", choices=("pad", "crop"), default="pad")
    parser.add_argument("--train-scale-min", type=float, default=0.8)
    parser.add_argument(
        "--train-scale-crop-probability",
        type=float,
        default=1.0,
        help=(
            "Xac suat random scale-crop tren train. Dat <1 de giu mot phan anh day du "
            "va chi mo phong partial-view tren mot ty le sample."
        ),
    )
    parser.add_argument("--brightness", "--color-jitter-brightness", type=float, default=0.2)
    parser.add_argument("--contrast", "--color-jitter-contrast", type=float, default=0.2)
    parser.add_argument("--saturation", "--color-jitter-saturation", type=float, default=0.0)
    parser.add_argument("--hue", "--color-jitter-hue", type=float, default=0.0)
    parser.add_argument("--illumination-normalization", action="store_true", default=False)
    parser.add_argument("--illumination-normalization-strength", type=float, default=0.0)
    parser.add_argument(
        "--background-suppression-mode",
        choices=("none", "gray", "blur", "mean", "desaturate_blur", "blur_gray"),
        default="none",
    )
    parser.add_argument("--background-suppression-probability", type=float, default=0.0)
    parser.add_argument("--background-suppression-margin", type=float, default=0.08)
    parser.add_argument("--background-suppression-blur-radius", type=float, default=7.0)
    parser.add_argument("--local-exposure-probability", type=float, default=0.0)
    parser.add_argument("--local-exposure-strength", type=float, default=0.25)
    parser.add_argument("--obstacle-probability", type=float, default=0.0)
    parser.add_argument("--obstacle-max-area", type=float, default=0.12)
    parser.add_argument("--random-erasing-probability", type=float, default=0.2)
    parser.add_argument("--random-affine-degrees", type=float, default=8.0)
    parser.add_argument("--random-affine-translate", type=float, default=0.05)
    parser.add_argument("--random-affine-scale-min", type=float, default=0.9)
    parser.add_argument("--horizontal-flip-probability", type=float, default=0.5)
    parser.add_argument("--vertical-flip-probability", type=float, default=0.1)
    parser.add_argument("--rotate90-probability", type=float, default=0.15)
    parser.add_argument("--lighting-probability", type=float, default=0.15)
    parser.add_argument("--class-aware-augmentation", dest="class_aware_augmentation", action="store_true")
    parser.add_argument("--disable-class-aware-augmentation", dest="class_aware_augmentation", action="store_false")
    parser.set_defaults(class_aware_augmentation=True)
    parser.add_argument("--class-augmentation-power", type=float, default=0.75)
    parser.add_argument("--class-augmentation-max-scale", type=float, default=1.8)
    parser.add_argument(
        "--class-aware-photometric-augmentation",
        action="store_true",
        default=False,
        help="Cho phep scale color/lighting/erasing theo class hiem; mac dinh tat de giu mau sac.",
    )
    parser.add_argument("--class-aware-mix-probability-boost", type=float, default=0.5)
    parser.add_argument("--class-aware-mix-source-power", type=float, default=1.0)
    parser.add_argument("--rare-class-repeat", dest="rare_class_repeat", action="store_true")
    parser.add_argument("--disable-rare-class-repeat", dest="rare_class_repeat", action="store_false")
    parser.set_defaults(rare_class_repeat=True)
    parser.add_argument("--rare-class-repeat-power", type=float, default=0.5)
    parser.add_argument("--rare-class-repeat-max-factor", type=float, default=3.0)
    parser.add_argument("--rare-class-repeat-min-ratio", type=float, default=0.35)
    parser.add_argument("--randaugment-num-ops", type=int, default=0)
    parser.add_argument("--randaugment-magnitude", type=int, default=0)
    args, unknown = parser.parse_known_args()

    ignored_tokens: List[str] = []
    remaining_unknown: List[str] = []
    index = 0
    while index < len(unknown):
        token = unknown[index]
        matched_flag = next(
            (flag for flag in DEPRECATED_PLATEAU_FLAGS if token == flag or token.startswith(f"{flag}=")),
            None,
        )
        if matched_flag is not None:
            ignored_tokens.append(token)
            if token == matched_flag and index + 1 < len(unknown) and not unknown[index + 1].startswith("-"):
                ignored_tokens.append(unknown[index + 1])
                index += 1
        else:
            remaining_unknown.append(token)
        index += 1

    if ignored_tokens:
        print(
            "Warning: bo qua cac tham so CLI da bi loai bo: "
            + ", ".join(ignored_tokens),
            flush=True,
        )
    if remaining_unknown:
        parser.error(f"Unrecognized arguments: {' '.join(remaining_unknown)}")
    return args


def build_configs(args: argparse.Namespace) -> Tuple[ModelConfig, TrainConfig, AugmentationConfig]:
    if args.grad_accum_steps < 1:
        raise ValueError("--grad-accum-steps phai >= 1.")
    if args.weighted_sampler_epoch_multiplier < 1.0:
        raise ValueError("--weighted-sampler-epoch-multiplier phai >= 1.0.")
    if args.resume_reset_epoch and not args.resume_reset_scheduler:
        raise ValueError("--resume-reset-epoch phai dung cung --resume-reset-scheduler de tranh scheduler state/progress lech.")
    if args.scheduler_total_epochs < 0:
        raise ValueError("--scheduler-total-epochs phai >= 0.")
    if not 0.0 < float(args.model_ema_decay) < 1.0:
        raise ValueError("--model-ema-decay phai nam trong (0, 1).")
    if args.backbone_lr_scale <= 0.0:
        raise ValueError("--backbone-lr-scale phai > 0.")
    if args.temporal_frames < 1:
        raise ValueError("--temporal-frames phai >= 1.")
    if args.eval_num_workers < -1:
        raise ValueError("--eval-num-workers phai >= -1.")
    if args.train_image_cache_mb < -1 or args.eval_image_cache_mb < -1:
        raise ValueError("--train-image-cache-mb/--eval-image-cache-mb phai >= -1.")
    if args.cnn_fusion_dropout < 0.0:
        raise ValueError("--cnn-fusion-dropout phai >= 0.")
    if args.color_stat_fusion_dropout < 0.0:
        raise ValueError("--color-stat-fusion-dropout phai >= 0.")
    if args.defect_stat_fusion_dropout < 0.0:
        raise ValueError("--defect-stat-fusion-dropout phai >= 0.")
    if args.foreground_surface_fusion_dropout < 0.0:
        raise ValueError("--foreground-surface-fusion-dropout phai >= 0.")
    if args.bilinear_patch_rank < 8:
        raise ValueError("--bilinear-patch-rank phai >= 8.")
    if args.bilinear_patch_dropout < 0.0:
        raise ValueError("--bilinear-patch-dropout phai >= 0.")
    if args.fine_grained_pooling_dropout < 0.0:
        raise ValueError("--fine-grained-pooling-dropout phai >= 0.")
    if args.branch_token_dropout < 0.0:
        raise ValueError("--branch-token-dropout phai >= 0.")
    if args.branch_color_tokens < 0 or args.branch_edge_tokens < 0 or args.branch_cnn_tokens < 0:
        raise ValueError("--branch-*-tokens phai >= 0.")
    if args.detail_patch_dropout < 0.0:
        raise ValueError("--detail-patch-dropout phai >= 0.")
    if args.token_prune_foreground_weight < 0.0:
        raise ValueError("--token-prune-foreground-weight phai >= 0.")
    if args.balanced_epoch_multiplier <= 0.0:
        raise ValueError("--balanced-epoch-multiplier phai > 0.")
    if not 0.0 <= args.balanced_epoch_tolerance <= 1.0:
        raise ValueError("--balanced-epoch-tolerance phai nam trong [0, 1].")
    if args.sam_rho < 0.0:
        raise ValueError("--sam-rho phai >= 0.")
    if args.ldam_max_margin < 0.0:
        raise ValueError("--ldam-max-margin phai >= 0.")
    if args.ldam_scale < 1.0:
        raise ValueError("--ldam-scale phai >= 1.")
    if args.balanced_softmax_tau < 0.0:
        raise ValueError("--balanced-softmax-tau phai >= 0.")
    if args.metric_learning_loss_weight < 0.0:
        raise ValueError("--metric-learning-loss-weight phai >= 0.")
    if args.metric_learning_temperature <= 0.0:
        raise ValueError("--metric-learning-temperature phai > 0.")
    valid_metric_sources = {"head", "cnn", "patch", "registers", "all"}
    requested_metric_sources = [
        item.strip().lower()
        for item in str(args.metric_learning_sources or "head").replace(";", ",").split(",")
        if item.strip()
    ]
    if not requested_metric_sources:
        requested_metric_sources = ["head"]
    invalid_metric_sources = sorted(set(requested_metric_sources) - valid_metric_sources)
    if invalid_metric_sources:
        raise ValueError(
            "--metric-learning-sources chi ho tro head, cnn, patch, registers, all; "
            f"khong hop le: {invalid_metric_sources}"
        )
    if args.boundary_contrastive_loss_weight < 0.0:
        raise ValueError("--boundary-contrastive-loss-weight phai >= 0.")
    if args.boundary_contrastive_margin < 0.0:
        raise ValueError("--boundary-contrastive-margin phai >= 0.")
    if args.boundary_contrastive_temperature <= 0.0:
        raise ValueError("--boundary-contrastive-temperature phai > 0.")
    if args.boundary_contrastive_max_pairs < 1:
        raise ValueError("--boundary-contrastive-max-pairs phai >= 1.")
    requested_boundary_sources = [
        item.strip().lower()
        for item in str(args.boundary_contrastive_sources or "head").replace(";", ",").split(",")
        if item.strip()
    ]
    if not requested_boundary_sources:
        requested_boundary_sources = ["head"]
    invalid_boundary_sources = sorted(set(requested_boundary_sources) - valid_metric_sources)
    if invalid_boundary_sources:
        raise ValueError(
            "--boundary-contrastive-sources chi ho tro head, cnn, patch, registers, all; "
            f"khong hop le: {invalid_boundary_sources}"
        )
    _parse_boundary_contrastive_pairs(
        args.boundary_contrastive_pairs,
        int(args.expected_num_classes or 0),
    )
    if args.foreground_consistency_loss_weight < 0.0:
        raise ValueError("--foreground-consistency-loss-weight phai >= 0.")
    if args.foreground_consistency_margin < 0.0:
        raise ValueError("--foreground-consistency-margin phai >= 0.")
    if args.attention_view_loss_weight < 0.0:
        raise ValueError("--attention-view-loss-weight phai >= 0.")
    if not 0.0 <= args.attention_crop_probability <= 1.0:
        raise ValueError("--attention-crop-probability phai nam trong [0, 1].")
    if not 0.0 <= args.attention_drop_probability <= 1.0:
        raise ValueError("--attention-drop-probability phai nam trong [0, 1].")
    if args.attention_crop_probability + args.attention_drop_probability > 1.0:
        raise ValueError(
            "Tong --attention-crop-probability va --attention-drop-probability phai <= 1."
        )
    if args.attention_view_start_epoch < 1:
        raise ValueError("--attention-view-start-epoch phai >= 1.")
    if not 0.0 < args.attention_crop_threshold <= 1.0:
        raise ValueError("--attention-crop-threshold phai nam trong (0, 1].")
    if not 0.0 < args.attention_drop_threshold <= 1.0:
        raise ValueError("--attention-drop-threshold phai nam trong (0, 1].")
    if not 0.0 <= args.attention_crop_padding_ratio <= 0.5:
        raise ValueError("--attention-crop-padding-ratio phai nam trong [0, 0.5].")
    if not 0.0 < args.attention_crop_min_area_ratio <= 1.0:
        raise ValueError("--attention-crop-min-area-ratio phai nam trong (0, 1].")
    if not 0.0 <= args.attention_view_foreground_weight <= 1.0:
        raise ValueError("--attention-view-foreground-weight phai nam trong [0, 1].")
    if args.attention_drop_blur_kernel < 1 or args.attention_drop_blur_kernel % 2 == 0:
        raise ValueError("--attention-drop-blur-kernel phai la so le >= 1.")
    if (
        args.attention_drop_dilation_kernel < 1
        or args.attention_drop_dilation_kernel % 2 == 0
    ):
        raise ValueError("--attention-drop-dilation-kernel phai la so le >= 1.")
    if not 0.0 < args.attention_drop_min_area_ratio <= 1.0:
        raise ValueError("--attention-drop-min-area-ratio phai nam trong (0, 1].")
    if not 0.0 < args.attention_drop_max_area_ratio <= 1.0:
        raise ValueError("--attention-drop-max-area-ratio phai nam trong (0, 1].")
    if args.attention_drop_min_area_ratio > args.attention_drop_max_area_ratio:
        raise ValueError(
            "--attention-drop-min-area-ratio phai <= --attention-drop-max-area-ratio."
        )
    if args.sam and args.attention_view_loss_weight > 0.0:
        raise ValueError(
            "Attention-guided views chua ho tro SAM vi hai SAM forward can cung view."
        )
    if args.register_diversity_loss_weight < 0.0:
        raise ValueError("--register-diversity-loss-weight phai >= 0.")
    if 0.0 < args.balance_auto_max_repeat_factor < 1.0:
        raise ValueError("--balance-auto-max-repeat-factor phai >= 1 hoac <= 0 de tat cap.")
    if not 0.0 <= args.fair_f1_gap_target <= 1.0:
        raise ValueError("--fair-f1-gap-target phai nam trong [0, 1].")
    if args.fair_f1_gap_penalty < 0.0:
        raise ValueError("--fair-f1-gap-penalty phai >= 0.")
    if args.fair_f1_min_weight < 0.0:
        raise ValueError("--fair-f1-min-weight phai >= 0.")
    if args.fair_f1_loss_weight < 0.0:
        raise ValueError("--fair-f1-loss-weight phai >= 0.")
    if args.cls_loss_weight < 0.0 or args.bbox_l1_loss_weight < 0.0 or args.bbox_giou_loss_weight < 0.0:
        raise ValueError("Loss weights phai >= 0.")
    if args.stage1_epochs < 0:
        raise ValueError("--stage1-epochs phai >= 0.")
    if not 0.0 <= args.stage1_auto_advance_macro_f1_threshold <= 1.0:
        raise ValueError("--stage1-auto-advance-macro-f1-threshold phai nam trong [0, 1].")
    if args.stage1_auto_advance_min_epochs < 1:
        raise ValueError("--stage1-auto-advance-min-epochs phai >= 1.")
    if args.stage1_epochs_optimized < 0:
        raise ValueError("--stage1-epochs-optimized phai >= 0.")
    if args.optimized_scheduler and int(args.stage1_epochs_optimized) != 5:
        raise ValueError("--stage1-epochs-optimized phai bang 5 khi bat --optimized-scheduler.")
    if args.multi_scale_epochs < 1:
        raise ValueError("--multi-scale-epochs phai >= 1.")
    if args.copy_paste_probability < 0.0:
        raise ValueError("--copy-paste-probability phai >= 0.")
    if args.copy_paste_max_objects < 0:
        raise ValueError("--copy-paste-max-objects phai >= 0.")
    if args.targeted_copy_paste_scale_threshold < 1.0:
        raise ValueError("--targeted-copy-paste-scale-threshold phai >= 1.")
    if not 0.0 <= args.targeted_copy_paste_probability <= 1.0:
        raise ValueError("--targeted-copy-paste-probability phai nam trong [0, 1].")
    if args.stage1_bbox_l1_loss_weight < 0.0 or args.stage1_bbox_giou_loss_weight < 0.0:
        raise ValueError("Stage 1 detection loss weights phai >= 0.")
    if not 0.0 <= args.classification_guard_macro_f1_threshold < 1.0:
        raise ValueError("--classification-guard-macro-f1-threshold phai nam trong [0, 1).")
    if not 0.0 <= args.classification_guard_detection_gap <= 1.0:
        raise ValueError("--classification-guard-detection-gap phai nam trong [0, 1].")
    if not 0.0 <= args.classification_guard_min_cls_weight <= 1.0:
        raise ValueError("--classification-guard-min-cls-weight phai nam trong [0, 1].")
    if not 0.0 <= args.adaptive_detection_macro_f1_threshold < 1.0:
        raise ValueError("--adaptive-detection-macro-f1-threshold phai nam trong [0, 1).")
    if not 0.0 <= args.adaptive_detection_f1_target <= 1.0:
        raise ValueError("--adaptive-detection-f1-target phai nam trong [0, 1].")
    if not 0.0 <= args.adaptive_detection_gap_threshold <= 1.0:
        raise ValueError("--adaptive-detection-gap-threshold phai nam trong [0, 1].")
    if not 0.0 <= args.adaptive_detection_bbox_iou_target <= 1.0:
        raise ValueError("--adaptive-detection-bbox-iou-target phai nam trong [0, 1].")
    if args.adaptive_detection_max_multiplier < 1.0:
        raise ValueError("--adaptive-detection-max-multiplier phai >= 1.")
    if not 0.0 <= args.rare_class_recall_target <= 1.0:
        raise ValueError("--rare-class-recall-target phai nam trong [0, 1].")
    if args.rare_class_recall_guard_scale_threshold < 1.0:
        raise ValueError("--rare-class-recall-guard-scale-threshold phai >= 1.")
    if args.rare_class_recall_guard_max_multiplier < 1.0:
        raise ValueError("--rare-class-recall-guard-max-multiplier phai >= 1.")
    if not 0.0 <= args.rare_class_recall_guard_min_precision <= 1.0:
        raise ValueError("--rare-class-recall-guard-min-precision phai nam trong [0, 1].")
    if args.hard_sample_repeat_factor < 1.0:
        raise ValueError("--hard-sample-repeat-factor phai >= 1.")
    if args.background_loss_weight < 0.0 or args.cardinality_loss_weight < 0.0:
        raise ValueError("Detection loss weights phai >= 0.")
    if args.eval_detection_nms_iou_threshold < 0.0:
        raise ValueError("--eval-detection-nms-iou-threshold phai >= 0.")
    if args.eval_max_detections_per_image < 0:
        raise ValueError("--eval-max-detections-per-image phai >= 0.")
    if args.full_image_detection and args.model_type not in {"detr_vit_registers", "vit_registers_hybrid"}:
        raise ValueError("--full-image-detection chi ho tro cac model detection DETR.")
    if args.crop_margin_ratio < 0.0:
        raise ValueError("--crop-margin-ratio phai >= 0.")
    if args.class_crop_margin_scale_threshold < 1.0:
        raise ValueError("--class-crop-margin-scale-threshold phai >= 1.")
    if args.class_crop_margin_max_ratio < 0.0:
        raise ValueError("--class-crop-margin-max-ratio phai >= 0.")
    if args.class_augmentation_power < 0.0:
        raise ValueError("--class-augmentation-power phai >= 0.")
    if args.class_augmentation_max_scale < 1.0:
        raise ValueError("--class-augmentation-max-scale phai >= 1.")
    if args.rare_class_repeat_power < 0.0:
        raise ValueError("--rare-class-repeat-power phai >= 0.")
    if args.rare_class_repeat_max_factor < 1.0:
        raise ValueError("--rare-class-repeat-max-factor phai >= 1.")
    if not 0.0 <= args.rare_class_repeat_min_ratio <= 1.0:
        raise ValueError("--rare-class-repeat-min-ratio phai nam trong [0, 1].")
    if args.class_aware_mix_probability_boost < 0.0:
        raise ValueError("--class-aware-mix-probability-boost phai >= 0.")
    if args.class_aware_mix_source_power < 0.0:
        raise ValueError("--class-aware-mix-source-power phai >= 0.")
    if args.decoder_memory_adapter_dropout < 0.0:
        raise ValueError("--decoder-memory-adapter-dropout phai >= 0.")
    if not 0.0 < args.objectness_prior_prob < 1.0:
        raise ValueError("--objectness-prior-prob phai nam trong (0, 1).")
    if not 0.0 < args.quality_prior_prob < 1.0:
        raise ValueError("--quality-prior-prob phai nam trong (0, 1).")
    if args.query_denoising_noise < 0.0:
        raise ValueError("--query-denoising-noise phai >= 0.")
    if args.count_head_hidden_dim < 1:
        raise ValueError("--count-head-hidden-dim phai >= 1.")
    if args.count_head_dropout < 0.0:
        raise ValueError("--count-head-dropout phai >= 0.")
    if args.count_head_prior <= 0.0:
        raise ValueError("--count-head-prior phai > 0.")
    if args.pairwise_margin_logit_scale < 0.0:
        raise ValueError("--pairwise-margin-logit-scale phai >= 0.")
    if args.pairwise_margin_dropout < 0.0:
        raise ValueError("--pairwise-margin-dropout phai >= 0.")
    if args.pairwise_margin_loss_weight < 0.0:
        raise ValueError("--pairwise-margin-loss-weight phai >= 0.")
    if args.ordinal_maturity_logit_scale < 0.0:
        raise ValueError("--ordinal-maturity-logit-scale phai >= 0.")
    if args.ordinal_maturity_dropout < 0.0:
        raise ValueError("--ordinal-maturity-dropout phai >= 0.")
    if args.ordinal_maturity_loss_weight < 0.0:
        raise ValueError("--ordinal-maturity-loss-weight phai >= 0.")
    if args.distillation_weight < 0.0:
        raise ValueError("--distillation-weight phai >= 0.")
    if args.distillation_temperature <= 0.0:
        raise ValueError("--distillation-temperature phai > 0.")
    if args.distillation_focus_class_index < 0:
        raise ValueError("--distillation-focus-class-index phai >= 0.")
    if args.distillation_focus_class_weight <= 0.0:
        raise ValueError("--distillation-focus-class-weight phai > 0.")
    if args.elr_loss_weight < 0.0:
        raise ValueError("--elr-loss-weight phai >= 0.")
    if not 0.0 <= args.elr_beta < 1.0:
        raise ValueError("--elr-beta phai nam trong [0, 1).")
    if args.elr_start_epoch < 0:
        raise ValueError("--elr-start-epoch phai >= 0.")
    if args.sample_weight_factor <= 0.0:
        raise ValueError("--sample-weight-factor phai > 0.")
    if args.sample_weight_max <= 0.0:
        raise ValueError("--sample-weight-max phai > 0.")
    if args.sample_weight_max < args.sample_weight_factor:
        raise ValueError("--sample-weight-max phai >= --sample-weight-factor.")
    if args.sample_weight_manifest is not None and not args.sample_weight_manifest.is_file():
        raise FileNotFoundError(
            f"Khong tim thay sample weight manifest: {args.sample_weight_manifest}"
        )
    if (
        args.pretrained_distillation
        and args.distillation_teacher_checkpoint is None
        and args.distillation_teacher_csv is None
    ):
        raise ValueError(
            "--pretrained-distillation yeu cau --distillation-teacher-checkpoint "
            "hoac --distillation-teacher-csv."
        )
    if (
        args.distillation_teacher_checkpoint is not None
        and not args.distillation_teacher_checkpoint.is_file()
    ):
        raise FileNotFoundError(
            f"Khong tim thay teacher checkpoint: {args.distillation_teacher_checkpoint}"
        )
    if (
        args.distillation_teacher_csv is not None
        and not args.distillation_teacher_csv.is_file()
    ):
        raise FileNotFoundError(
            f"Khong tim thay teacher probability CSV: {args.distillation_teacher_csv}"
        )
    if not 0.0 <= args.train_scale_crop_probability <= 1.0:
        raise ValueError("--train-scale-crop-probability phai nam trong [0, 1].")
    if args.objectness_loss_weight < 0.0:
        raise ValueError("--objectness-loss-weight phai >= 0.")
    if not 0.0 <= args.objectness_focal_alpha <= 1.0:
        raise ValueError("--objectness-focal-alpha phai nam trong [0, 1].")
    if args.objectness_focal_gamma < 0.0:
        raise ValueError("--objectness-focal-gamma phai >= 0.")
    if args.matcher_class_cost < 0.0:
        raise ValueError("--matcher-class-cost phai >= 0.")
    if args.matcher_objectness_cost < 0.0:
        raise ValueError("--matcher-objectness-cost phai >= 0.")
    if args.count_loss_weight < 0.0:
        raise ValueError("--count-loss-weight phai >= 0.")
    if args.quality_loss_weight < 0.0:
        raise ValueError("--quality-loss-weight phai >= 0.")
    if args.auxiliary_loss_weight < 0.0:
        raise ValueError("--auxiliary-loss-weight phai >= 0.")
    if args.count_objectness_consistency_weight < 0.0:
        raise ValueError("--count-objectness-consistency-weight phai >= 0.")
    if args.eval_adaptive_count_margin < 0:
        raise ValueError("--eval-adaptive-count-margin phai >= 0.")
    if args.eval_adaptive_min_detections < 0:
        raise ValueError("--eval-adaptive-min-detections phai >= 0.")
    for name in (
        "background_suppression_probability",
        "local_exposure_probability",
        "obstacle_probability",
    ):
        value = float(getattr(args, name))
        if not 0.0 <= value <= 1.0:
            raise ValueError(f"--{name.replace('_', '-')} phai nam trong [0, 1].")
    if args.illumination_normalization_strength < 0.0:
        raise ValueError("--illumination-normalization-strength phai >= 0.")
    if args.background_suppression_margin < 0.0:
        raise ValueError("--background-suppression-margin phai >= 0.")
    if args.background_suppression_blur_radius <= 0.0:
        raise ValueError("--background-suppression-blur-radius phai > 0.")
    if args.local_exposure_strength < 0.0:
        raise ValueError("--local-exposure-strength phai >= 0.")
    if args.obstacle_max_area < 0.0:
        raise ValueError("--obstacle-max-area phai >= 0.")

    model_config = ModelConfig(
        model_type=args.model_type,
        pretrained=bool(args.pretrained),
        head_pooling=args.head_pooling,
        image_size=args.image_size,
        patch_size=args.patch_size,
        use_cnn_stem=not args.disable_cnn_stem,
        stem_channels=args.stem_channels,
        dual_patch_norm=bool(args.dual_patch_norm),
        cnn_feature_fusion=bool(args.cnn_feature_fusion),
        cnn_fusion_dropout=args.cnn_fusion_dropout,
        color_stat_fusion=bool(args.color_stat_fusion),
        color_stat_fusion_dropout=args.color_stat_fusion_dropout,
        defect_stat_fusion=bool(args.defect_stat_fusion),
        defect_stat_fusion_dropout=args.defect_stat_fusion_dropout,
        foreground_surface_fusion=bool(args.foreground_surface_fusion),
        foreground_surface_fusion_dropout=args.foreground_surface_fusion_dropout,
        bilinear_patch_fusion=bool(args.bilinear_patch_fusion),
        bilinear_patch_rank=args.bilinear_patch_rank,
        bilinear_patch_dropout=args.bilinear_patch_dropout,
        fine_grained_pooling=bool(args.fine_grained_pooling),
        fine_grained_pooling_dropout=args.fine_grained_pooling_dropout,
        multi_branch_fusion=bool(args.multi_branch_fusion),
        branch_color_tokens=args.branch_color_tokens,
        branch_edge_tokens=args.branch_edge_tokens,
        branch_cnn_tokens=args.branch_cnn_tokens,
        branch_token_dropout=args.branch_token_dropout,
        detail_patch_enhancement=bool(args.detail_patch_enhancement),
        detail_patch_dropout=args.detail_patch_dropout,
        token_pruning=bool(args.token_pruning),
        token_prune_layers=args.token_prune_layers,
        token_keep_rates=args.token_keep_rates,
        token_prune_foreground_weight=args.token_prune_foreground_weight,
        pairwise_margin_head=bool(args.pairwise_margin_head),
        pairwise_margin_pairs=args.pairwise_margin_pairs,
        pairwise_margin_logit_scale=args.pairwise_margin_logit_scale,
        pairwise_margin_dropout=args.pairwise_margin_dropout,
        ordinal_maturity_head=bool(args.ordinal_maturity_head),
        ordinal_maturity_classes=args.ordinal_maturity_classes,
        ordinal_maturity_logit_scale=args.ordinal_maturity_logit_scale,
        ordinal_maturity_dropout=args.ordinal_maturity_dropout,
        embed_dim=args.embed_dim,
        depth=args.depth,
        num_heads=args.num_heads,
        mlp_ratio=args.mlp_ratio,
        num_registers=args.num_registers,
        dropout=args.dropout,
        attention_dropout=args.attention_dropout,
        drop_path_rate=args.drop_path_rate,
        register_positional_embedding=args.register_positional_embedding,
        gradient_checkpointing=args.gradient_checkpointing,
        bbox_head_hidden_dim=args.bbox_head_hidden_dim,
        num_queries=args.num_queries,
        decoder_depth=args.decoder_depth,
        decoder_num_heads=args.decoder_num_heads,
        decoder_ffn_dim=args.decoder_ffn_dim,
        decoder_dropout=args.decoder_dropout,
        decoder_memory_adapter=args.decoder_memory_adapter,
        decoder_memory_adapter_dropout=args.decoder_memory_adapter_dropout,
        learned_query_content=not args.disable_query_content,
        separate_objectness=not args.legacy_softmax_background_head,
        objectness_prior_prob=args.objectness_prior_prob,
        quality_head=bool(args.quality_head),
        quality_prior_prob=args.quality_prior_prob,
        auxiliary_decoder_outputs=bool(args.auxiliary_decoder_loss or args.auxiliary_loss_weight > 0.0),
        query_denoising_noise=args.query_denoising_noise,
        count_head=bool(args.count_head),
        count_head_hidden_dim=args.count_head_hidden_dim,
        count_head_dropout=args.count_head_dropout,
        count_head_prior=args.count_head_prior,
        temporal_frames=args.temporal_frames,
        temporal_num_heads=args.temporal_num_heads,
        temporal_dropout=args.dropout,
        temporal_kv_quant_bits=args.temporal_kv_quant_bits,
    )
    train_config = TrainConfig(
        batch_size=args.batch_size,
        grad_accum_steps=args.grad_accum_steps,
        epochs=args.epochs,
        learning_rate=args.learning_rate,
        backbone_lr_scale=args.backbone_lr_scale,
        min_learning_rate=args.min_learning_rate,
        weight_decay=args.weight_decay,
        warmup_epochs=args.warmup_epochs,
        warmup_start_factor=args.warmup_start_factor,
        lr_scheduler=args.lr_scheduler,
        scheduler_total_epochs=args.scheduler_total_epochs,
        use_sam=args.sam,
        sam_rho=args.sam_rho,
        sam_adaptive=args.sam_adaptive,
        model_ema=args.model_ema,
        model_ema_decay=args.model_ema_decay,
        label_smoothing=args.label_smoothing,
        grad_clip_norm=args.grad_clip_norm,
        max_nonfinite_grad_steps=args.max_nonfinite_grad_steps,
        num_workers=args.num_workers,
        eval_num_workers=args.eval_num_workers,
        train_image_cache_mb=args.train_image_cache_mb,
        eval_image_cache_mb=args.eval_image_cache_mb,
        early_stopping_patience=args.patience,
        seed=args.seed,
        amp=not args.disable_amp,
        deterministic=args.deterministic,
        use_class_weights=not args.disable_class_weights,
        class_weight_mode=args.class_weight_mode,
        class_weight_beta=args.class_weight_beta,
        use_weighted_sampler=args.weighted_sampler,
        weighted_sampler_power=args.weighted_sampler_power,
        weighted_sampler_epoch_multiplier=args.weighted_sampler_epoch_multiplier,
        balanced_epoch_sampling=not args.disable_balanced_epoch_sampling,
        balanced_epoch_multiplier=args.balanced_epoch_multiplier,
        balanced_epoch_tolerance=args.balanced_epoch_tolerance,
        auto_tune_imbalance=bool(args.imbalance_auto_tune and not args.disable_imbalance_auto_tune),
        imbalance_sampler_disable_threshold=args.imbalance_sampler_disable_threshold,
        max_train_batches=args.max_train_batches,
        max_val_batches=args.max_val_batches,
        focal_loss_gamma=args.focal_loss_gamma,
        focal_loss_mix=args.focal_loss_mix,
        use_ldam=not args.disable_ldam,
        ldam_max_margin=args.ldam_max_margin,
        ldam_scale=args.ldam_scale,
        classification_loss=args.classification_loss,
        balanced_softmax_tau=args.balanced_softmax_tau,
        metric_learning_loss_weight=args.metric_learning_loss_weight,
        metric_learning_temperature=args.metric_learning_temperature,
        metric_learning_class_balanced=not args.disable_metric_learning_class_balanced,
        metric_learning_sources=",".join(requested_metric_sources),
        boundary_contrastive_loss_weight=args.boundary_contrastive_loss_weight,
        boundary_contrastive_pairs=args.boundary_contrastive_pairs,
        boundary_contrastive_sources=",".join(requested_boundary_sources),
        boundary_contrastive_margin=args.boundary_contrastive_margin,
        boundary_contrastive_temperature=args.boundary_contrastive_temperature,
        boundary_contrastive_max_pairs=args.boundary_contrastive_max_pairs,
        foreground_consistency_loss_weight=args.foreground_consistency_loss_weight,
        foreground_consistency_margin=args.foreground_consistency_margin,
        attention_view_loss_weight=args.attention_view_loss_weight,
        attention_crop_probability=args.attention_crop_probability,
        attention_drop_probability=args.attention_drop_probability,
        attention_view_start_epoch=args.attention_view_start_epoch,
        attention_crop_threshold=args.attention_crop_threshold,
        attention_drop_threshold=args.attention_drop_threshold,
        attention_crop_padding_ratio=args.attention_crop_padding_ratio,
        attention_crop_min_area_ratio=args.attention_crop_min_area_ratio,
        attention_view_foreground_weight=args.attention_view_foreground_weight,
        attention_view_score_source=args.attention_view_score_source,
        attention_drop_blur_kernel=args.attention_drop_blur_kernel,
        attention_drop_dilation_kernel=args.attention_drop_dilation_kernel,
        attention_drop_min_area_ratio=args.attention_drop_min_area_ratio,
        attention_drop_max_area_ratio=args.attention_drop_max_area_ratio,
        register_diversity_loss_weight=args.register_diversity_loss_weight,
        pairwise_margin_loss_weight=args.pairwise_margin_loss_weight,
        ordinal_maturity_loss_weight=args.ordinal_maturity_loss_weight,
        pretrained_distillation=bool(
            args.pretrained_distillation or args.distillation_teacher_csv is not None
        ),
        distillation_teacher_checkpoint=str(args.distillation_teacher_checkpoint or ""),
        distillation_teacher_csv=str(args.distillation_teacher_csv or ""),
        distillation_weight=(
            float(args.distillation_weight)
            if args.pretrained_distillation or args.distillation_teacher_csv is not None
            else 0.0
        ),
        distillation_temperature=args.distillation_temperature,
        distillation_focus_class_index=args.distillation_focus_class_index,
        distillation_focus_class_weight=args.distillation_focus_class_weight,
        elr_loss_weight=args.elr_loss_weight,
        elr_beta=args.elr_beta,
        elr_start_epoch=args.elr_start_epoch,
        sample_weight_manifest=str(args.sample_weight_manifest or ""),
        sample_weight_factor=args.sample_weight_factor,
        sample_weight_max=args.sample_weight_max,
        balance_auto_max_repeat_factor=args.balance_auto_max_repeat_factor,
        fair_f1_gap_target=args.fair_f1_gap_target,
        fair_f1_gap_penalty=args.fair_f1_gap_penalty,
        fair_f1_min_weight=args.fair_f1_min_weight,
        fair_f1_loss_weight=args.fair_f1_loss_weight,
        batch_mix_probability=args.batch_mix_probability,
        mosaic_probability=args.mosaic_probability,
        mosaic_min_split=args.mosaic_min_split,
        mosaic_max_split=args.mosaic_max_split,
        mixup_probability=args.mixup_probability,
        mixup_alpha=args.mixup_alpha,
        cutmix_probability=args.cutmix_probability,
        cutmix_alpha=args.cutmix_alpha,
        copy_paste_probability=args.copy_paste_probability,
        copy_paste_max_objects=args.copy_paste_max_objects,
        targeted_copy_paste_scale_threshold=args.targeted_copy_paste_scale_threshold,
        targeted_copy_paste_probability=args.targeted_copy_paste_probability,
        eval_tta=args.eval_tta,
        tta_brightness_delta=args.tta_brightness_delta,
        log_artifact_stats=not args.disable_artifact_logging,
        optimized_scheduler=args.optimized_scheduler,
        stage1_epochs_optimized=5 if args.optimized_scheduler else args.stage1_epochs_optimized,
        multi_scale_training=args.multi_scale_training,
        multi_scale_epochs=args.multi_scale_epochs,
        stage1_epochs=5 if args.optimized_scheduler else args.stage1_epochs,
        stage1_auto_advance_macro_f1_threshold=args.stage1_auto_advance_macro_f1_threshold,
        stage1_auto_advance_min_epochs=args.stage1_auto_advance_min_epochs,
        stage1_bbox_l1_loss_weight=args.stage1_bbox_l1_loss_weight,
        stage1_bbox_giou_loss_weight=args.stage1_bbox_giou_loss_weight,
        best_metric=args.best_metric,
        cls_loss_weight=args.cls_loss_weight,
        classification_overfit_guard=not args.disable_classification_overfit_guard,
        classification_guard_macro_f1_threshold=args.classification_guard_macro_f1_threshold,
        classification_guard_detection_gap=args.classification_guard_detection_gap,
        classification_guard_min_cls_weight=args.classification_guard_min_cls_weight,
        adaptive_detection_loss=not args.disable_adaptive_detection_loss,
        adaptive_detection_macro_f1_threshold=args.adaptive_detection_macro_f1_threshold,
        adaptive_detection_f1_target=args.adaptive_detection_f1_target,
        adaptive_detection_gap_threshold=args.adaptive_detection_gap_threshold,
        adaptive_detection_bbox_iou_target=args.adaptive_detection_bbox_iou_target,
        adaptive_detection_max_multiplier=args.adaptive_detection_max_multiplier,
        rare_class_recall_guard=not args.disable_rare_class_recall_guard,
        rare_class_recall_target=args.rare_class_recall_target,
        rare_class_recall_guard_scale_threshold=args.rare_class_recall_guard_scale_threshold,
        rare_class_recall_guard_max_multiplier=args.rare_class_recall_guard_max_multiplier,
        rare_class_recall_guard_min_precision=args.rare_class_recall_guard_min_precision,
        hard_sample_manifest=str(args.hard_sample_manifest or ""),
        hard_sample_repeat_factor=args.hard_sample_repeat_factor,
        bbox_l1_loss_weight=args.bbox_l1_loss_weight,
        bbox_giou_loss_weight=args.bbox_giou_loss_weight,
        background_loss_weight=args.background_loss_weight,
        objectness_loss_weight=args.objectness_loss_weight,
        objectness_focal_alpha=args.objectness_focal_alpha,
        objectness_focal_gamma=args.objectness_focal_gamma,
        matcher_class_cost=args.matcher_class_cost,
        matcher_objectness_cost=args.matcher_objectness_cost,
        cardinality_loss_weight=args.cardinality_loss_weight,
        count_loss_weight=args.count_loss_weight,
        quality_loss_weight=args.quality_loss_weight,
        auxiliary_loss_weight=args.auxiliary_loss_weight,
        count_objectness_consistency_weight=args.count_objectness_consistency_weight,
        eval_detection_nms_iou_threshold=args.eval_detection_nms_iou_threshold,
        eval_max_detections_per_image=args.eval_max_detections_per_image,
        eval_detection_score_mode=args.eval_detection_score_mode,
        eval_require_foreground_argmax=args.eval_require_foreground_argmax,
        eval_adaptive_max_detections=args.eval_adaptive_max_detections,
        eval_adaptive_count_source=args.eval_adaptive_count_source,
        eval_adaptive_count_margin=args.eval_adaptive_count_margin,
        eval_adaptive_min_detections=args.eval_adaptive_min_detections,
    )
    augmentation_config = AugmentationConfig(
        crop_margin_ratio=args.crop_margin_ratio,
        class_crop_margin_scale_threshold=args.class_crop_margin_scale_threshold,
        class_crop_margin_max_ratio=args.class_crop_margin_max_ratio,
        resize_mode=args.resize_mode,
        random_resized_crop_scale_min=args.train_scale_min,
        random_resized_crop_probability=args.train_scale_crop_probability,
        color_jitter_brightness=args.brightness,
        color_jitter_contrast=args.contrast,
        color_jitter_saturation=args.saturation,
        color_jitter_hue=args.hue,
        random_erasing_probability=args.random_erasing_probability,
        random_affine_degrees=args.random_affine_degrees,
        random_affine_translate=args.random_affine_translate,
        random_affine_scale_min=args.random_affine_scale_min,
        horizontal_flip_probability=args.horizontal_flip_probability,
        vertical_flip_probability=args.vertical_flip_probability,
        rotate90_probability=args.rotate90_probability,
        lighting_probability=args.lighting_probability,
        illumination_normalization=bool(args.illumination_normalization),
        illumination_normalization_strength=args.illumination_normalization_strength,
        background_suppression_mode=args.background_suppression_mode,
        background_suppression_probability=args.background_suppression_probability,
        background_suppression_margin=args.background_suppression_margin,
        background_suppression_blur_radius=args.background_suppression_blur_radius,
        local_exposure_probability=args.local_exposure_probability,
        local_exposure_strength=args.local_exposure_strength,
        obstacle_probability=args.obstacle_probability,
        obstacle_max_area=args.obstacle_max_area,
        class_aware_augmentation=bool(args.class_aware_augmentation),
        class_augmentation_power=args.class_augmentation_power,
        class_augmentation_max_scale=args.class_augmentation_max_scale,
        class_aware_photometric_augmentation=bool(args.class_aware_photometric_augmentation),
        class_aware_mix_probability_boost=args.class_aware_mix_probability_boost,
        class_aware_mix_source_power=args.class_aware_mix_source_power,
        rare_class_repeat=bool(args.rare_class_repeat),
        rare_class_repeat_power=args.rare_class_repeat_power,
        rare_class_repeat_max_factor=args.rare_class_repeat_max_factor,
        rare_class_repeat_min_ratio=args.rare_class_repeat_min_ratio,
        randaugment_num_ops=args.randaugment_num_ops,
        randaugment_magnitude=args.randaugment_magnitude,
    )
    return model_config, train_config, augmentation_config


def _dataclass_from_checkpoint(config_cls, payload: object):
    if not isinstance(payload, dict):
        return config_cls()
    allowed_fields = {field.name for field in fields(config_cls)}
    filtered = {key: value for key, value in payload.items() if key in allowed_fields}
    return config_cls(**filtered)


def _infer_run_location_from_resume_path(resume_path: Optional[Path]) -> Tuple[Optional[Path], Optional[str]]:
    if resume_path is None:
        return None, None
    path = Path(resume_path).resolve()
    if path.parent.name != "checkpoints":
        return None, None
    run_dir = path.parent.parent
    return run_dir.parent, run_dir.name


def _load_resume_configs_from_checkpoint(
    resume_checkpoint: Dict[str, object],
) -> Tuple[ModelConfig, TrainConfig, AugmentationConfig, Optional[Path]]:
    model_config = _dataclass_from_checkpoint(
        ModelConfig,
        resume_checkpoint.get("model_config", {}),
    )
    train_config = _dataclass_from_checkpoint(
        TrainConfig,
        resume_checkpoint.get("train_config", {}),
    )
    augmentation_config = _dataclass_from_checkpoint(
        AugmentationConfig,
        resume_checkpoint.get("augmentation_config", {}),
    )
    data_yaml = resume_checkpoint.get("data_yaml")
    return (
        model_config,
        train_config,
        augmentation_config,
        Path(str(data_yaml)) if data_yaml else None,
    )


def _preload_resume_checkpoint_for_config(args: argparse.Namespace) -> Tuple[Optional[Path], Optional[Dict[str, object]]]:
    if bool(getattr(args, "disable_resume", False)):
        return None, None
    resume_path = Path(args.resume).resolve() if args.resume is not None else None
    if resume_path is not None:
        output_dir, run_name = _infer_run_location_from_resume_path(resume_path)
        if output_dir is not None and args.run_name is None:
            args.output_dir = output_dir
            args.run_name = run_name
    elif args.auto_resume and args.run_name is not None:
        candidate = Path(args.output_dir) / str(args.run_name) / "checkpoints" / "last.pt"
        if candidate.is_file():
            resume_path = candidate.resolve()

    if resume_path is None:
        return None, None
    if not resume_path.is_file():
        raise FileNotFoundError(f"Khong tim thay checkpoint resume: {resume_path}")
    return resume_path, load_checkpoint(resume_path, map_location="cpu")


def apply_balance_file_auto_adjustment(
    data_spec,
    train_config: TrainConfig,
    augmentation_config: AugmentationConfig,
) -> Dict[str, object]:
    balance_spec = getattr(data_spec, "balance", None)
    if balance_spec is None:
        return {"enabled": False, "source": None}

    balance_counts = balance_spec.class_counts(data_spec.num_classes)
    balance_ratios = balance_spec.class_ratios(data_spec.num_classes)
    positive_counts = [int(count) for count in balance_counts if int(count) > 0]
    target_per_class = (
        float(sum(positive_counts)) / float(len(positive_counts))
        if positive_counts
        else 0.0
    )
    sampling_factors = [
        float(target_per_class / float(count)) if int(count) > 0 else 0.0
        for count in balance_counts
    ]
    min_positive_ratio = min((ratio for ratio in balance_ratios if ratio > 0.0), default=0.0)
    max_ratio = max(balance_ratios, default=0.0)

    return {
        "enabled": True,
        "source": str(balance_spec.balance_yaml),
        "version_note": balance_spec.version_note,
        "total_pairs": int(balance_spec.total_pairs),
        "count_scope": str(getattr(balance_spec, "count_scope", "all")),
        "class_counts": balance_counts,
        "class_ratios": balance_ratios,
        "target_samples_per_class": float(target_per_class),
        "balanced_sampling_factors": sampling_factors,
        "auto_repeat_factors": [1.0 for _ in balance_counts],
        "raw_auto_repeat_factors": [1.0 for _ in balance_counts],
        "min_positive_ratio": float(min_positive_ratio),
        "max_ratio": float(max_ratio),
        "train_ratio_config": float(balance_spec.train_ratio_config),
        "val_ratio_config": float(balance_spec.val_ratio_config),
        "test_ratio_config": float(balance_spec.test_ratio_config),
        "imbalance_auto_tune": bool(train_config.auto_tune_imbalance),
        "balanced_epoch_sampling": bool(train_config.balanced_epoch_sampling),
        "photometric_scaled_for_rare_classes": bool(
            augmentation_config.class_aware_photometric_augmentation
        ),
    }


def compute_class_weights(
    class_counts: Sequence[int],
    num_classes: int,
    mode: str = "sqrt_inverse",
    beta: float = 0.999,
    blend: float = 1.0,
) -> Tensor:
    counts = torch.tensor([max(1, int(count)) for count in class_counts], dtype=torch.float32)
    if mode == "uniform":
        weights = torch.ones_like(counts)
    elif mode == "inverse":
        weights = 1.0 / counts
    elif mode == "sqrt_inverse":
        weights = 1.0 / torch.sqrt(counts)
    elif mode == "effective_num":
        beta = float(min(max(beta, 0.0), 0.999999))
        effective_num = 1.0 - torch.pow(torch.full_like(counts, beta), counts)
        weights = (1.0 - beta) / effective_num.clamp(min=1e-12)
    else:
        raise ValueError(f"Khong ho tro class_weight_mode: {mode}")

    if weights.numel() != num_classes:
        raise ValueError("So luong class weight khong khop num_classes.")
    weights = weights / weights.mean().clamp(min=1e-12)
    blend = float(min(max(blend, 0.0), 1.0))
    weights = torch.lerp(torch.ones_like(weights), weights, blend)
    return weights.to(dtype=torch.float32)


def build_weighted_sampler(
    dataset: MangoYOLOCropDataset,
    batch_size: int,
    num_classes: int,
    epoch_multiplier: float = 2.0,
) -> StrictBalancedBatchSampler:
    return StrictBalancedBatchSampler(
        labels=dataset.labels(),
        batch_size=batch_size,
        num_classes=num_classes,
        epoch_multiplier=epoch_multiplier,
    )


def resolve_imbalance_strategy(
    train_config: TrainConfig,
    class_counts: Sequence[int],
) -> Dict[str, object]:
    class_counts = [max(0, int(count)) for count in class_counts]
    skew_metrics = compute_class_distribution_skew(class_counts)
    strength = float(skew_metrics["intervention_strength"]) if train_config.auto_tune_imbalance else 1.0
    strength = min(max(strength, 0.0), 1.0)

    use_weighted_sampler = bool(
        train_config.use_weighted_sampler or train_config.balanced_epoch_sampling
    )
    if (
        train_config.auto_tune_imbalance
        and not train_config.use_weighted_sampler
        and not train_config.balanced_epoch_sampling
        and strength < float(train_config.imbalance_sampler_disable_threshold)
    ):
        use_weighted_sampler = False

    focal_loss_gamma = float(train_config.focal_loss_gamma) * strength
    focal_loss_mix = float(train_config.focal_loss_mix) * strength
    if focal_loss_mix < 0.05:
        focal_loss_mix = 0.0
    if focal_loss_gamma < 0.1:
        focal_loss_gamma = 0.0

    class_weight_blend = (
        0.0
        if train_config.balanced_epoch_sampling
        else strength if train_config.use_class_weights else 0.0
    )
    class_weight_mode = train_config.class_weight_mode if class_weight_blend > 0.05 else "uniform"
    use_class_weights = bool(train_config.use_class_weights and class_weight_blend > 0.05)
    resolved_class_weights = (
        compute_class_weights(
            class_counts,
            len(class_counts),
            mode=class_weight_mode,
            beta=train_config.class_weight_beta,
            blend=class_weight_blend,
        ).tolist()
        if use_class_weights
        else [1.0 for _ in class_counts]
    )

    sampler_power = 1.0 + max(0.0, float(train_config.weighted_sampler_power) - 1.0) * strength
    if train_config.balanced_epoch_sampling:
        sampler_epoch_multiplier = float(train_config.balanced_epoch_multiplier)
    else:
        sampler_epoch_multiplier = 1.0 + max(
            0.0,
            float(train_config.weighted_sampler_epoch_multiplier) - 1.0,
        ) * strength
    ldam_margin_scale = 0.35 + 0.65 * strength
    resolved_ldam_max_margin = float(train_config.ldam_max_margin) * ldam_margin_scale if train_config.use_ldam else 0.0

    return {
        "class_counts": class_counts,
        "skew_metrics": skew_metrics,
        "intervention_strength": strength,
        "use_weighted_sampler": use_weighted_sampler,
        "sampler_type": "strict_balanced" if use_weighted_sampler else "random",
        "balanced_epoch_sampling": bool(train_config.balanced_epoch_sampling),
        "balanced_epoch_tolerance": float(train_config.balanced_epoch_tolerance),
        "weighted_sampler_power": sampler_power,
        "weighted_sampler_epoch_multiplier": sampler_epoch_multiplier,
        "use_class_weights": use_class_weights,
        "class_weight_mode": class_weight_mode,
        "class_weight_blend": class_weight_blend,
        "class_weights": resolved_class_weights,
        "focal_loss_gamma": focal_loss_gamma,
        "focal_loss_mix": focal_loss_mix,
        "ldam_max_margin": resolved_ldam_max_margin,
    }


def _move_batch_item_to_device(item, device: torch.device):
    if torch.is_tensor(item):
        return item.to(device, non_blocking=True)
    if isinstance(item, dict):
        return {
            key: value.to(device, non_blocking=True) if torch.is_tensor(value) else value
            for key, value in item.items()
        }
    if isinstance(item, list):
        return [_move_batch_item_to_device(value, device) for value in item]
    return item


def _clone_batch_item(item):
    if torch.is_tensor(item):
        return item.detach().clone()
    if isinstance(item, dict):
        return {
            key: value.detach().clone() if torch.is_tensor(value) else value
            for key, value in item.items()
        }
    if isinstance(item, list):
        return [_clone_batch_item(value) for value in item]
    return item


def _is_detection_targets(targets) -> bool:
    if isinstance(targets, list):
        return all(isinstance(item, dict) and "labels" in item and "boxes" in item for item in targets)
    return False


def _resolve_scheduler_total_epochs(train_config: TrainConfig) -> int:
    if int(getattr(train_config, "scheduler_total_epochs", 0) or 0) > 0:
        return max(
            int(train_config.warmup_epochs) + 1,
            int(train_config.scheduler_total_epochs),
        )
    if 0 < int(train_config.epochs) < 9999:
        return max(int(train_config.warmup_epochs) + 1, int(train_config.epochs))
    return max(
        int(train_config.warmup_epochs) + 1,
        int(train_config.stage1_epochs) + 120,
    )


def _resolve_detection_stage(
    train_config: TrainConfig,
    epoch_index: int,
    detection_mode: bool,
    stage1_auto_advance_epoch: Optional[int] = None,
) -> Dict[str, object]:
    if not detection_mode:
        return {
            "stage_name": "classification",
            "cls_weight": float(train_config.cls_loss_weight),
            "bbox_l1_weight": float(train_config.bbox_l1_loss_weight),
            "bbox_giou_weight": float(train_config.bbox_giou_loss_weight),
            "objectness_weight": float(train_config.objectness_loss_weight),
            "cardinality_weight": float(train_config.cardinality_loss_weight),
            "count_weight": float(train_config.count_loss_weight),
            "quality_weight": float(train_config.quality_loss_weight),
            "auxiliary_weight": float(train_config.auxiliary_loss_weight),
            "count_objectness_consistency_weight": float(train_config.count_objectness_consistency_weight),
            "matcher_class_cost": float(train_config.matcher_class_cost),
        }
    auto_advanced = (
        stage1_auto_advance_epoch is not None
        and int(stage1_auto_advance_epoch) > 0
        and int(epoch_index) > int(stage1_auto_advance_epoch)
    )
    stage1_active = (
        not auto_advanced
        and int(train_config.stage1_epochs) > 0
        and int(epoch_index) <= int(train_config.stage1_epochs)
    )
    if stage1_active:
        return {
            "stage_name": "stage1_cls_only",
            "cls_weight": float(train_config.cls_loss_weight),
            "bbox_l1_weight": float(train_config.stage1_bbox_l1_loss_weight),
            "bbox_giou_weight": float(train_config.stage1_bbox_giou_loss_weight),
            "objectness_weight": 0.0,
            "cardinality_weight": 0.0,
            "count_weight": 0.0,
            "quality_weight": 0.0,
            "auxiliary_weight": 0.0,
            "count_objectness_consistency_weight": 0.0,
            "matcher_class_cost": max(1.0, float(train_config.matcher_class_cost)),
        }
    return {
        "stage_name": "stage2_full_detection",
        "cls_weight": float(train_config.cls_loss_weight),
        "bbox_l1_weight": float(train_config.bbox_l1_loss_weight),
        "bbox_giou_weight": float(train_config.bbox_giou_loss_weight),
        "objectness_weight": float(train_config.objectness_loss_weight),
        "cardinality_weight": float(train_config.cardinality_loss_weight),
        "count_weight": float(train_config.count_loss_weight),
        "quality_weight": float(train_config.quality_loss_weight),
        "auxiliary_weight": float(train_config.auxiliary_loss_weight),
        "count_objectness_consistency_weight": float(train_config.count_objectness_consistency_weight),
        "matcher_class_cost": float(train_config.matcher_class_cost),
    }


def _configure_detection_criterion(
    criterion: nn.Module,
    stage_config: Dict[str, object],
) -> None:
    if not hasattr(criterion, "set_loss_weights"):
        return
    loss_weight_kwargs = {
        "cls_weight": float(stage_config["cls_weight"]),
        "bbox_l1_weight": float(stage_config["bbox_l1_weight"]),
        "bbox_giou_weight": float(stage_config["bbox_giou_weight"]),
        "sync_matcher_to_loss": True,
    }
    for stage_key, loss_key in (
        ("objectness_weight", "objectness_weight"),
        ("cardinality_weight", "cardinality_weight"),
        ("count_weight", "count_weight"),
        ("quality_weight", "quality_weight"),
        ("auxiliary_weight", "auxiliary_weight"),
        ("count_objectness_consistency_weight", "count_objectness_consistency_weight"),
        ("matcher_class_cost", "matcher_class_cost"),
    ):
        if stage_key in stage_config:
            loss_weight_kwargs[loss_key] = float(stage_config[stage_key])
    criterion.set_loss_weights(**loss_weight_kwargs)


def _detection_f1_from_metrics(metrics: Dict[str, object]) -> float:
    detection_curve = metrics.get("detection_confidence_curve", {})
    if isinstance(detection_curve, dict) and "best_f1_50" in detection_curve:
        return float(detection_curve.get("best_f1_50", 0.0) or 0.0)
    detection = metrics.get("detection", {})
    if isinstance(detection, dict):
        return float(detection.get("f1_50", 0.0) or 0.0)
    return 0.0


def _resolve_classification_overfit_guard(
    train_config: TrainConfig,
    detection_mode: bool,
    stage_name: str,
    previous_val_metrics: Optional[Dict[str, object]],
) -> Dict[str, object]:
    summary: Dict[str, object] = {
        "enabled": bool(train_config.classification_overfit_guard),
        "active": False,
        "multiplier": 1.0,
        "reason": "not_applicable",
    }
    if not train_config.classification_overfit_guard:
        summary["reason"] = "disabled"
        return summary
    if not detection_mode:
        summary["reason"] = "classification_only_model"
        return summary
    if str(stage_name).strip().lower() == "stage1_cls_only":
        summary["reason"] = "stage1_cls_only"
        return summary
    if not previous_val_metrics:
        summary["reason"] = "no_previous_validation_metrics"
        return summary

    macro_f1 = float(previous_val_metrics.get("macro_f1", 0.0) or 0.0)
    detection_f1 = _detection_f1_from_metrics(previous_val_metrics)
    threshold = float(train_config.classification_guard_macro_f1_threshold)
    gap_threshold = float(train_config.classification_guard_detection_gap)
    min_multiplier = float(train_config.classification_guard_min_cls_weight)
    min_multiplier = min(max(min_multiplier, 0.0), 1.0)
    gap = macro_f1 - detection_f1
    summary.update(
        {
            "previous_macro_f1": macro_f1,
            "previous_detection_f1": detection_f1,
            "macro_detection_gap": gap,
            "macro_f1_threshold": threshold,
            "detection_gap_threshold": gap_threshold,
            "min_multiplier": min_multiplier,
        }
    )
    if macro_f1 < threshold:
        summary["reason"] = "classification_not_saturated"
        return summary
    if gap < gap_threshold:
        summary["reason"] = "detection_not_lagging_enough"
        return summary

    f1_excess = (macro_f1 - threshold) / max(1e-6, 1.0 - threshold)
    gap_excess = (gap - gap_threshold) / max(1e-6, 1.0 - gap_threshold)
    guard_strength = min(1.0, max(0.0, max(f1_excess, gap_excess)))
    multiplier = 1.0 - guard_strength * (1.0 - min_multiplier)
    summary.update(
        {
            "active": True,
            "multiplier": float(min(1.0, max(min_multiplier, multiplier))),
            "guard_strength": float(guard_strength),
            "reason": "classification_saturated_detection_lagging",
        }
    )
    return summary


def _bbox_iou_from_metrics(metrics: Dict[str, object]) -> float:
    bbox_metrics = metrics.get("bbox", {})
    if isinstance(bbox_metrics, dict):
        return float(bbox_metrics.get("mean_iou", 0.0) or 0.0)
    return 0.0


def _resolve_adaptive_detection_loss(
    train_config: TrainConfig,
    detection_mode: bool,
    stage_name: str,
    previous_val_metrics: Optional[Dict[str, object]],
) -> Dict[str, object]:
    summary: Dict[str, object] = {
        "enabled": bool(train_config.adaptive_detection_loss),
        "active": False,
        "multiplier": 1.0,
        "reason": "not_applicable",
    }
    if not train_config.adaptive_detection_loss:
        summary["reason"] = "disabled"
        return summary
    if not detection_mode:
        summary["reason"] = "classification_only_model"
        return summary
    if str(stage_name).strip().lower() == "stage1_cls_only":
        summary["reason"] = "stage1_cls_only"
        return summary
    if not previous_val_metrics:
        summary["reason"] = "no_previous_validation_metrics"
        return summary

    macro_f1 = float(previous_val_metrics.get("macro_f1", 0.0) or 0.0)
    detection_f1 = _detection_f1_from_metrics(previous_val_metrics)
    bbox_iou = _bbox_iou_from_metrics(previous_val_metrics)
    macro_threshold = float(train_config.adaptive_detection_macro_f1_threshold)
    detection_target = float(train_config.adaptive_detection_f1_target)
    gap_threshold = float(train_config.adaptive_detection_gap_threshold)
    bbox_target = float(train_config.adaptive_detection_bbox_iou_target)
    max_multiplier = max(1.0, float(train_config.adaptive_detection_max_multiplier))
    macro_detection_gap = macro_f1 - detection_f1
    summary.update(
        {
            "previous_macro_f1": macro_f1,
            "previous_detection_f1": detection_f1,
            "previous_bbox_iou": bbox_iou,
            "macro_f1_threshold": macro_threshold,
            "detection_f1_target": detection_target,
            "detection_gap_threshold": gap_threshold,
            "bbox_iou_target": bbox_target,
            "max_multiplier": max_multiplier,
            "macro_detection_gap": macro_detection_gap,
        }
    )
    if macro_f1 < macro_threshold:
        summary["reason"] = "macro_f1_not_ready"
        return summary
    if detection_f1 >= detection_target and bbox_iou >= bbox_target:
        summary["reason"] = "detection_target_reached"
        return summary
    if macro_detection_gap < gap_threshold and detection_f1 >= detection_target:
        summary["reason"] = "detection_not_lagging"
        return summary

    detection_deficit = max(0.0, detection_target - detection_f1) / max(1e-6, detection_target)
    bbox_deficit = max(0.0, bbox_target - bbox_iou) / max(1e-6, bbox_target)
    gap_excess = max(0.0, macro_detection_gap - gap_threshold) / max(1e-6, 1.0 - gap_threshold)
    adaptation_strength = min(1.0, max(detection_deficit, bbox_deficit, gap_excess))
    multiplier = 1.0 + adaptation_strength * (max_multiplier - 1.0)
    summary.update(
        {
            "active": True,
            "multiplier": float(min(max_multiplier, max(1.0, multiplier))),
            "adaptation_strength": float(adaptation_strength),
            "reason": "macro_ready_detection_lagging",
        }
    )
    return summary


def _apply_adaptive_detection_loss(
    stage_config: Dict[str, object],
    adaptation: Dict[str, object],
) -> None:
    if not adaptation.get("active"):
        return
    multiplier = float(adaptation.get("multiplier", 1.0) or 1.0)
    if multiplier <= 1.0:
        return
    for key in (
        "bbox_l1_weight",
        "bbox_giou_weight",
        "objectness_weight",
        "cardinality_weight",
        "count_weight",
        "quality_weight",
        "auxiliary_weight",
        "count_objectness_consistency_weight",
    ):
        if key in stage_config:
            stage_config[key] = float(stage_config[key]) * multiplier


def _combine_class_target_scales(
    num_classes: int,
    class_augmentation_scales: Sequence[float],
    rare_class_repeat_factors: Sequence[float],
    raw_class_target_scales: Optional[Sequence[float]] = None,
) -> List[float]:
    combined: List[float] = []
    for class_index in range(max(0, int(num_classes))):
        augmentation_scale = (
            float(class_augmentation_scales[class_index])
            if class_index < len(class_augmentation_scales)
            else 1.0
        )
        repeat_factor = (
            float(rare_class_repeat_factors[class_index])
            if class_index < len(rare_class_repeat_factors)
            else 1.0
        )
        raw_factor = (
            float(raw_class_target_scales[class_index])
            if raw_class_target_scales is not None and class_index < len(raw_class_target_scales)
            else 1.0
        )
        combined.append(max(1.0, augmentation_scale, repeat_factor, raw_factor))
    return combined


def _resolve_rare_class_recall_guard(
    train_config: TrainConfig,
    detection_mode: bool,
    stage_name: str,
    previous_val_metrics: Optional[Dict[str, object]],
    class_target_scales: Sequence[float],
) -> Dict[str, object]:
    multipliers = [1.0 for _ in class_target_scales]
    summary: Dict[str, object] = {
        "enabled": bool(train_config.rare_class_recall_guard),
        "active": False,
        "multipliers": multipliers,
        "active_class_count": 0,
        "max_multiplier": 1.0,
        "reason": "not_applicable",
    }
    if not train_config.rare_class_recall_guard:
        summary["reason"] = "disabled"
        return summary
    if str(stage_name).strip().lower() == "stage1_cls_only":
        summary["reason"] = "stage1_cls_only"
        return summary
    if not previous_val_metrics:
        summary["reason"] = "no_previous_validation_metrics"
        return summary

    per_class = previous_val_metrics.get("per_class", [])
    if not isinstance(per_class, list) or not per_class:
        summary["reason"] = "no_per_class_metrics"
        return summary

    target_recall = float(train_config.rare_class_recall_target)
    scale_threshold = float(train_config.rare_class_recall_guard_scale_threshold)
    max_multiplier = max(1.0, float(train_config.rare_class_recall_guard_max_multiplier))
    min_precision = float(train_config.rare_class_recall_guard_min_precision)
    target_classes: List[Dict[str, object]] = []
    metrics_by_class = {
        int(item.get("class_index", index)): item
        for index, item in enumerate(per_class)
        if isinstance(item, dict)
    }
    for class_index, class_scale in enumerate(class_target_scales):
        if float(class_scale) < scale_threshold:
            continue
        class_metrics = metrics_by_class.get(class_index)
        if not class_metrics:
            continue
        support = int(class_metrics.get("support", 0) or 0)
        if support <= 0:
            continue
        precision = float(class_metrics.get("precision", 0.0) or 0.0)
        recall = float(class_metrics.get("recall", 0.0) or 0.0)
        if recall >= target_recall:
            continue
        if precision < min_precision:
            continue

        deficit = (target_recall - recall) / max(1e-6, target_recall)
        multiplier = 1.0 + max(0.0, deficit) * (max_multiplier - 1.0)
        multiplier = min(max_multiplier, max(1.0, multiplier))
        multipliers[class_index] = float(multiplier)
        target_classes.append(
            {
                "class_index": int(class_index),
                "scale": float(class_scale),
                "precision": precision,
                "recall": recall,
                "support": support,
                "multiplier": float(multiplier),
            }
        )

    if not target_classes:
        summary.update(
            {
                "reason": "no_scaled_class_below_recall_target",
                "target_recall": target_recall,
                "scale_threshold": scale_threshold,
                "min_precision": min_precision,
            }
        )
        return summary

    summary.update(
        {
            "active": True,
            "reason": "scaled_class_recall_below_target",
            "multipliers": multipliers,
            "target_classes": target_classes,
            "active_class_count": len(target_classes),
            "max_multiplier": max(float(value) for value in multipliers),
            "target_recall": target_recall,
            "scale_threshold": scale_threshold,
            "min_precision": min_precision,
        }
    )
    return summary


def _resolve_resume_checkpoint_path(
    resume_path: Optional[Path],
    auto_resume: bool,
    run_dir: Path,
    disable_resume: bool = False,
) -> Optional[Path]:
    if disable_resume:
        return None
    if resume_path is not None:
        resolved = Path(resume_path).resolve()
        if not resolved.is_file():
            raise FileNotFoundError(f"Khong tim thay checkpoint resume: {resolved}")
        return resolved
    if auto_resume:
        candidate = run_dir / "checkpoints" / "last.pt"
        return candidate.resolve() if candidate.is_file() else None
    return None


def _move_optimizer_state_to_device(optimizer: optim.Optimizer, device: torch.device) -> None:
    for state in optimizer.state.values():
        for key, value in list(state.items()):
            if torch.is_tensor(value):
                state[key] = value.to(device=device)


ALLOWED_RESUME_EXTENSION_PREFIXES = (
    "quality_head.",
    "cnn_fusion_norm.",
    "cnn_fusion_head.",
    "foreground_surface_fusion_head.",
    "bilinear_patch_fusion_head.",
)


def _load_model_state_allowing_extensions(
    model: nn.Module,
    state_dict: Dict[str, Tensor],
    *,
    allow_extensions: bool,
) -> Optional[Dict[str, object]]:
    try:
        load_model_state(model, state_dict, strict=True)
        return None
    except RuntimeError:
        if not allow_extensions:
            raise
        missing_keys, unexpected_keys = load_model_state(model, state_dict, strict=False)
        disallowed_missing = [
            key
            for key in missing_keys
            if not any(str(key).startswith(prefix) for prefix in ALLOWED_RESUME_EXTENSION_PREFIXES)
        ]
        if disallowed_missing or unexpected_keys:
            raise RuntimeError(
                "Resume partial load chi cho phep them cac module mo rong da duoc khai bao. "
                f"missing={missing_keys}, unexpected={unexpected_keys}"
            )
        return {
            "allowed_missing_keys": list(missing_keys),
            "unexpected_keys": list(unexpected_keys),
            "reason": "allowed_architecture_extension_initialized_from_scratch",
        }


def _load_training_checkpoint(
    *,
    resume_path: Path,
    model: nn.Module,
    optimizer: optim.Optimizer,
    scheduler,
    scaler: Optional[GradScaler],
    device: torch.device,
    class_names: Sequence[str],
    restore_optimizer: bool = True,
    restore_scheduler: bool = True,
    restore_scaler: bool = True,
    allow_added_detection_heads: bool = False,
) -> Tuple[Dict[str, object], Dict[str, object]]:
    checkpoint = load_checkpoint(resume_path, map_location="cpu")
    checkpoint_class_names = checkpoint.get("class_names")
    if checkpoint_class_names is not None and list(checkpoint_class_names) != list(class_names):
        raise ValueError("Class names trong resume checkpoint khong khop voi data.yaml hien tai.")
    if "model_state" not in checkpoint:
        raise ValueError(f"Checkpoint resume thieu model_state: {resume_path}")

    resume_model_state = checkpoint.get("train_model_state", checkpoint["model_state"])
    partial_load_summary = _load_model_state_allowing_extensions(
        model,
        resume_model_state,
        allow_extensions=bool(allow_added_detection_heads),
    )
    if partial_load_summary is not None:
        print(
            {
                "resume_partial_model_load": partial_load_summary
            },
            flush=True,
        )
    optimizer_state = checkpoint.get("optimizer_state") if restore_optimizer else None
    optimizer_restored = optimizer_state is not None
    if optimizer_restored:
        optimizer.load_state_dict(optimizer_state)
        _move_optimizer_state_to_device(optimizer, device)

    scheduler_state = checkpoint.get("scheduler_state") if restore_scheduler else None
    scheduler_restored = False
    if scheduler_state is not None and hasattr(scheduler, "load_state_dict"):
        scheduler.load_state_dict(scheduler_state)
        scheduler_restored = True
    elif scheduler is not None and hasattr(scheduler, "step"):
        scheduler.step(float(int(checkpoint.get("epoch", 0) or 0)))

    scaler_state = checkpoint.get("scaler_state") if restore_scaler else None
    scaler_restored = False
    if scaler is not None and scaler_state:
        scaler.load_state_dict(scaler_state)
        scaler_restored = True

    completed_epoch = int(checkpoint.get("epoch", 0) or 0)
    summary = {
        "path": str(Path(resume_path).resolve()),
        "completed_epoch": completed_epoch,
        "next_epoch": completed_epoch + 1,
        "optimizer_restored": bool(optimizer_restored),
        "scheduler_restored": bool(scheduler_restored),
        "scaler_restored": bool(scaler_restored),
        "best_epoch": int(checkpoint.get("best_epoch", completed_epoch) or 0),
        "epochs_without_improvement": int(checkpoint.get("epochs_without_improvement", 0) or 0),
    }
    return summary, checkpoint


def _initial_training_progress_from_resume(
    resume_summary: Optional[Dict[str, object]],
    resume_checkpoint: Optional[Dict[str, object]],
    *,
    reset_epoch: bool = False,
) -> Dict[str, object]:
    progress = {
        "start_epoch": 1,
        "best_macro_f1": -1.0,
        "best_epoch": 0,
        "best_selection_metric_name": "",
        "best_selection_metric_value": None,
        "best_selection_metric_higher_is_better": True,
        "epochs_without_improvement": 0,
        "stage1_auto_advance_epoch": None,
        "previous_val_metrics": None,
        "reset_epoch": bool(reset_epoch),
    }
    if resume_checkpoint is None or resume_summary is None:
        return progress

    checkpoint_metrics = (
        resume_checkpoint.get("metrics")
        if isinstance(resume_checkpoint.get("metrics"), dict)
        else None
    )
    progress["previous_val_metrics"] = checkpoint_metrics
    if reset_epoch:
        progress["source_completed_epoch"] = int(resume_summary.get("completed_epoch", 0) or 0)
        progress["source_best_epoch"] = int(
            resume_checkpoint.get("best_epoch", resume_checkpoint.get("epoch", 0)) or 0
        )
        return progress

    progress["start_epoch"] = max(1, int(resume_summary.get("next_epoch", 1) or 1))
    progress["best_macro_f1"] = float(resume_checkpoint.get("best_macro_f1", -1.0) or -1.0)
    progress["best_epoch"] = int(resume_checkpoint.get("best_epoch", resume_checkpoint.get("epoch", 0)) or 0)
    best_selection_metric = resume_checkpoint.get("best_selection_metric")
    if not isinstance(best_selection_metric, dict):
        best_selection_metric = resume_checkpoint.get("selection_metric", {})
    if isinstance(best_selection_metric, dict):
        progress["best_selection_metric_name"] = str(best_selection_metric.get("name", ""))
        metric_value = best_selection_metric.get("value")
        progress["best_selection_metric_value"] = float(metric_value) if metric_value is not None else None
        progress["best_selection_metric_higher_is_better"] = bool(
            best_selection_metric.get("higher_is_better", True)
        )
    progress["epochs_without_improvement"] = int(
        resume_checkpoint.get("epochs_without_improvement", 0) or 0
    )
    auto_epoch = resume_checkpoint.get("stage1_auto_advance_epoch")
    if auto_epoch is not None:
        progress["stage1_auto_advance_epoch"] = int(auto_epoch)
    return progress


def _save_interrupt_checkpoint(
    *,
    path: Path,
    model: nn.Module,
    optimizer: optim.Optimizer,
    scheduler,
    scaler: Optional[GradScaler],
    epoch: int,
    model_config: ModelConfig,
    train_config: TrainConfig,
    augmentation_config: AugmentationConfig,
    data_yaml: Path,
    class_names: Sequence[str],
    metrics: Optional[Dict[str, object]],
    data_summary: Dict[str, object],
    imbalance_summary: Dict[str, object],
    best_macro_f1: float,
    best_epoch: int,
    best_selection_metric_name: str,
    best_selection_metric_value: Optional[float],
    best_selection_metric_higher_is_better: bool,
    epochs_without_improvement: int,
    stage1_auto_advance_epoch: Optional[int] = None,
    model_ema: Optional[ModelEMA] = None,
) -> Dict[str, object]:
    checkpoint = {
        "epoch": int(epoch),
        "best_macro_f1": float(best_macro_f1),
        "model_state": model.state_dict(),
        "optimizer_state": optimizer.state_dict(),
        "scheduler_state": scheduler.state_dict() if hasattr(scheduler, "state_dict") else None,
        "scaler_state": scaler.state_dict() if scaler is not None else None,
        "model_config": to_serializable(model_config),
        "train_config": to_serializable(train_config),
        "augmentation_config": to_serializable(augmentation_config),
        "data_yaml": str(data_yaml),
        "class_names": list(class_names),
        "metrics": to_serializable(metrics or {}),
        "data_summary": to_serializable(data_summary),
        "imbalance_summary": to_serializable(imbalance_summary),
        "best_epoch": int(best_epoch),
        "best_selection_metric": {
            "name": best_selection_metric_name,
            "value": best_selection_metric_value,
            "higher_is_better": best_selection_metric_higher_is_better,
        },
        "epochs_without_improvement": int(epochs_without_improvement),
        "stage1_auto_advance_epoch": (
            int(stage1_auto_advance_epoch) if stage1_auto_advance_epoch is not None else None
        ),
        "resume_state": {
            "checkpoint_kind": "keyboard_interrupt",
            "interrupted_epoch": int(epoch),
            "next_epoch": int(epoch) + 1,
        },
    }
    if model_ema is not None:
        checkpoint["ema_model_state"] = model_ema.state_dict()
        checkpoint["ema_updates"] = int(model_ema.updates)
        checkpoint["ema_decay"] = float(model_ema.decay)
        checkpoint["validation_weight_source"] = "ema"
    save_checkpoint(path, checkpoint)
    return checkpoint


def _resolve_checkpoint_selection(
    model_config: ModelConfig,
    train_config: TrainConfig,
    metrics: Dict[str, object],
) -> Tuple[str, float, bool]:
    best_metric = str(train_config.best_metric).strip().lower()
    if best_metric in {"balanced_macro_f1", "fair_macro_f1", "loss_aware_fair_macro_f1"}:
        macro_f1 = float(metrics["macro_f1"])
        per_class = metrics.get("per_class", [])
        class_f1_values = [
            float(item.get("f1", 0.0) or 0.0)
            for item in per_class
            if isinstance(item, dict) and item.get("f1") is not None
        ] if isinstance(per_class, list) else []
        class_gap = (max(class_f1_values) - min(class_f1_values)) if class_f1_values else 0.0
        if best_metric in {"fair_macro_f1", "loss_aware_fair_macro_f1"}:
            min_class_f1 = min(class_f1_values) if class_f1_values else macro_f1
            score = (
                macro_f1
                + float(train_config.fair_f1_min_weight) * min_class_f1
                - float(train_config.fair_f1_gap_penalty)
                * max(0.0, class_gap - float(train_config.fair_f1_gap_target))
            )
            if best_metric == "loss_aware_fair_macro_f1":
                score -= float(train_config.fair_f1_loss_weight) * float(metrics.get("loss", 0.0) or 0.0)
                return "loss_aware_fair_macro_f1_min_class_gap_loss_penalty", score, True
            return "fair_macro_f1_min_class_gap_penalty", score, True
        score = macro_f1 - 0.75 * max(0.0, class_gap - 0.05)
        return "balanced_macro_f1_gap_penalty", score, True

    if model_config.model_type in DETECTION_MODEL_TYPES:
        if best_metric == "val_loss":
            return "val_loss", float(metrics["loss"]), False
        if best_metric == "macro_f1":
            return "val_macro_f1", float(metrics["macro_f1"]), True
        bbox_iou = float(metrics.get("bbox", {}).get("mean_iou", 0.0))
        detection_f1 = float(
            metrics.get("detection_confidence_curve", {}).get(
                "best_f1_50",
                metrics.get("detection", {}).get("f1_50", 0.0),
            )
        )
        if best_metric == "detection_f1":
            return "val_best_detection_f1_50", detection_f1, True
        macro_f1 = float(metrics["macro_f1"])
        if best_metric == "macro_detection_hmean":
            score = (2.0 * macro_f1 * detection_f1) / max(1e-12, macro_f1 + detection_f1)
            return "hmean_macro_f1_detection_f1", score, True
        # The raw DETR validation loss is not well aligned with deploy quality:
        # keep object detection quality as the anchor and use class/box metrics
        # as tie-breakers for stable checkpoints.
        score = 0.20 * macro_f1 + 0.25 * bbox_iou + 0.55 * detection_f1
        return "composite_detection_f1_macro_f1_bbox_iou", score, True
    return "val_macro_f1", float(metrics["macro_f1"]), True


def _is_best_checkpoint_candidate(
    model_config: ModelConfig,
    stage_name: str,
) -> bool:
    if model_config.model_type not in DETECTION_MODEL_TYPES:
        return True
    return str(stage_name).strip().lower() != "stage1_cls_only"


def _forward_model_outputs(
    model: nn.Module,
    images: Tensor,
    image_valid_mask: Optional[Tensor] = None,
) -> Tuple[Optional[Dict[str, Tensor]], object]:
    if hasattr(model, "forward_features") and hasattr(model, "forward_heads") and hasattr(model, "num_registers"):
        features = model.forward_features(images, image_valid_mask=image_valid_mask)
        outputs = model.forward_heads(features)
        return features, outputs
    if hasattr(model, "forward_features") and hasattr(model, "head") and hasattr(model, "num_registers"):
        features = model.forward_features(images, image_valid_mask=image_valid_mask)
        logits = _classification_logits_from_features(model, features)
        return features, logits
    return None, model(images)


def _classification_logits_from_features(model: nn.Module, features: Dict[str, Tensor]) -> Tensor:
    return classification_logits_from_features(model, features)


def _classification_target_indices(targets, logits: Tensor) -> Optional[Tensor]:
    if not torch.is_tensor(targets):
        return None
    if targets.ndim == 1:
        return targets.to(device=logits.device, dtype=torch.long)
    if targets.ndim == 2 and targets.size(1) == logits.size(1):
        return targets.to(device=logits.device, dtype=logits.dtype).argmax(dim=1).to(dtype=torch.long)
    return None


def _parse_metric_learning_sources(sources: Union[str, Sequence[str]]) -> List[str]:
    if isinstance(sources, str):
        items = [
            item.strip().lower()
            for item in sources.replace(";", ",").split(",")
            if item.strip()
        ]
    else:
        items = [str(item).strip().lower() for item in sources if str(item).strip()]
    if not items:
        items = ["head"]
    if "all" in items:
        items = ["head", "cnn", "patch", "registers"]

    allowed = {"head", "cnn", "patch", "registers"}
    ordered: List[str] = []
    for item in items:
        if item not in allowed:
            raise ValueError(
                "metric_learning_sources chi ho tro head, cnn, patch, registers, all."
            )
        if item not in ordered:
            ordered.append(item)
    return ordered


def _parse_boundary_contrastive_pairs(pairs: str, num_classes: int = 0) -> List[Tuple[int, int]]:
    parsed: List[Tuple[int, int]] = []
    for raw_item in str(pairs or "").replace(";", ",").split(","):
        item = raw_item.strip().lower().replace(":", "-")
        if not item:
            continue
        if "-" not in item:
            raise ValueError(f"boundary contrastive pair khong hop le: {raw_item!r}")
        left_text, right_text = [part.strip() for part in item.split("-", 1)]
        if right_text in {"rest", "others", "other"}:
            pair = (-1, int(left_text))
        elif left_text in {"rest", "others", "other"}:
            pair = (-1, int(right_text))
        else:
            pair = (int(left_text), int(right_text))
        for class_index in pair:
            if class_index < 0:
                continue
            if int(num_classes) > 0 and class_index >= int(num_classes):
                raise ValueError(
                    "boundary contrastive pair nam ngoai khoang class: "
                    f"pair={raw_item!r}, num_classes={num_classes}"
                )
        if pair not in parsed:
            parsed.append(pair)
    if not parsed:
        raise ValueError("--boundary-contrastive-pairs khong duoc rong khi loss weight > 0.")
    return parsed


def _embedding_from_features_source(
    *,
    model: nn.Module,
    features: Dict[str, Tensor],
    source: str,
) -> Optional[Tensor]:
    if source == "head":
        return extract_head_input_from_features(model, features)
    if source == "cnn":
        embedding = features.get("cnn_pooled")
        if embedding is not None and hasattr(model, "cnn_fusion_norm"):
            embedding = model.cnn_fusion_norm(embedding)
        return embedding if torch.is_tensor(embedding) else None
    if source == "patch":
        patches = features.get("patches")
        return patches.mean(dim=1) if torch.is_tensor(patches) and patches.ndim == 3 else None
    if source == "registers":
        registers = features.get("registers")
        return registers.mean(dim=1) if torch.is_tensor(registers) and registers.ndim == 3 else None
    return None


def _metric_learning_loss_from_features(
    *,
    model: nn.Module,
    features: Dict[str, Tensor],
    targets: Tensor,
    criterion: nn.Module,
    sources: Union[str, Sequence[str]],
) -> Tuple[Tensor, List[str]]:
    source_losses: List[Tensor] = []
    used_sources: List[str] = []
    for source in _parse_metric_learning_sources(sources):
        embedding = _embedding_from_features_source(
            model=model,
            features=features,
            source=source,
        )

        if embedding is None or not torch.is_tensor(embedding) or embedding.ndim != 2:
            continue
        source_losses.append(criterion(embedding, targets))
        used_sources.append(source)

    if not source_losses:
        zero = next(iter(features.values())).sum() * 0.0
        return zero, used_sources
    return torch.stack(source_losses).mean(), used_sources


def _boundary_contrastive_loss_for_embedding(
    embeddings: Tensor,
    targets: Tensor,
    *,
    pairs: Sequence[Tuple[int, int]],
    margin: float,
    temperature: float,
    max_pairs: int,
) -> Tuple[Tensor, int]:
    if embeddings.ndim != 2:
        raise ValueError("boundary contrastive embeddings phai co shape [batch, dim].")
    if targets.ndim != 1:
        raise ValueError("boundary contrastive targets phai co shape [batch].")
    if embeddings.size(0) != targets.size(0):
        raise ValueError("So sample cua embeddings va targets khong khop.")
    if embeddings.size(0) < 3:
        return embeddings.sum() * 0.0, 0

    device = embeddings.device
    target_indices = targets.to(device=device, dtype=torch.long).view(-1)
    normalized = F.normalize(embeddings.float(), dim=1, eps=1e-6)
    similarity = torch.matmul(normalized, normalized.T)
    batch_size = int(embeddings.size(0))
    self_mask = torch.eye(batch_size, device=device, dtype=torch.bool)
    pair_losses: List[Tensor] = []
    pair_terms = 0

    for left_class, right_class in pairs:
        if int(left_class) < 0:
            right_mask = target_indices == int(right_class)
            left_mask = target_indices != int(right_class)
        else:
            left_mask = target_indices == int(left_class)
            right_mask = target_indices == int(right_class)
        if not bool(left_mask.any().item()) or not bool(right_mask.any().item()):
            continue

        side_losses: List[Tensor] = []
        for anchor_mask, negative_mask in ((left_mask, right_mask), (right_mask, left_mask)):
            anchor_positive_mask = anchor_mask.unsqueeze(0) & anchor_mask.unsqueeze(1) & ~self_mask
            anchor_negative_mask = anchor_mask.unsqueeze(1) & negative_mask.unsqueeze(0)
            valid_anchor_mask = anchor_positive_mask.any(dim=1) & anchor_negative_mask.any(dim=1)
            if not bool(valid_anchor_mask.any().item()):
                continue
            positive_counts = anchor_positive_mask.sum(dim=1).clamp(min=1).to(dtype=similarity.dtype)
            mean_positive = (
                similarity.masked_fill(~anchor_positive_mask, 0.0).sum(dim=1)
                / positive_counts
            )
            hardest_negative = similarity.masked_fill(~anchor_negative_mask, -1.0e4).max(dim=1).values
            anchor_losses = F.softplus(
                (hardest_negative - mean_positive + float(margin))
                / max(float(temperature), 1e-6)
            )[valid_anchor_mask]
            if anchor_losses.numel() == 0:
                continue
            if anchor_losses.numel() > int(max_pairs):
                anchor_losses = torch.topk(anchor_losses, k=int(max_pairs), largest=True).values
            pair_terms += int(anchor_losses.numel())
            side_losses.append(anchor_losses.mean())
        if side_losses:
            pair_losses.append(torch.stack(side_losses).mean())

    if not pair_losses:
        return embeddings.sum() * 0.0, 0
    return torch.stack(pair_losses).mean().to(dtype=embeddings.dtype), pair_terms


def _boundary_contrastive_loss_from_features(
    *,
    model: nn.Module,
    features: Dict[str, Tensor],
    targets: Tensor,
    pairs: str,
    sources: Union[str, Sequence[str]],
    margin: float,
    temperature: float,
    max_pairs: int,
    num_classes: int,
) -> Tuple[Tensor, Dict[str, object]]:
    parsed_pairs = _parse_boundary_contrastive_pairs(pairs, num_classes)
    source_losses: List[Tensor] = []
    used_sources: List[str] = []
    total_terms = 0
    for source in _parse_metric_learning_sources(sources):
        embedding = _embedding_from_features_source(
            model=model,
            features=features,
            source=source,
        )
        if embedding is None or not torch.is_tensor(embedding) or embedding.ndim != 2:
            continue
        source_loss, source_terms = _boundary_contrastive_loss_for_embedding(
            embedding,
            targets,
            pairs=parsed_pairs,
            margin=margin,
            temperature=temperature,
            max_pairs=max_pairs,
        )
        source_losses.append(source_loss)
        used_sources.append(source)
        total_terms += int(source_terms)

    if not source_losses:
        zero = next(iter(features.values())).sum() * 0.0
        return zero, {"used_sources": used_sources, "terms": 0, "pairs": parsed_pairs}
    return torch.stack(source_losses).mean(), {
        "used_sources": used_sources,
        "terms": int(total_terms),
        "pairs": parsed_pairs,
    }


def _pseudo_foreground_mask_from_normalized_images(
    images: Tensor,
    *,
    margin: float = 0.08,
) -> Tensor:
    if images.ndim != 4 or images.size(1) != 3:
        raise ValueError("foreground mask yeu cau images co shape [B, 3, H, W].")
    device = images.device
    dtype = images.dtype
    mean = torch.tensor(IMAGENET_MEAN, device=device, dtype=dtype).view(1, 3, 1, 1)
    std = torch.tensor(IMAGENET_STD, device=device, dtype=dtype).view(1, 3, 1, 1)
    rgb = (images * std + mean).clamp(0.0, 1.0)
    gray = rgb.mean(dim=1, keepdim=True)
    batch_size = int(gray.size(0))
    flattened_rgb = rgb.flatten(2)
    flattened_gray = gray.flatten(2)
    median_rgb = flattened_rgb.median(dim=2).values.view(batch_size, 3, 1, 1)
    median_gray = flattened_gray.median(dim=2).values.view(batch_size, 1, 1, 1)

    color_delta = (rgb - median_rgb).abs().mean(dim=1, keepdim=True)
    intensity_delta = (gray - median_gray).abs()
    edge_delta = torch.zeros_like(gray)
    edge_delta[:, :, :, 1:] = torch.maximum(
        edge_delta[:, :, :, 1:],
        (gray[:, :, :, 1:] - gray[:, :, :, :-1]).abs(),
    )
    edge_delta[:, :, 1:, :] = torch.maximum(
        edge_delta[:, :, 1:, :],
        (gray[:, :, 1:, :] - gray[:, :, :-1, :]).abs(),
    )
    foreground_score = color_delta + intensity_delta + 0.5 * edge_delta
    threshold = float(max(0.0, margin))

    red, green, blue = rgb[:, 0], rgb[:, 1], rgb[:, 2]
    max_channel, max_index = rgb.max(dim=1)
    min_channel, _ = rgb.min(dim=1)
    delta = max_channel - min_channel
    eps = 1e-6
    hue_red = torch.remainder((green - blue) / delta.clamp(min=eps), 6.0)
    hue_green = ((blue - red) / delta.clamp(min=eps)) + 2.0
    hue_blue = ((red - green) / delta.clamp(min=eps)) + 4.0
    hue = torch.where(
        max_index == 0,
        hue_red,
        torch.where(max_index == 1, hue_green, hue_blue),
    )
    hue = torch.where(delta > eps, hue / 6.0, torch.zeros_like(hue))
    saturation = torch.where(max_channel > eps, delta / max_channel.clamp(min=eps), torch.zeros_like(max_channel))
    fill_distance = (rgb - mean).abs().mean(dim=1, keepdim=True)
    not_padding = ~((fill_distance < 0.035) & (edge_delta < 0.025))

    height, width = int(images.size(-2)), int(images.size(-1))
    y = torch.linspace(-1.0, 1.0, height, device=device, dtype=dtype).view(1, 1, height, 1)
    x = torch.linspace(-1.0, 1.0, width, device=device, dtype=dtype).view(1, 1, 1, width)
    central_ellipse = ((x / 0.92).pow(2) + (y / 0.96).pow(2)) <= 1.0

    green_yellow = (
        (hue >= 0.08)
        & (hue <= 0.45)
        & (saturation >= 0.07)
        & (max_channel >= 0.12)
    ).unsqueeze(1)
    brown_or_orange = (
        (hue >= 0.035)
        & (hue <= 0.17)
        & (saturation >= 0.10)
        & (max_channel >= 0.10)
    ).unsqueeze(1)
    dark_defect = (
        (max_channel <= 0.45)
        & (saturation >= 0.08)
        & (foreground_score.squeeze(1) > max(0.02, threshold * 0.35))
    ).unsqueeze(1)
    detail_center = central_ellipse & (foreground_score > max(0.035, threshold))
    mask = not_padding & (green_yellow | brown_or_orange | dark_defect | detail_center)
    area = mask.flatten(1).float().mean(dim=1)
    if bool((area > 0.92).any().item()):
        tightened = not_padding & (
            green_yellow
            | brown_or_orange
            | dark_defect
            | (central_ellipse & (foreground_score > max(0.05, threshold * 1.25)))
        )
        mask = torch.where((area > 0.92).view(batch_size, 1, 1, 1), tightened, mask)

    mask_float = mask.to(dtype=torch.float32)
    mask_float = F.max_pool2d(mask_float, kernel_size=5, stride=1, padding=2)
    mask_float = -F.max_pool2d(-mask_float, kernel_size=5, stride=1, padding=2)
    mask = mask_float > 0.5

    # If a crop is almost uniform, keep a central ellipse instead of returning an empty mask.
    area = mask.flatten(1).float().mean(dim=1)
    if bool((area < 0.08).any().item()):
        fallback = not_padding & central_ellipse
        fallback_area = fallback.flatten(1).float().mean(dim=1)
        fallback = torch.where((fallback_area >= 0.03).view(batch_size, 1, 1, 1), fallback, not_padding)
        mask = torch.where((area < 0.08).view(batch_size, 1, 1, 1), fallback, mask)
    return mask.to(dtype=torch.float32)


def _foreground_consistency_loss_from_features(
    *,
    images: Tensor,
    features: Dict[str, Tensor],
    margin: float = 0.08,
) -> Tensor:
    patches = features.get("patches")
    if not torch.is_tensor(patches) or patches.ndim != 3:
        return images.sum() * 0.0
    grid_size = features.get("grid_size")
    if not isinstance(grid_size, tuple) or len(grid_size) != 2:
        grid_side = int(round(math.sqrt(float(patches.size(1)))))
        grid_size = (grid_side, grid_side)
    grid_h, grid_w = int(grid_size[0]), int(grid_size[1])
    if grid_h <= 0 or grid_w <= 0:
        return patches.sum() * 0.0

    with torch.no_grad():
        foreground = _pseudo_foreground_mask_from_normalized_images(images.detach(), margin=margin)
        patch_mask = F.interpolate(foreground, size=(grid_h, grid_w), mode="area")
        patch_mask = patch_mask.flatten(1).clamp(0.0, 1.0)
        patch_indices = features.get("patch_indices")
        if torch.is_tensor(patch_indices) and patch_indices.ndim == 2:
            patch_mask = patch_mask.gather(1, patch_indices.to(device=patch_mask.device, dtype=torch.long))
        elif grid_h * grid_w != int(patches.size(1)):
            return patches.sum() * 0.0
        background_mask = 1.0 - patch_mask

    patch_energy = patches.float().pow(2).mean(dim=-1)
    patch_energy = patch_energy / patch_energy.sum(dim=1, keepdim=True).clamp(min=1e-6)
    background_energy = (patch_energy * background_mask).sum(dim=1)
    foreground_mass = patch_mask.mean(dim=1)
    valid = foreground_mass > 0.05
    if not bool(valid.any().item()):
        return patches.sum() * 0.0
    return background_energy[valid].mean().to(dtype=patches.dtype)


def _normalize_attention_scores(scores: Tensor) -> Tensor:
    flattened = scores.flatten(1)
    minimum = flattened.amin(dim=1, keepdim=True)
    maximum = flattened.amax(dim=1, keepdim=True)
    normalized = (flattened - minimum) / (maximum - minimum).clamp(min=1e-6)
    return normalized.view_as(scores)


def _attention_guided_score_map(
    *,
    images: Tensor,
    features: Dict[str, Tensor],
    foreground_weight: float,
    score_source: str = "learned_attention",
) -> Tensor:
    patches = features.get("patches")
    grid_size = features.get("grid_size")
    if (
        not torch.is_tensor(patches)
        or patches.ndim != 3
        or not isinstance(grid_size, tuple)
        or len(grid_size) != 2
    ):
        return _pseudo_foreground_mask_from_normalized_images(images.detach())

    grid_h, grid_w = int(grid_size[0]), int(grid_size[1])
    original_patch_count = grid_h * grid_w
    if original_patch_count <= 0:
        return _pseudo_foreground_mask_from_normalized_images(images.detach())

    learned_scores = features.get("fine_grained_attention")
    if not torch.is_tensor(learned_scores) or learned_scores.shape[:2] != patches.shape[:2]:
        learned_scores = patches.detach().float().norm(dim=-1)
    else:
        learned_scores = learned_scores.detach().float()
    learned_scores = _normalize_attention_scores(learned_scores)

    patch_indices = features.get("patch_indices")
    if (
        torch.is_tensor(patch_indices)
        and patch_indices.ndim == 2
        and patch_indices.shape == learned_scores.shape
    ):
        spatial_scores = learned_scores.new_zeros(
            (learned_scores.size(0), original_patch_count)
        )
        spatial_scores.scatter_(
            1,
            patch_indices.detach().to(device=learned_scores.device, dtype=torch.long),
            learned_scores,
        )
    elif learned_scores.size(1) == original_patch_count:
        spatial_scores = learned_scores
    else:
        return _pseudo_foreground_mask_from_normalized_images(images.detach())

    foreground_prior = features.get("foreground_prior")
    if (
        torch.is_tensor(foreground_prior)
        and foreground_prior.ndim == 2
        and foreground_prior.shape == spatial_scores.shape
        and bool((foreground_prior.detach().abs().sum() > 0).item())
    ):
        foreground_scores = foreground_prior.detach().float()
    else:
        pseudo_foreground = _pseudo_foreground_mask_from_normalized_images(images.detach())
        foreground_scores = F.interpolate(
            pseudo_foreground,
            size=(grid_h, grid_w),
            mode="area",
        ).flatten(1)

    normalized_learned = _normalize_attention_scores(spatial_scores)
    normalized_foreground = _normalize_attention_scores(foreground_scores)
    detail_map = features.get("detail_map")
    if (
        torch.is_tensor(detail_map)
        and detail_map.ndim == 4
        and detail_map.size(0) == images.size(0)
        and detail_map.size(1) == 1
    ):
        detail_scores = F.adaptive_avg_pool2d(
            detail_map.detach().float(),
            output_size=(grid_h, grid_w),
        ).flatten(1)
        normalized_detail = _normalize_attention_scores(detail_scores)
    else:
        normalized_detail = normalized_learned

    foreground_weight = float(min(1.0, max(0.0, foreground_weight)))
    score_source = str(score_source).strip().lower()
    if score_source == "learned_attention":
        combined = (
            (1.0 - foreground_weight) * normalized_learned
            + foreground_weight * normalized_foreground
        )
    else:
        foreground_gate = (
            (1.0 - foreground_weight)
            + foreground_weight * normalized_foreground
        )
        surface_detail = normalized_detail * foreground_gate
        if score_source == "surface_detail":
            combined = surface_detail
        elif score_source == "hybrid":
            learned_with_prior = (
                (1.0 - foreground_weight) * normalized_learned
                + foreground_weight * normalized_foreground
            )
            combined = 0.35 * learned_with_prior + 0.65 * surface_detail
        else:
            raise ValueError(
                "attention view score source chi ho tro "
                "learned_attention, surface_detail, hybrid."
            )
    combined = combined.view(images.size(0), 1, grid_h, grid_w)
    combined = F.interpolate(
        combined,
        size=images.shape[-2:],
        mode="bilinear",
        align_corners=False,
    )
    return _normalize_attention_scores(combined).clamp(0.0, 1.0)


def _attention_crop_single(
    image: Tensor,
    score_map: Tensor,
    *,
    threshold: float,
    padding_ratio: float,
    min_area_ratio: float,
) -> Tensor:
    height, width = int(image.size(-2)), int(image.size(-1))
    score = score_map.squeeze()
    cutoff = float(threshold) * float(score.max().item())
    mask = score >= cutoff
    nonzero = torch.nonzero(mask, as_tuple=False)
    if nonzero.numel() == 0:
        flat_index = int(score.argmax().item())
        center_y, center_x = divmod(flat_index, width)
        y1 = y2 = center_y
        x1 = x2 = center_x
    else:
        y1 = int(nonzero[:, 0].min().item())
        y2 = int(nonzero[:, 0].max().item())
        x1 = int(nonzero[:, 1].min().item())
        x2 = int(nonzero[:, 1].max().item())

    pad_y = int(round(float(padding_ratio) * height))
    pad_x = int(round(float(padding_ratio) * width))
    y1, y2 = max(0, y1 - pad_y), min(height - 1, y2 + pad_y)
    x1, x2 = max(0, x1 - pad_x), min(width - 1, x2 + pad_x)

    min_scale = math.sqrt(float(min(1.0, max(1e-4, min_area_ratio))))
    min_height = max(1, int(math.ceil(height * min_scale)))
    min_width = max(1, int(math.ceil(width * min_scale)))
    center_y = (y1 + y2) // 2
    center_x = (x1 + x2) // 2
    if y2 - y1 + 1 < min_height:
        y1 = center_y - min_height // 2
        y2 = y1 + min_height - 1
    if x2 - x1 + 1 < min_width:
        x1 = center_x - min_width // 2
        x2 = x1 + min_width - 1
    if y1 < 0:
        y2 -= y1
        y1 = 0
    if x1 < 0:
        x2 -= x1
        x1 = 0
    if y2 >= height:
        y1 -= y2 - height + 1
        y2 = height - 1
    if x2 >= width:
        x1 -= x2 - width + 1
        x2 = width - 1
    y1, x1 = max(0, y1), max(0, x1)

    crop = image[:, y1 : y2 + 1, x1 : x2 + 1].unsqueeze(0)
    return F.interpolate(
        crop,
        size=(height, width),
        mode="bilinear",
        align_corners=False,
    ).squeeze(0)


def _bounded_attention_drop_mask(
    score_map: Tensor,
    *,
    threshold: float,
    dilation_kernel: int,
    min_area_ratio: float,
    max_area_ratio: float,
) -> Tensor:
    if score_map.ndim != 4 or score_map.size(1) != 1:
        raise ValueError("score_map cho attention drop phai co shape [B, 1, H, W].")

    batch_size = int(score_map.size(0))
    flat_scores = score_map.detach().float().flatten(1)
    pixel_count = int(flat_scores.size(1))
    if batch_size == 0 or pixel_count == 0:
        return torch.zeros_like(score_map)

    min_count = max(1, int(math.ceil(pixel_count * float(min_area_ratio))))
    max_count = max(min_count, int(math.floor(pixel_count * float(max_area_ratio))))
    max_count = min(pixel_count, max_count)
    cutoffs = (
        float(threshold)
        * flat_scores.amax(dim=1, keepdim=True)
    )
    masks = (flat_scores >= cutoffs).view_as(score_map)

    dilation_kernel = max(1, int(dilation_kernel))
    if dilation_kernel > 1:
        masks = F.max_pool2d(
            masks.to(dtype=score_map.dtype),
            kernel_size=dilation_kernel,
            stride=1,
            padding=dilation_kernel // 2,
        ) > 0

    bounded_masks: List[Tensor] = []
    for sample_index in range(batch_size):
        sample_mask = masks[sample_index].flatten()
        selected_count = int(sample_mask.sum().item())
        if selected_count < min_count or selected_count > max_count:
            target_count = min_count if selected_count < min_count else max_count
            top_indices = torch.topk(
                flat_scores[sample_index],
                k=target_count,
                largest=True,
                sorted=False,
            ).indices
            sample_mask = torch.zeros_like(sample_mask, dtype=torch.bool)
            sample_mask.scatter_(0, top_indices, True)
        bounded_masks.append(sample_mask.view(1, score_map.size(-2), score_map.size(-1)))

    return torch.stack(bounded_masks, dim=0).to(dtype=score_map.dtype)


def _build_attention_guided_views(
    *,
    images: Tensor,
    features: Dict[str, Tensor],
    crop_probability: float,
    drop_probability: float,
    crop_threshold: float,
    drop_threshold: float,
    crop_padding_ratio: float,
    crop_min_area_ratio: float,
    foreground_weight: float,
    drop_blur_kernel: int,
    score_source: str = "learned_attention",
    drop_dilation_kernel: int = 5,
    drop_min_area_ratio: float = 0.06,
    drop_max_area_ratio: float = 0.16,
) -> Tuple[Optional[Tensor], Tensor, Dict[str, float]]:
    batch_size = int(images.size(0))
    empty_indices = torch.zeros((0,), device=images.device, dtype=torch.long)
    if batch_size == 0 or crop_probability + drop_probability <= 0.0:
        return None, empty_indices, {
            "attention_view_fraction": 0.0,
            "attention_crop_fraction": 0.0,
            "attention_drop_fraction": 0.0,
            "attention_drop_area_fraction": 0.0,
        }

    with torch.no_grad():
        score_maps = _attention_guided_score_map(
            images=images,
            features=features,
            foreground_weight=foreground_weight,
            score_source=score_source,
        )
        draws = torch.rand((batch_size,), device=images.device)
        crop_mask = draws < float(crop_probability)
        drop_mask = (
            (draws >= float(crop_probability))
            & (draws < float(crop_probability + drop_probability))
        )
        selected = crop_mask | drop_mask
        selected_indices = torch.nonzero(selected, as_tuple=False).flatten()
        if selected_indices.numel() == 0:
            return None, empty_indices, {
                "attention_view_fraction": 0.0,
                "attention_crop_fraction": 0.0,
                "attention_drop_fraction": 0.0,
                "attention_drop_area_fraction": 0.0,
            }

        blur_kernel = max(1, int(drop_blur_kernel))
        blurred = F.avg_pool2d(
            images,
            kernel_size=blur_kernel,
            stride=1,
            padding=blur_kernel // 2,
        )
        views: List[Tensor] = []
        drop_area_fractions: List[float] = []
        for sample_index in selected_indices.tolist():
            if bool(crop_mask[sample_index].item()):
                view = _attention_crop_single(
                    images[sample_index],
                    score_maps[sample_index],
                    threshold=crop_threshold,
                    padding_ratio=crop_padding_ratio,
                    min_area_ratio=crop_min_area_ratio,
                )
            else:
                score = score_maps[sample_index : sample_index + 1]
                salient = _bounded_attention_drop_mask(
                    score,
                    threshold=drop_threshold,
                    dilation_kernel=drop_dilation_kernel,
                    min_area_ratio=drop_min_area_ratio,
                    max_area_ratio=drop_max_area_ratio,
                ).to(dtype=images.dtype)
                drop_area_fractions.append(float(salient.float().mean().item()))
                view = (
                    images[sample_index : sample_index + 1] * (1.0 - salient)
                    + blurred[sample_index : sample_index + 1] * salient
                ).squeeze(0)
            views.append(view)

        return torch.stack(views, dim=0), selected_indices, {
            "attention_view_fraction": float(selected.float().mean().item()),
            "attention_crop_fraction": float(crop_mask.float().mean().item()),
            "attention_drop_fraction": float(drop_mask.float().mean().item()),
            "attention_drop_area_fraction": (
                float(sum(drop_area_fractions) / len(drop_area_fractions))
                if drop_area_fractions
                else 0.0
            ),
        }


def _register_diversity_loss_from_features(features: Dict[str, Tensor]) -> Tensor:
    registers = features.get("registers")
    if not torch.is_tensor(registers) or registers.ndim != 3 or registers.size(1) < 2:
        reference = next(
            (value for value in features.values() if torch.is_tensor(value)),
            None,
        )
        if reference is None:
            return torch.tensor(0.0)
        return reference.sum() * 0.0

    normalized = F.normalize(registers.float(), dim=-1, eps=1e-6)
    similarity = torch.matmul(normalized, normalized.transpose(1, 2))
    register_count = int(registers.size(1))
    off_diagonal = ~torch.eye(
        register_count,
        device=registers.device,
        dtype=torch.bool,
    ).unsqueeze(0)
    loss = similarity.square().masked_select(off_diagonal.expand_as(similarity)).mean()
    return loss.to(dtype=registers.dtype)


def _pairwise_margin_loss_from_features(
    *,
    model: nn.Module,
    features: Dict[str, Tensor],
    targets: Tensor,
) -> Tensor:
    pairwise_logits = features.get("pairwise_margin_logits")
    pairs = getattr(model, "pairwise_margin_pairs", [])
    if not torch.is_tensor(pairwise_logits) or pairwise_logits.ndim != 2 or not pairs:
        reference = next((value for value in features.values() if torch.is_tensor(value)), targets)
        return reference.sum() * 0.0
    target_indices = targets.to(device=pairwise_logits.device, dtype=torch.long).view(-1)
    losses: List[Tensor] = []
    for pair_index, pair in enumerate(pairs):
        if pair_index >= int(pairwise_logits.size(1)):
            break
        left_class, right_class = [int(value) for value in pair]
        logits = pairwise_logits[:, pair_index].float()
        if left_class < 0:
            if right_class < 0:
                continue
            binary_targets = (target_indices == right_class).to(dtype=logits.dtype)
            losses.append(F.binary_cross_entropy_with_logits(logits, binary_targets))
            continue
        pair_mask = (target_indices == left_class) | (target_indices == right_class)
        if not bool(pair_mask.any().item()):
            continue
        binary_targets = (target_indices[pair_mask] == right_class).to(dtype=logits.dtype)
        losses.append(F.binary_cross_entropy_with_logits(logits[pair_mask], binary_targets))
    if not losses:
        return pairwise_logits.sum() * 0.0
    return torch.stack(losses).mean().to(dtype=pairwise_logits.dtype)


def _ordinal_maturity_loss_from_features(
    *,
    model: nn.Module,
    features: Dict[str, Tensor],
    targets: Tensor,
) -> Tensor:
    maturity_score = features.get("ordinal_maturity_score")
    maturity_classes = [
        int(value)
        for value in getattr(model, "ordinal_maturity_classes", [])
    ]
    if (
        not torch.is_tensor(maturity_score)
        or maturity_score.ndim != 2
        or maturity_score.size(1) != 1
        or len(maturity_classes) < 2
    ):
        reference = next((value for value in features.values() if torch.is_tensor(value)), targets)
        return reference.sum() * 0.0

    target_indices = targets.to(device=maturity_score.device, dtype=torch.long).view(-1)
    valid = torch.zeros_like(target_indices, dtype=torch.bool)
    target_scores = torch.zeros_like(target_indices, dtype=torch.float32)
    centered_ranks = torch.arange(
        len(maturity_classes),
        device=maturity_score.device,
        dtype=torch.float32,
    )
    centered_ranks = centered_ranks - centered_ranks.mean()
    for rank, class_index in enumerate(maturity_classes):
        class_mask = target_indices == int(class_index)
        valid = valid | class_mask
        target_scores[class_mask] = centered_ranks[rank]
    if not bool(valid.any().item()):
        return maturity_score.sum() * 0.0

    predicted = maturity_score[:, 0].float()[valid]
    expected = target_scores[valid]
    return F.smooth_l1_loss(predicted, expected).to(dtype=maturity_score.dtype)


def _build_pretrained_distillation_teacher(
    *,
    checkpoint_path: Path,
    target_class_names: Sequence[str],
    device: torch.device,
) -> Tuple[nn.Module, Tensor, Dict[str, object]]:
    try:
        import timm
    except ImportError as exc:  # pragma: no cover - environment dependent.
        raise RuntimeError(
            "Pretrained distillation teacher yeu cau package timm."
        ) from exc

    checkpoint = torch.load(checkpoint_path, map_location="cpu", weights_only=False)
    if not isinstance(checkpoint, dict):
        raise ValueError(f"Teacher checkpoint khong hop le: {checkpoint_path}")
    teacher_args = checkpoint.get("args", {})
    teacher_classes = [str(value) for value in checkpoint.get("classes", [])]
    teacher_model_name = str(
        teacher_args.get("model", "")
        if isinstance(teacher_args, dict)
        else ""
    ).strip()
    teacher_state = checkpoint.get("model")
    if not teacher_model_name or not isinstance(teacher_state, dict):
        raise ValueError(
            "Teacher checkpoint phai co args.model va model state theo format baseline TIMM."
        )
    missing_classes = sorted(set(target_class_names) - set(teacher_classes))
    if missing_classes or len(teacher_classes) != len(target_class_names):
        raise ValueError(
            "Teacher class names khong khop dataset: "
            f"teacher={teacher_classes} target={list(target_class_names)}"
        )

    teacher = timm.create_model(
        teacher_model_name,
        pretrained=False,
        num_classes=len(teacher_classes),
    )
    teacher.load_state_dict(teacher_state, strict=True)
    teacher_total_parameters = sum(
        int(parameter.numel()) for parameter in teacher.parameters()
    )
    teacher.eval().requires_grad_(False).to(device)
    target_to_teacher = torch.tensor(
        [teacher_classes.index(str(class_name)) for class_name in target_class_names],
        dtype=torch.long,
        device=device,
    )
    summary = {
        "enabled": True,
        "checkpoint": str(Path(checkpoint_path).resolve()),
        "model": teacher_model_name,
        "teacher_classes": teacher_classes,
        "target_classes": list(target_class_names),
        "target_to_teacher_indices": target_to_teacher.detach().cpu().tolist(),
        "parameter_count": teacher_total_parameters,
        "trainable_parameters": sum(
            int(parameter.numel())
            for parameter in teacher.parameters()
            if parameter.requires_grad
        ),
    }
    return teacher, target_to_teacher, summary


def _load_offline_distillation_probabilities(
    *,
    csv_path: Path,
    num_classes: int,
    class_names: Sequence[str],
) -> Tuple[Dict[str, List[float]], Dict[str, object]]:
    csv_path = Path(csv_path)
    if not csv_path.is_file():
        raise FileNotFoundError(f"Khong tim thay teacher probability CSV: {csv_path}")
    probabilities_by_path: Dict[str, List[float]] = {}
    confidence_sum = 0.0
    pred_counts = [0 for _ in range(max(1, int(num_classes)))]
    duplicate_count = 0
    with csv_path.open("r", newline="", encoding="utf-8") as handle:
        reader = csv.DictReader(handle)
        if reader.fieldnames is None:
            raise ValueError(f"Teacher probability CSV rong: {csv_path}")
        required = {"path", *[f"prob_{index}" for index in range(int(num_classes))]}
        missing = required.difference(reader.fieldnames)
        if missing:
            raise ValueError(
                f"Teacher probability CSV thieu cot {sorted(missing)}: {csv_path}"
            )
        for row in reader:
            path_text = str(row.get("path", "") or "").strip()
            if not path_text:
                continue
            values = [float(row[f"prob_{index}"]) for index in range(int(num_classes))]
            probabilities = torch.tensor(values, dtype=torch.float32).clamp(min=0.0)
            total = probabilities.sum().clamp(min=1e-12)
            probabilities = probabilities / total
            if not torch.isfinite(probabilities).all():
                raise ValueError(f"Teacher probability CSV co prob khong huu han: {path_text}")
            key = str(Path(path_text).resolve()).lower()
            if key in probabilities_by_path:
                duplicate_count += 1
            probability_values = [float(value) for value in probabilities.tolist()]
            probabilities_by_path[key] = probability_values
            pred_index = int(probabilities.argmax().item())
            if 0 <= pred_index < len(pred_counts):
                pred_counts[pred_index] += 1
            confidence_sum += float(probabilities.max().item())
    if not probabilities_by_path:
        raise ValueError(f"Teacher probability CSV khong co dong hop le: {csv_path}")
    sample_count = len(probabilities_by_path)
    summary = {
        "enabled": True,
        "csv": str(csv_path.resolve()),
        "samples": sample_count,
        "duplicate_rows": duplicate_count,
        "num_classes": int(num_classes),
        "class_names": list(class_names),
        "mean_confidence": float(confidence_sum / max(1, sample_count)),
        "predicted_class_counts": pred_counts,
    }
    return probabilities_by_path, summary


def _distillation_loss(
    *,
    student_logits: Tensor,
    teacher_logits: Tensor,
    targets,
    temperature: float,
    focus_class_index: int,
    focus_class_weight: float,
) -> Tensor:
    temperature = max(1e-4, float(temperature))
    teacher_probabilities = F.softmax(
        teacher_logits.float() / temperature,
        dim=1,
    )
    student_log_probabilities = F.log_softmax(
        student_logits.float() / temperature,
        dim=1,
    )
    teacher_log_probabilities = torch.log(
        teacher_probabilities.clamp(min=1e-8)
    )
    per_sample = (
        teacher_probabilities
        * (teacher_log_probabilities - student_log_probabilities)
    ).sum(dim=1)
    target_indices = _classification_target_indices(targets, student_logits)
    if (
        target_indices is not None
        and 0 <= int(focus_class_index) < int(student_logits.size(1))
        and float(focus_class_weight) != 1.0
    ):
        sample_weights = torch.ones_like(per_sample)
        sample_weights[target_indices == int(focus_class_index)] = float(
            focus_class_weight
        )
        per_sample = per_sample * sample_weights
        loss = per_sample.sum() / sample_weights.sum().clamp(min=1.0)
    else:
        loss = per_sample.mean()
    return loss.to(dtype=student_logits.dtype) * (temperature * temperature)


def _probability_distillation_loss(
    *,
    student_logits: Tensor,
    teacher_probabilities: Tensor,
    targets,
    temperature: float,
    focus_class_index: int,
    focus_class_weight: float,
) -> Tensor:
    temperature = max(1e-4, float(temperature))
    teacher_probabilities = teacher_probabilities.to(
        device=student_logits.device,
        dtype=torch.float32,
    ).clamp(min=0.0)
    teacher_probabilities = teacher_probabilities / teacher_probabilities.sum(
        dim=1,
        keepdim=True,
    ).clamp(min=1e-12)
    student_log_probabilities = F.log_softmax(
        student_logits.float() / temperature,
        dim=1,
    )
    teacher_log_probabilities = torch.log(
        teacher_probabilities.clamp(min=1e-8)
    )
    per_sample = (
        teacher_probabilities
        * (teacher_log_probabilities - student_log_probabilities)
    ).sum(dim=1)
    target_indices = _classification_target_indices(targets, student_logits)
    if (
        target_indices is not None
        and 0 <= int(focus_class_index) < int(student_logits.size(1))
        and float(focus_class_weight) != 1.0
    ):
        sample_weights = torch.ones_like(per_sample)
        sample_weights[target_indices == int(focus_class_index)] = float(
            focus_class_weight
        )
        per_sample = per_sample * sample_weights
        loss = per_sample.sum() / sample_weights.sum().clamp(min=1.0)
    else:
        loss = per_sample.mean()
    return loss.to(dtype=student_logits.dtype) * (temperature * temperature)


def _early_learning_regularization_loss(
    logits: Tensor,
    sample_indices: Optional[Tensor],
    target_state: Optional[Tensor],
    *,
    beta: float,
    update_state: bool = True,
) -> Tensor:
    zero = logits.sum() * 0.0
    if (
        target_state is None
        or sample_indices is None
        or not torch.is_tensor(sample_indices)
        or logits.ndim != 2
        or target_state.ndim != 2
        or target_state.size(1) != logits.size(1)
    ):
        return zero
    flat_indices = sample_indices.to(device=target_state.device, dtype=torch.long).view(-1)
    if flat_indices.numel() != logits.size(0):
        return zero
    valid_mask = (flat_indices >= 0) & (flat_indices < int(target_state.size(0)))
    if not bool(valid_mask.any().item()):
        return zero
    valid_indices = flat_indices[valid_mask]
    valid_logits = logits[valid_mask.to(device=logits.device)]
    with torch.no_grad():
        probabilities = F.softmax(valid_logits.detach().float(), dim=1).to(
            device=target_state.device,
            dtype=target_state.dtype,
        )
        previous = target_state.index_select(0, valid_indices)
        updated = float(beta) * previous + (1.0 - float(beta)) * probabilities
        updated = updated / updated.sum(dim=1, keepdim=True).clamp(min=1e-12)
        if update_state:
            target_state.index_copy_(0, valid_indices, updated)
        target = updated.detach().to(device=logits.device, dtype=torch.float32)
    student_probabilities = F.softmax(valid_logits.float(), dim=1)
    agreement = (student_probabilities * target).sum(dim=1).clamp(min=1e-6, max=1.0 - 1e-6)
    return torch.log1p(-agreement).mean().to(dtype=logits.dtype)


def _classification_loss_with_sample_weights(
    criterion: nn.Module,
    logits: Tensor,
    targets,
    sample_weights: Optional[Tensor],
) -> Tuple[Tensor, Tensor]:
    if sample_weights is None or not torch.is_tensor(sample_weights):
        return criterion(logits, targets), logits.new_tensor(1.0)
    weights = sample_weights.to(device=logits.device, dtype=logits.dtype).view(-1)
    if weights.numel() != logits.size(0):
        return criterion(logits, targets), logits.new_tensor(1.0)
    weights = weights.clamp(min=0.0)
    if float(weights.sum().detach().cpu().item()) <= 0.0:
        return criterion(logits, targets), logits.new_tensor(0.0)

    per_sample_loss_fn = getattr(criterion, "per_sample_loss", None)
    if callable(per_sample_loss_fn):
        per_sample = per_sample_loss_fn(logits, targets)
    elif isinstance(criterion, nn.CrossEntropyLoss) and torch.is_tensor(targets):
        ce_weight = criterion.weight
        if torch.is_tensor(ce_weight):
            ce_weight = ce_weight.to(device=logits.device, dtype=logits.dtype)
        per_sample = F.cross_entropy(
            logits,
            targets.to(device=logits.device, dtype=torch.long),
            weight=ce_weight,
            ignore_index=criterion.ignore_index,
            reduction="none",
            label_smoothing=float(getattr(criterion, "label_smoothing", 0.0) or 0.0),
        )
    else:
        return criterion(logits, targets), weights.mean().detach()
    if per_sample.ndim != 1 or per_sample.numel() != weights.numel():
        return criterion(logits, targets), weights.mean().detach()
    weighted_loss = (per_sample * weights).sum() / weights.sum().clamp(min=1e-12)
    return weighted_loss.to(dtype=logits.dtype), weights.mean().detach()


def _stack_image_masks_from_targets(targets) -> Optional[Tensor]:
    if not _is_detection_targets(targets):
        return None
    masks = []
    for target in targets:
        image_mask = target.get("image_mask")
        if not torch.is_tensor(image_mask):
            return None
        masks.append(image_mask.to(dtype=torch.bool))
    if not masks:
        return None
    return torch.stack(masks, dim=0)


def _forward_train_loss(
    model: nn.Module,
    criterion: nn.Module,
    images: Tensor,
    labels: Optional[Tensor],
    targets,
    device: torch.device,
    amp: bool,
    debug_bbox: bool = False,
    metric_learning_criterion: Optional[nn.Module] = None,
    metric_learning_loss_weight: float = 0.0,
    metric_learning_sources: Union[str, Sequence[str]] = "head",
    boundary_contrastive_loss_weight: float = 0.0,
    boundary_contrastive_pairs: str = "0-1,1-2,2-3,4-rest",
    boundary_contrastive_sources: Union[str, Sequence[str]] = "head,patch",
    boundary_contrastive_margin: float = 0.12,
    boundary_contrastive_temperature: float = 0.20,
    boundary_contrastive_max_pairs: int = 128,
    foreground_consistency_loss_weight: float = 0.0,
    foreground_consistency_margin: float = 0.08,
    attention_view_loss_weight: float = 0.0,
    attention_crop_probability: float = 0.50,
    attention_drop_probability: float = 0.25,
    attention_view_start_epoch: int = 2,
    attention_crop_threshold: float = 0.55,
    attention_drop_threshold: float = 0.70,
    attention_crop_padding_ratio: float = 0.08,
    attention_crop_min_area_ratio: float = 0.20,
    attention_view_foreground_weight: float = 0.35,
    attention_view_score_source: str = "learned_attention",
    attention_drop_blur_kernel: int = 15,
    attention_drop_dilation_kernel: int = 5,
    attention_drop_min_area_ratio: float = 0.06,
    attention_drop_max_area_ratio: float = 0.16,
    epoch_index: int = 0,
    register_diversity_loss_weight: float = 0.0,
    pairwise_margin_loss_weight: float = 0.0,
    ordinal_maturity_loss_weight: float = 0.0,
    distillation_teacher: Optional[nn.Module] = None,
    distillation_class_indices: Optional[Tensor] = None,
    distillation_loss_weight: float = 0.0,
    distillation_temperature: float = 2.0,
    distillation_focus_class_index: int = 1,
    distillation_focus_class_weight: float = 1.0,
    offline_teacher_probabilities: Optional[Tensor] = None,
    sample_indices: Optional[Tensor] = None,
    sample_weights: Optional[Tensor] = None,
    elr_target_state: Optional[Tensor] = None,
    elr_loss_weight: float = 0.0,
    elr_beta: float = 0.70,
    elr_start_epoch: int = 2,
    elr_update_state: bool = True,
) -> Tuple[Tensor, Optional[Dict[str, Tensor]], Tensor, Dict[str, float], Optional[Tensor]]:
    with autocast_context(device, amp):
        image_valid_mask = _stack_image_masks_from_targets(targets)
        features, outputs = _forward_model_outputs(model, images, image_valid_mask=image_valid_mask)
        logits, pred_boxes = extract_bbox_from_model_output(outputs)
        if _is_detection_targets(targets):
            loss, loss_details = criterion(outputs, targets, return_details=True, debug_bbox=debug_bbox)
        else:
            loss, sample_weight_mean = _classification_loss_with_sample_weights(
                criterion,
                logits,
                targets,
                sample_weights,
            )
            metric_learning_loss = logits.sum() * 0.0
            boundary_contrastive_loss = logits.sum() * 0.0
            boundary_contrastive_terms = 0
            foreground_consistency_loss = logits.sum() * 0.0
            attention_view_loss = logits.sum() * 0.0
            attention_view_stats = {
                "attention_view_fraction": 0.0,
                "attention_crop_fraction": 0.0,
                "attention_drop_fraction": 0.0,
                "attention_drop_area_fraction": 0.0,
            }
            register_diversity_loss = logits.sum() * 0.0
            pairwise_margin_loss = logits.sum() * 0.0
            ordinal_maturity_loss = logits.sum() * 0.0
            distillation_loss = logits.sum() * 0.0
            elr_loss = logits.sum() * 0.0
            if (
                metric_learning_criterion is not None
                and float(metric_learning_loss_weight) > 0.0
                and features is not None
            ):
                target_indices = _classification_target_indices(targets, logits)
                if target_indices is not None:
                    metric_learning_loss, _ = _metric_learning_loss_from_features(
                        model=model,
                        features=features,
                        targets=target_indices,
                        criterion=metric_learning_criterion,
                        sources=metric_learning_sources,
                    )
                    loss = loss + float(metric_learning_loss_weight) * metric_learning_loss
            if (
                features is not None
                and float(boundary_contrastive_loss_weight) > 0.0
                and torch.is_tensor(targets)
            ):
                target_indices = _classification_target_indices(targets, logits)
                if target_indices is not None:
                    boundary_contrastive_loss, boundary_contrastive_stats = (
                        _boundary_contrastive_loss_from_features(
                            model=model,
                            features=features,
                            targets=target_indices,
                            pairs=boundary_contrastive_pairs,
                            sources=boundary_contrastive_sources,
                            margin=boundary_contrastive_margin,
                            temperature=boundary_contrastive_temperature,
                            max_pairs=boundary_contrastive_max_pairs,
                            num_classes=int(logits.size(1)),
                        )
                    )
                    boundary_contrastive_terms = int(
                        boundary_contrastive_stats.get("terms", 0) or 0
                    )
                    loss = loss + float(boundary_contrastive_loss_weight) * boundary_contrastive_loss
            if features is not None and float(foreground_consistency_loss_weight) > 0.0:
                foreground_consistency_loss = _foreground_consistency_loss_from_features(
                    images=images,
                    features=features,
                    margin=foreground_consistency_margin,
                )
                loss = loss + float(foreground_consistency_loss_weight) * foreground_consistency_loss
            if (
                features is not None
                and torch.is_tensor(targets)
                and float(attention_view_loss_weight) > 0.0
                and int(epoch_index) >= int(attention_view_start_epoch)
            ):
                attention_views, selected_indices, attention_view_stats = (
                    _build_attention_guided_views(
                        images=images,
                        features=features,
                        crop_probability=attention_crop_probability,
                        drop_probability=attention_drop_probability,
                        crop_threshold=attention_crop_threshold,
                        drop_threshold=attention_drop_threshold,
                        crop_padding_ratio=attention_crop_padding_ratio,
                        crop_min_area_ratio=attention_crop_min_area_ratio,
                        foreground_weight=attention_view_foreground_weight,
                        drop_blur_kernel=attention_drop_blur_kernel,
                        score_source=attention_view_score_source,
                        drop_dilation_kernel=attention_drop_dilation_kernel,
                        drop_min_area_ratio=attention_drop_min_area_ratio,
                        drop_max_area_ratio=attention_drop_max_area_ratio,
                    )
                )
                if attention_views is not None and selected_indices.numel() > 0:
                    _, attention_outputs = _forward_model_outputs(
                        model,
                        attention_views,
                        image_valid_mask=None,
                    )
                    attention_logits, _ = extract_bbox_from_model_output(attention_outputs)
                    attention_targets = targets.index_select(
                        0,
                        selected_indices.to(device=targets.device),
                    )
                    attention_view_loss = criterion(attention_logits, attention_targets)
                    loss = loss + float(attention_view_loss_weight) * attention_view_loss
            if features is not None and float(register_diversity_loss_weight) > 0.0:
                register_diversity_loss = _register_diversity_loss_from_features(features)
                loss = loss + float(register_diversity_loss_weight) * register_diversity_loss
            if features is not None and float(pairwise_margin_loss_weight) > 0.0:
                target_indices = _classification_target_indices(targets, logits)
                if target_indices is not None:
                    pairwise_margin_loss = _pairwise_margin_loss_from_features(
                        model=model,
                        features=features,
                        targets=target_indices,
                    )
                    loss = loss + float(pairwise_margin_loss_weight) * pairwise_margin_loss
            if features is not None and float(ordinal_maturity_loss_weight) > 0.0:
                target_indices = _classification_target_indices(targets, logits)
                if target_indices is not None:
                    ordinal_maturity_loss = _ordinal_maturity_loss_from_features(
                        model=model,
                        features=features,
                        targets=target_indices,
                    )
                    loss = loss + float(ordinal_maturity_loss_weight) * ordinal_maturity_loss
            if (
                distillation_teacher is not None
                and distillation_class_indices is not None
                and float(distillation_loss_weight) > 0.0
            ):
                with torch.no_grad():
                    teacher_logits = distillation_teacher(images)
                    if isinstance(teacher_logits, (tuple, list)):
                        teacher_logits = teacher_logits[0]
                    if isinstance(teacher_logits, dict):
                        teacher_logits = teacher_logits.get("logits")
                    if not torch.is_tensor(teacher_logits) or teacher_logits.ndim != 2:
                        raise ValueError("Teacher phai tra ve classification logits [B, C].")
                    teacher_logits = teacher_logits.index_select(
                        1,
                        distillation_class_indices,
                    )
                distillation_loss = _distillation_loss(
                    student_logits=logits,
                    teacher_logits=teacher_logits,
                    targets=targets,
                    temperature=distillation_temperature,
                    focus_class_index=distillation_focus_class_index,
                    focus_class_weight=distillation_focus_class_weight,
                )
                loss = loss + float(distillation_loss_weight) * distillation_loss
            if (
                offline_teacher_probabilities is not None
                and float(distillation_loss_weight) > 0.0
            ):
                offline_distillation_loss = _probability_distillation_loss(
                    student_logits=logits,
                    teacher_probabilities=offline_teacher_probabilities,
                    targets=targets,
                    temperature=distillation_temperature,
                    focus_class_index=distillation_focus_class_index,
                    focus_class_weight=distillation_focus_class_weight,
                )
                distillation_loss = distillation_loss + offline_distillation_loss
                loss = loss + float(distillation_loss_weight) * offline_distillation_loss
            if (
                elr_target_state is not None
                and sample_indices is not None
                and float(elr_loss_weight) > 0.0
                and torch.is_tensor(targets)
            ):
                elr_loss = _early_learning_regularization_loss(
                    logits,
                    sample_indices,
                    elr_target_state,
                    beta=elr_beta,
                    update_state=elr_update_state,
                )
                if int(epoch_index) >= int(elr_start_epoch):
                    loss = loss + float(elr_loss_weight) * elr_loss
            loss_details = {
                "loss": float(loss.detach().cpu().item()),
                "cls_loss": float(
                    (
                        loss
                        - float(metric_learning_loss_weight) * metric_learning_loss
                        - float(boundary_contrastive_loss_weight) * boundary_contrastive_loss
                        - float(foreground_consistency_loss_weight) * foreground_consistency_loss
                        - float(attention_view_loss_weight) * attention_view_loss
                        - float(register_diversity_loss_weight) * register_diversity_loss
                        - float(pairwise_margin_loss_weight) * pairwise_margin_loss
                        - float(ordinal_maturity_loss_weight) * ordinal_maturity_loss
                        - float(distillation_loss_weight) * distillation_loss
                        - (
                            float(elr_loss_weight) * elr_loss
                            if int(epoch_index) >= int(elr_start_epoch)
                            else 0.0
                        )
                    )
                    .detach()
                    .cpu()
                    .item()
                ),
                "metric_learning_loss": float(metric_learning_loss.detach().cpu().item()),
                "boundary_contrastive_loss": float(boundary_contrastive_loss.detach().cpu().item()),
                "boundary_contrastive_terms": float(boundary_contrastive_terms),
                "foreground_consistency_loss": float(foreground_consistency_loss.detach().cpu().item()),
                "attention_view_loss": float(attention_view_loss.detach().cpu().item()),
                **attention_view_stats,
                "register_diversity_loss": float(register_diversity_loss.detach().cpu().item()),
                "pairwise_margin_loss": float(pairwise_margin_loss.detach().cpu().item()),
                "ordinal_maturity_loss": float(ordinal_maturity_loss.detach().cpu().item()),
                "distillation_loss": float(distillation_loss.detach().cpu().item()),
                "elr_loss": float(elr_loss.detach().cpu().item()),
                "sample_weight_mean": float(sample_weight_mean.detach().cpu().item()),
                "objectness_loss": 0.0,
                "bbox_l1_loss": 0.0,
                "bbox_giou_loss": 0.0,
                "cardinality_loss": 0.0,
                "count_loss": 0.0,
                "quality_loss": 0.0,
                "count_objectness_consistency_loss": 0.0,
                "auxiliary_loss": 0.0,
            }
    return loss, features, logits, loss_details, pred_boxes


def _clone_batch_for_replay(batch) -> Tuple[object, ...]:
    return tuple(_clone_batch_item(item) for item in batch)


def _first_nonfinite_parameter(model: nn.Module) -> Optional[Tuple[str, int]]:
    for name, parameter in model.named_parameters():
        if parameter is None:
            continue
        data = parameter.detach()
        if not torch.isfinite(data).all():
            return name, int((~torch.isfinite(data)).sum().item())
    return None


def _first_nonfinite_gradient(model: nn.Module) -> Optional[Tuple[str, int]]:
    for name, parameter in model.named_parameters():
        grad = parameter.grad
        if grad is None:
            continue
        finite_mask = torch.isfinite(grad.detach())
        if not bool(finite_mask.all().item()):
            return name, int((~finite_mask).sum().item())
    return None


def _raise_if_model_parameters_nonfinite(model: nn.Module, context: str) -> None:
    bad_parameter = _first_nonfinite_parameter(model)
    if bad_parameter is None:
        return
    name, invalid_count = bad_parameter
    raise RuntimeError(
        "Non-finite model parameter detected after optimizer update; "
        f"context={context} parameter={name} invalid_entries={invalid_count}. "
        "Dung checkpoint best.pt gan nhat, khong dung last.pt/interrupt.pt cua run nay."
    )


def _clip_gradients_and_check(
    model: nn.Module,
    *,
    max_norm: float,
) -> Tuple[bool, float]:
    if _first_nonfinite_gradient(model) is not None:
        return False, float("inf")
    total_norm = torch.nn.utils.clip_grad_norm_(
        model.parameters(),
        max_norm=max_norm,
        error_if_nonfinite=False,
    )
    if torch.is_tensor(total_norm):
        is_finite = bool(torch.isfinite(total_norm).item())
        norm_value = float(torch.nan_to_num(total_norm.detach(), nan=float("inf")).cpu().item())
    else:
        norm_value = float(total_norm)
        is_finite = math.isfinite(norm_value)
    return is_finite, norm_value


def _nonfinite_gradient_context(model: nn.Module) -> Dict[str, object]:
    bad_gradient = _first_nonfinite_gradient(model)
    if bad_gradient is None:
        return {}
    name, invalid_count = bad_gradient
    return {"parameter": name, "invalid_grad_entries": int(invalid_count)}


def _raise_if_nonfinite_gradient_streak_exceeded(
    *,
    streak: int,
    limit: int,
    epoch_index: int,
    batch_index: int,
    grad_norm: float,
    gradient_context: Dict[str, object],
) -> None:
    if int(limit) <= 0 or int(streak) < int(limit):
        return
    raise RuntimeError(
        "Gradient norm became non-finite for too many optimizer steps in a row; "
        f"epoch={epoch_index} batch={batch_index} streak={streak} limit={limit} "
        f"grad_norm={grad_norm} gradient_context={gradient_context}. "
        "Resume from checkpoints/best.pt, reset optimizer/scheduler/scaler, "
        "and reduce LR or auxiliary/count/quality loss weights before continuing."
    )


class ModelEMA:
    def __init__(
        self,
        model: nn.Module,
        *,
        decay: float,
        updates: int = 0,
    ) -> None:
        self.module = copy.deepcopy(model).eval()
        self.module.requires_grad_(False)
        self.decay = float(decay)
        self.updates = max(0, int(updates))

    @torch.no_grad()
    def update(self, model: nn.Module) -> None:
        self.updates += 1
        warmup_decay = (1.0 + float(self.updates)) / (10.0 + float(self.updates))
        decay = min(self.decay, warmup_decay)
        model_state = model.state_dict()
        for name, ema_value in self.module.state_dict().items():
            source_value = model_state[name].detach()
            if ema_value.is_floating_point():
                ema_value.mul_(decay).add_(source_value, alpha=1.0 - decay)
            else:
                ema_value.copy_(source_value)

    def state_dict(self) -> Dict[str, Tensor]:
        return self.module.state_dict()

    def load_state_dict(self, state_dict: Dict[str, Tensor]) -> None:
        load_model_state(self.module, state_dict, strict=True)


def train_one_epoch(
    model: nn.Module,
    dataloader: DataLoader,
    criterion: nn.Module,
    optimizer: optim.Optimizer,
    scheduler,
    scaler: Optional[GradScaler],
    device: torch.device,
    amp: bool,
    grad_clip_norm: float,
    epoch_index: int,
    metric_learning_criterion: Optional[nn.Module] = None,
    metric_learning_loss_weight: float = 0.0,
    metric_learning_sources: Union[str, Sequence[str]] = "head",
    boundary_contrastive_loss_weight: float = 0.0,
    boundary_contrastive_pairs: str = "0-1,1-2,2-3,4-rest",
    boundary_contrastive_sources: Union[str, Sequence[str]] = "head,patch",
    boundary_contrastive_margin: float = 0.12,
    boundary_contrastive_temperature: float = 0.20,
    boundary_contrastive_max_pairs: int = 128,
    foreground_consistency_loss_weight: float = 0.0,
    foreground_consistency_margin: float = 0.08,
    attention_view_loss_weight: float = 0.0,
    attention_crop_probability: float = 0.50,
    attention_drop_probability: float = 0.25,
    attention_view_start_epoch: int = 2,
    attention_crop_threshold: float = 0.55,
    attention_drop_threshold: float = 0.70,
    attention_crop_padding_ratio: float = 0.08,
    attention_crop_min_area_ratio: float = 0.20,
    attention_view_foreground_weight: float = 0.35,
    attention_view_score_source: str = "learned_attention",
    attention_drop_blur_kernel: int = 15,
    attention_drop_dilation_kernel: int = 5,
    attention_drop_min_area_ratio: float = 0.06,
    attention_drop_max_area_ratio: float = 0.16,
    register_diversity_loss_weight: float = 0.0,
    pairwise_margin_loss_weight: float = 0.0,
    ordinal_maturity_loss_weight: float = 0.0,
    distillation_teacher: Optional[nn.Module] = None,
    distillation_class_indices: Optional[Tensor] = None,
    distillation_loss_weight: float = 0.0,
    distillation_temperature: float = 2.0,
    distillation_focus_class_index: int = 1,
    distillation_focus_class_weight: float = 1.0,
    elr_target_state: Optional[Tensor] = None,
    elr_loss_weight: float = 0.0,
    elr_beta: float = 0.70,
    elr_start_epoch: int = 2,
    use_sam: bool = False,
    grad_accum_steps: int = 1,
    max_nonfinite_grad_steps: int = 8,
    max_batches: int = 0,
    debug_bbox: bool = False,
    model_ema: Optional[ModelEMA] = None,
) -> Tuple[float, Dict[str, float], float]:
    model.train()
    batch_sampler = getattr(dataloader, "batch_sampler", None)
    if batch_sampler is not None and hasattr(batch_sampler, "set_epoch"):
        batch_sampler.set_epoch(epoch_index)
    loss_sum = 0.0
    batch_count = 0
    train_artifact_totals = {
        "patch_norm_mean": 0.0,
        "patch_norm_std": 0.0,
        "patch_norm_max": 0.0,
        "register_norm_mean": 0.0,
        "register_norm_max": 0.0,
        "register_to_patch_ratio": 0.0,
        "high_norm_patch_fraction": 0.0,
    }
    train_loss_components = {
        "cls_loss": 0.0,
        "metric_learning_loss": 0.0,
        "boundary_contrastive_loss": 0.0,
        "boundary_contrastive_terms": 0.0,
        "foreground_consistency_loss": 0.0,
        "attention_view_loss": 0.0,
        "attention_view_fraction": 0.0,
        "attention_crop_fraction": 0.0,
        "attention_drop_fraction": 0.0,
        "attention_drop_area_fraction": 0.0,
        "register_diversity_loss": 0.0,
        "pairwise_margin_loss": 0.0,
        "ordinal_maturity_loss": 0.0,
        "distillation_loss": 0.0,
        "elr_loss": 0.0,
        "sample_weight_mean": 0.0,
        "objectness_loss": 0.0,
        "bbox_l1_loss": 0.0,
        "bbox_giou_loss": 0.0,
        "cardinality_loss": 0.0,
        "count_loss": 0.0,
        "quality_loss": 0.0,
        "count_objectness_consistency_loss": 0.0,
        "auxiliary_loss": 0.0,
    }

    total_batches = len(dataloader)
    batch_iterator = dataloader
    if max_batches:
        total_batches = min(total_batches, max_batches)
        batch_iterator = islice(dataloader, total_batches)

    optimizer.zero_grad(set_to_none=True)
    replay_batches: List[Tuple[object, ...]] = []
    nonfinite_batch_streak = 0
    max_nonfinite_batch_streak = 3
    nonfinite_grad_step_streak = 0
    with tqdm(batch_iterator, desc="Train", leave=False, total=total_batches, dynamic_ncols=True) as pbar:
        for batch_index, batch in enumerate(pbar):
            if len(batch) not in (2, 3):
                raise ValueError("Train dataloader phai tra ve (images, targets) hoac (images, labels, targets).")
            if use_sam:
                # SAM needs a second forward pass; avoid holding cloned CPU batches
                # during regular AdamW training where replay is never consumed.
                replay_batches.append(_clone_batch_for_replay(batch))
            offline_teacher_probabilities = None
            sample_indices = None
            sample_weights = None
            if len(batch) == 2 and _is_detection_targets(batch[1]):
                batch_images, batch_targets = batch
                labels = None
                targets = _move_batch_item_to_device(batch_targets, device)
            elif len(batch) == 3:
                batch_images, batch_labels, batch_targets = batch
                labels = batch_labels.to(device, non_blocking=True) if batch_labels is not None else None
                moved_batch_targets = _move_batch_item_to_device(batch_targets, device)
                offline_teacher_probabilities = None
                if isinstance(moved_batch_targets, dict):
                    sample_value = moved_batch_targets.get("sample_index")
                    if torch.is_tensor(sample_value):
                        sample_indices = sample_value.to(
                            device=device,
                            dtype=torch.long,
                            non_blocking=True,
                        )
                    weight_value = moved_batch_targets.get("sample_weight")
                    if torch.is_tensor(weight_value):
                        sample_weights = weight_value.to(
                            device=device,
                            dtype=torch.float32,
                            non_blocking=True,
                        )
                    teacher_value = moved_batch_targets.get("teacher_probs")
                    if torch.is_tensor(teacher_value):
                        offline_teacher_probabilities = teacher_value.to(
                            device=device,
                            dtype=torch.float32,
                            non_blocking=True,
                        )
                    targets = labels if labels is not None else moved_batch_targets
                else:
                    targets = moved_batch_targets
            else:
                batch_images, batch_targets = batch
                labels = None
                targets = batch_targets.to(device, non_blocking=True)
                offline_teacher_probabilities = None
            images = batch_images.to(device, non_blocking=True)

            loss, features, _, loss_details, _ = _forward_train_loss(
                model=model,
                criterion=criterion,
                metric_learning_criterion=metric_learning_criterion,
                metric_learning_loss_weight=metric_learning_loss_weight,
                metric_learning_sources=metric_learning_sources,
                boundary_contrastive_loss_weight=boundary_contrastive_loss_weight,
                boundary_contrastive_pairs=boundary_contrastive_pairs,
                boundary_contrastive_sources=boundary_contrastive_sources,
                boundary_contrastive_margin=boundary_contrastive_margin,
                boundary_contrastive_temperature=boundary_contrastive_temperature,
                boundary_contrastive_max_pairs=boundary_contrastive_max_pairs,
                foreground_consistency_loss_weight=foreground_consistency_loss_weight,
                foreground_consistency_margin=foreground_consistency_margin,
                attention_view_loss_weight=attention_view_loss_weight,
                attention_crop_probability=attention_crop_probability,
                attention_drop_probability=attention_drop_probability,
                attention_view_start_epoch=attention_view_start_epoch,
                attention_crop_threshold=attention_crop_threshold,
                attention_drop_threshold=attention_drop_threshold,
                attention_crop_padding_ratio=attention_crop_padding_ratio,
                attention_crop_min_area_ratio=attention_crop_min_area_ratio,
                attention_view_foreground_weight=attention_view_foreground_weight,
                attention_view_score_source=attention_view_score_source,
                attention_drop_blur_kernel=attention_drop_blur_kernel,
                attention_drop_dilation_kernel=attention_drop_dilation_kernel,
                attention_drop_min_area_ratio=attention_drop_min_area_ratio,
                attention_drop_max_area_ratio=attention_drop_max_area_ratio,
                epoch_index=epoch_index,
                register_diversity_loss_weight=register_diversity_loss_weight,
                pairwise_margin_loss_weight=pairwise_margin_loss_weight,
                ordinal_maturity_loss_weight=ordinal_maturity_loss_weight,
                distillation_teacher=distillation_teacher,
                distillation_class_indices=distillation_class_indices,
                distillation_loss_weight=distillation_loss_weight,
                distillation_temperature=distillation_temperature,
                distillation_focus_class_index=distillation_focus_class_index,
                distillation_focus_class_weight=distillation_focus_class_weight,
                offline_teacher_probabilities=offline_teacher_probabilities,
                sample_indices=sample_indices,
                sample_weights=sample_weights,
                elr_target_state=elr_target_state,
                elr_loss_weight=elr_loss_weight,
                elr_beta=elr_beta,
                elr_start_epoch=elr_start_epoch,
                elr_update_state=True,
                images=images,
                labels=labels,
                targets=targets,
                device=device,
                amp=amp,
                debug_bbox=debug_bbox and batch_index == 0,
            )
            if not torch.isfinite(loss.detach()):
                nonfinite_batch_streak += 1
                logger.warning(
                    "Skipping non-finite training loss: epoch=%s batch=%s details=%s",
                    epoch_index,
                    batch_index,
                    loss_details,
                )
                optimizer.zero_grad(set_to_none=True)
                replay_batches.clear()
                del loss, features, images, targets, labels
                if nonfinite_batch_streak >= max_nonfinite_batch_streak:
                    raise RuntimeError(
                        "Qua nhieu batch co loss khong huu han lien tiep; "
                        f"epoch={epoch_index} batch={batch_index} streak={nonfinite_batch_streak}. "
                        "Dung best.pt gan nhat va giam LR/AMP neu can."
                    )
                continue
            nonfinite_batch_streak = 0

            accumulation_group_start = (batch_index // grad_accum_steps) * grad_accum_steps
            current_accum_steps = min(grad_accum_steps, total_batches - accumulation_group_start)
            scaled_loss = loss / float(max(1, current_accum_steps))
            if use_sam or scaler is None:
                scaled_loss.backward()
            else:
                scaler.scale(scaled_loss).backward()

            should_step = ((batch_index + 1) % grad_accum_steps == 0) or ((batch_index + 1) == total_batches)
            if should_step:
                if use_sam:
                    gradients_finite, grad_norm = _clip_gradients_and_check(
                        model,
                        max_norm=grad_clip_norm,
                    )
                    if not gradients_finite:
                        nonfinite_grad_step_streak += 1
                        gradient_context = _nonfinite_gradient_context(model)
                        logger.warning(
                            "Skipping optimizer step because gradient norm is non-finite: epoch=%s batch=%s grad_norm=%s streak=%s gradient_context=%s",
                            epoch_index,
                            batch_index,
                            grad_norm,
                            nonfinite_grad_step_streak,
                            gradient_context,
                        )
                        optimizer.zero_grad(set_to_none=True)
                        replay_batches.clear()
                        _raise_if_nonfinite_gradient_streak_exceeded(
                            streak=nonfinite_grad_step_streak,
                            limit=max_nonfinite_grad_steps,
                            epoch_index=epoch_index,
                            batch_index=batch_index,
                            grad_norm=grad_norm,
                            gradient_context=gradient_context,
                        )
                        del loss, scaled_loss, features, images, targets, labels
                        continue
                    optimizer.first_step(zero_grad=True)
                    _raise_if_model_parameters_nonfinite(
                        model,
                        context=f"epoch={epoch_index} batch={batch_index} after_sam_first_step",
                    )

                    for replay_batch in replay_batches:
                        replay_images = replay_batch[0].to(device, non_blocking=True)
                        replay_labels = None
                        replay_offline_teacher_probabilities = None
                        replay_sample_indices = None
                        replay_sample_weights = None
                        if len(replay_batch) == 2 and _is_detection_targets(replay_batch[1]):
                            replay_targets = _move_batch_item_to_device(replay_batch[1], device)
                        elif len(replay_batch) == 3:
                            replay_labels = replay_batch[1].to(device, non_blocking=True)
                            replay_moved_targets = _move_batch_item_to_device(replay_batch[2], device)
                            if isinstance(replay_moved_targets, dict):
                                sample_value = replay_moved_targets.get("sample_index")
                                if torch.is_tensor(sample_value):
                                    replay_sample_indices = sample_value.to(
                                        device=device,
                                        dtype=torch.long,
                                        non_blocking=True,
                                    )
                                weight_value = replay_moved_targets.get("sample_weight")
                                if torch.is_tensor(weight_value):
                                    replay_sample_weights = weight_value.to(
                                        device=device,
                                        dtype=torch.float32,
                                        non_blocking=True,
                                    )
                                teacher_value = replay_moved_targets.get("teacher_probs")
                                if torch.is_tensor(teacher_value):
                                    replay_offline_teacher_probabilities = teacher_value.to(
                                        device=device,
                                        dtype=torch.float32,
                                        non_blocking=True,
                                    )
                                replay_targets = replay_labels if replay_labels is not None else replay_moved_targets
                            else:
                                replay_targets = replay_moved_targets
                        else:
                            replay_targets = replay_batch[1].to(device, non_blocking=True)
                        replay_loss, _, _ = _forward_train_loss(
                            model=model,
                            criterion=criterion,
                            metric_learning_criterion=metric_learning_criterion,
                            metric_learning_loss_weight=metric_learning_loss_weight,
                            metric_learning_sources=metric_learning_sources,
                            boundary_contrastive_loss_weight=boundary_contrastive_loss_weight,
                            boundary_contrastive_pairs=boundary_contrastive_pairs,
                            boundary_contrastive_sources=boundary_contrastive_sources,
                            boundary_contrastive_margin=boundary_contrastive_margin,
                            boundary_contrastive_temperature=boundary_contrastive_temperature,
                            boundary_contrastive_max_pairs=boundary_contrastive_max_pairs,
                            foreground_consistency_loss_weight=foreground_consistency_loss_weight,
                            foreground_consistency_margin=foreground_consistency_margin,
                            attention_view_loss_weight=attention_view_loss_weight,
                            attention_crop_probability=attention_crop_probability,
                            attention_drop_probability=attention_drop_probability,
                            attention_view_start_epoch=attention_view_start_epoch,
                            attention_crop_threshold=attention_crop_threshold,
                            attention_drop_threshold=attention_drop_threshold,
                            attention_crop_padding_ratio=attention_crop_padding_ratio,
                            attention_crop_min_area_ratio=attention_crop_min_area_ratio,
                            attention_view_foreground_weight=attention_view_foreground_weight,
                            attention_view_score_source=attention_view_score_source,
                            attention_drop_blur_kernel=attention_drop_blur_kernel,
                            attention_drop_dilation_kernel=attention_drop_dilation_kernel,
                            attention_drop_min_area_ratio=attention_drop_min_area_ratio,
                            attention_drop_max_area_ratio=attention_drop_max_area_ratio,
                            epoch_index=epoch_index,
                            register_diversity_loss_weight=register_diversity_loss_weight,
                            pairwise_margin_loss_weight=pairwise_margin_loss_weight,
                            ordinal_maturity_loss_weight=ordinal_maturity_loss_weight,
                            distillation_teacher=distillation_teacher,
                            distillation_class_indices=distillation_class_indices,
                            distillation_loss_weight=distillation_loss_weight,
                            distillation_temperature=distillation_temperature,
                            distillation_focus_class_index=distillation_focus_class_index,
                            distillation_focus_class_weight=distillation_focus_class_weight,
                            offline_teacher_probabilities=replay_offline_teacher_probabilities,
                            sample_indices=replay_sample_indices,
                            sample_weights=replay_sample_weights,
                            elr_target_state=elr_target_state,
                            elr_loss_weight=elr_loss_weight,
                            elr_beta=elr_beta,
                            elr_start_epoch=elr_start_epoch,
                            elr_update_state=False,
                            images=replay_images,
                            labels=replay_labels,
                            targets=replay_targets,
                            device=device,
                            amp=amp,
                        )[:3]
                        (replay_loss / float(max(1, current_accum_steps))).backward()

                    gradients_finite, grad_norm = _clip_gradients_and_check(
                        model,
                        max_norm=grad_clip_norm,
                    )
                    if not gradients_finite:
                        nonfinite_grad_step_streak += 1
                        gradient_context = _nonfinite_gradient_context(model)
                        logger.warning(
                            "Skipping SAM second optimizer step because gradient norm is non-finite: epoch=%s batch=%s grad_norm=%s streak=%s gradient_context=%s",
                            epoch_index,
                            batch_index,
                            grad_norm,
                            nonfinite_grad_step_streak,
                            gradient_context,
                        )
                        optimizer.zero_grad(set_to_none=True)
                        replay_batches.clear()
                        _raise_if_nonfinite_gradient_streak_exceeded(
                            streak=nonfinite_grad_step_streak,
                            limit=max_nonfinite_grad_steps,
                            epoch_index=epoch_index,
                            batch_index=batch_index,
                            grad_norm=grad_norm,
                            gradient_context=gradient_context,
                        )
                        del loss, scaled_loss, features, images, targets, labels
                        continue
                    optimizer.second_step(zero_grad=True)
                    nonfinite_grad_step_streak = 0
                    _raise_if_model_parameters_nonfinite(
                        model,
                        context=f"epoch={epoch_index} batch={batch_index} after_sam_second_step",
                    )
                else:
                    if scaler is None:
                        gradients_finite, grad_norm = _clip_gradients_and_check(
                            model,
                            max_norm=grad_clip_norm,
                        )
                        if not gradients_finite:
                            nonfinite_grad_step_streak += 1
                            gradient_context = _nonfinite_gradient_context(model)
                            logger.warning(
                                "Skipping optimizer step because gradient norm is non-finite: epoch=%s batch=%s grad_norm=%s streak=%s gradient_context=%s",
                                epoch_index,
                                batch_index,
                                grad_norm,
                                nonfinite_grad_step_streak,
                                gradient_context,
                            )
                            optimizer.zero_grad(set_to_none=True)
                            replay_batches.clear()
                            _raise_if_nonfinite_gradient_streak_exceeded(
                                streak=nonfinite_grad_step_streak,
                                limit=max_nonfinite_grad_steps,
                                epoch_index=epoch_index,
                                batch_index=batch_index,
                                grad_norm=grad_norm,
                                gradient_context=gradient_context,
                            )
                            del loss, scaled_loss, features, images, targets, labels
                            continue
                        optimizer.step()
                        nonfinite_grad_step_streak = 0
                        optimizer.zero_grad(set_to_none=True)
                        _raise_if_model_parameters_nonfinite(
                            model,
                            context=f"epoch={epoch_index} batch={batch_index} after_optimizer_step",
                        )
                    else:
                        scaler.unscale_(optimizer)
                        gradients_finite, grad_norm = _clip_gradients_and_check(
                            model,
                            max_norm=grad_clip_norm,
                        )
                        if not gradients_finite:
                            nonfinite_grad_step_streak += 1
                            gradient_context = _nonfinite_gradient_context(model)
                            logger.warning(
                                "Skipping AMP optimizer step because gradient norm is non-finite: epoch=%s batch=%s grad_norm=%s streak=%s gradient_context=%s",
                                epoch_index,
                                batch_index,
                                grad_norm,
                                nonfinite_grad_step_streak,
                                gradient_context,
                            )
                            optimizer.zero_grad(set_to_none=True)
                            scaler.update()
                            replay_batches.clear()
                            _raise_if_nonfinite_gradient_streak_exceeded(
                                streak=nonfinite_grad_step_streak,
                                limit=max_nonfinite_grad_steps,
                                epoch_index=epoch_index,
                                batch_index=batch_index,
                                grad_norm=grad_norm,
                                gradient_context=gradient_context,
                            )
                            del loss, scaled_loss, features, images, targets, labels
                            continue
                        scaler.step(optimizer)
                        scaler.update()
                        nonfinite_grad_step_streak = 0
                        optimizer.zero_grad(set_to_none=True)
                        _raise_if_model_parameters_nonfinite(
                            model,
                            context=f"epoch={epoch_index} batch={batch_index} after_amp_optimizer_step",
                        )
                if model_ema is not None:
                    model_ema.update(model)
                replay_batches.clear()
                if scheduler is not None:
                    progress = (epoch_index - 1) + float(batch_index + 1) / float(max(1, total_batches))
                    scheduler.step(progress)

            loss_value = float(loss.detach().cpu().item())
            loss_sum += loss_value
            batch_count += 1
            for key in train_loss_components:
                train_loss_components[key] += float(loss_details.get(key, 0.0))
            current_lr = optimizer.param_groups[0]["lr"] if optimizer.param_groups else 0.0
            pbar.set_postfix(loss=f"{loss_value:.4f}", lr=f"{current_lr:.2e}")

            if features is not None:
                batch_stats = summarize_token_norms(features)
                for key, value in batch_stats.items():
                    train_artifact_totals[key] += float(value)
            del loss, scaled_loss, features, images, targets, labels

    train_artifact_stats = {
        key: value / max(1, batch_count) for key, value in train_artifact_totals.items()
    }
    for key, value in train_loss_components.items():
        train_artifact_stats[key] = value / max(1, batch_count)
    final_lr = optimizer.param_groups[0]["lr"] if optimizer.param_groups else 0.0
    return loss_sum / max(1, batch_count), train_artifact_stats, float(final_lr)


def make_checkpoint_payload(
    model: nn.Module,
    optimizer: optim.Optimizer,
    epoch: int,
    best_macro_f1: float,
    model_config: ModelConfig,
    train_config: TrainConfig,
    augmentation_config: AugmentationConfig,
    data_yaml: Path,
    class_names: Sequence[str],
    metrics: Dict[str, object],
    data_summary: Dict[str, object],
    imbalance_summary: Dict[str, object],
) -> Dict[str, object]:
    calibration = {}
    confidence_curves = metrics.get("confidence_curves", {})
    if confidence_curves:
        calibration["best_macro_f1"] = confidence_curves.get("best_macro_f1")
        calibration["best_macro_f1_confidence"] = confidence_curves.get("best_macro_f1_confidence")
        per_class_thresholds = {}
        for item in confidence_curves.get("per_class", []):
            if not isinstance(item, dict):
                continue
            class_index = item.get("class_index")
            threshold = item.get("best_confidence")
            if class_index is None or threshold is None:
                continue
            per_class_thresholds[str(int(class_index))] = float(threshold)
        if per_class_thresholds:
            calibration["per_class_macro_f1_confidence"] = per_class_thresholds
        calibration["source_split"] = "val"
        calibration["tta_enabled"] = metrics.get("tta_enabled", False)
        calibration["tta_brightness_delta"] = metrics.get("tta_brightness_delta", 0.0)
    detection_confidence_curve = metrics.get("detection_confidence_curve", {})
    if detection_confidence_curve:
        calibration["best_detection_f1"] = detection_confidence_curve.get("best_f1_50")
        calibration["best_detection_f1_confidence"] = detection_confidence_curve.get(
            "best_f1_50_confidence"
        )
        calibration["best_detection_metrics"] = detection_confidence_curve.get("best_metrics", {})
        calibration["source_split"] = "val"
        calibration["tta_enabled"] = metrics.get("tta_enabled", False)
        calibration["tta_brightness_delta"] = metrics.get("tta_brightness_delta", 0.0)

    return {
        "epoch": epoch,
        "best_macro_f1": best_macro_f1,
        "model_state": model.state_dict(),
        "optimizer_state": optimizer.state_dict(),
        "model_config": to_serializable(model_config),
        "train_config": to_serializable(train_config),
        "augmentation_config": to_serializable(augmentation_config),
        "data_yaml": str(data_yaml),
        "class_names": list(class_names),
        "metrics": to_serializable(metrics),
        "calibration": to_serializable(calibration),
        "data_summary": to_serializable(data_summary),
        "imbalance_summary": to_serializable(imbalance_summary),
    }


def _dataset_overview_payload(dataset: MangoYOLOCropDataset) -> Dict[str, object]:
    report = dataset.quality_report()
    return {
        "images": report["image_file_count"],
        "labels": report["label_file_count"],
        "valid_objects": report["valid_object_count"],
        "selected_samples": report.get("selected_sample_count", len(dataset)),
        "single_object_images": report.get("single_object_image_count", 0),
        "multi_object_images": report.get("multi_object_image_count", 0),
        "max_objects_per_image": report.get("max_objects_per_image", 0),
        "ignored_objects": report.get("ignored_object_count", 0),
        "empty_labels": report["empty_label_count"],
        "missing_images": report["missing_image_count"],
        "invalid_lines": report["invalid_line_count"],
        "invalid_bboxes": report["invalid_bbox_count"],
        "invalid_classes": report["invalid_class_count"],
    }


def _load_hard_sample_manifest(path_text: str) -> List[Path]:
    if not str(path_text or "").strip():
        return []
    manifest_path = Path(path_text).expanduser()
    if not manifest_path.exists():
        raise FileNotFoundError(f"Missing hard sample manifest: {manifest_path}")
    paths: List[Path] = []
    with manifest_path.open("r", encoding="utf-8", newline="") as handle:
        first_line = handle.readline()
        handle.seek(0)
        if "," in first_line and ("path" in first_line.lower() or "image" in first_line.lower()):
            reader = csv.DictReader(handle)
            for row in reader:
                value = row.get("image_path") or row.get("path") or row.get("sample_path")
                if value:
                    paths.append(Path(value.strip()))
        else:
            for line in handle:
                value = line.strip()
                if value and not value.startswith("#"):
                    paths.append(Path(value))
    return paths


def _load_sample_weight_manifest(
    path_text: str,
    *,
    default_weight: float,
    max_weight: float,
) -> Tuple[Dict[str, float], Dict[str, object]]:
    summary: Dict[str, object] = {
        "enabled": False,
        "manifest": str(path_text or ""),
    }
    if not str(path_text or "").strip():
        return {}, summary
    manifest_path = Path(path_text).expanduser()
    if not manifest_path.exists():
        raise FileNotFoundError(f"Missing sample weight manifest: {manifest_path}")
    weights_by_path: Dict[str, float] = {}
    by_reason: Counter[str] = Counter()
    by_pair: Counter[str] = Counter()
    duplicate_rows = 0
    invalid_rows = 0
    with manifest_path.open("r", encoding="utf-8-sig", newline="") as handle:
        reader = csv.DictReader(handle)
        if reader.fieldnames is None:
            raise ValueError(f"Sample weight manifest phai la CSV co header: {manifest_path}")
        for row in reader:
            raw_path = (
                str(row.get("image_path", "") or "").strip()
                or str(row.get("path", "") or "").strip()
                or str(row.get("sample_path", "") or "").strip()
            )
            if not raw_path:
                invalid_rows += 1
                continue
            raw_weight = (
                str(row.get("sample_weight", "") or "").strip()
                or str(row.get("weight", "") or "").strip()
            )
            try:
                weight = float(raw_weight) if raw_weight else float(default_weight)
            except ValueError:
                invalid_rows += 1
                continue
            if not math.isfinite(weight) or weight <= 0.0:
                invalid_rows += 1
                continue
            weight = min(float(max_weight), max(1e-6, weight))
            key = str(Path(raw_path).resolve()).lower()
            if key in weights_by_path:
                duplicate_rows += 1
                weight = max(weight, float(weights_by_path[key]))
            weights_by_path[key] = weight
            reason = str(row.get("reason", "") or "").strip()
            if reason:
                by_reason[reason] += 1
            target = str(row.get("target_index", "") or row.get("y_true", "") or "").strip()
            prediction = str(row.get("prediction_index", "") or row.get("y_pred", "") or "").strip()
            if target or prediction:
                by_pair[f"{target}->{prediction}"] += 1
    if not weights_by_path:
        raise ValueError(f"Sample weight manifest khong co dong hop le: {manifest_path}")
    weights = list(weights_by_path.values())
    summary = {
        "enabled": True,
        "manifest": str(manifest_path.resolve()),
        "paths": int(len(weights_by_path)),
        "duplicate_rows": int(duplicate_rows),
        "invalid_rows": int(invalid_rows),
        "default_weight": float(default_weight),
        "max_weight": float(max_weight),
        "mean_manifest_weight": float(sum(weights) / max(1, len(weights))),
        "max_manifest_weight": float(max(weights)),
        "by_reason": dict(by_reason),
        "by_target_prediction_pair": dict(by_pair),
    }
    return weights_by_path, summary


def _build_eval_loader(
    dataset: MangoYOLOCropDataset,
    batch_size: int,
    num_workers: int,
    device: torch.device,
    image_cache_mb: Optional[int] = None,
) -> Tuple[DataLoader, Dict[str, object], Dict[str, object]]:
    dataloader_kwargs, dataloader_summary = build_safe_dataloader_kwargs(
        requested_num_workers=num_workers,
        requested_pin_memory=device.type == "cuda",
        context=f"{getattr(dataset, 'split', 'eval')}_eval",
        prefetch_factor=2,
        persistent_workers=True,
        logger=logger,
    )
    cache_summary = maybe_enable_dataset_image_cache(
        dataset,
        enabled=int(dataloader_summary["effective_num_workers"]) == 0,
        context=f"{getattr(dataset, 'split', 'eval')}_eval",
        max_megabytes=image_cache_mb,
        logger=logger,
    )
    logger.info(
        "Creating eval DataLoader: batch_size=%s requested_workers=%s "
        "effective_workers=%s pin_memory=%s drop_last=%s cache=%s",
        batch_size,
        num_workers,
        dataloader_summary["effective_num_workers"],
        dataloader_summary["effective_pin_memory"],
        False,
        cache_summary.get("enabled", False),
    )
    loader = DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=False,
        collate_fn=build_train_collate_fn(
            num_classes=max(1, int(dataset.num_classes or 1)),
            batch_mix_probability=0.0,
        ),
        **dataloader_kwargs,
    )
    return loader, dataloader_summary, cache_summary


def _resolve_eval_num_workers(train_config: TrainConfig) -> int:
    raw_override = os.getenv("TRKH_EVAL_NUM_WORKERS", "").strip()
    if raw_override:
        try:
            return max(0, int(float(raw_override)))
        except ValueError:
            logger.warning("Ignoring invalid TRKH_EVAL_NUM_WORKERS=%r", raw_override)
    configured = int(getattr(train_config, "eval_num_workers", -1))
    if configured >= 0:
        return configured
    return max(0, int(train_config.num_workers))


def _cleanup_device_memory(device: torch.device, heavy: bool = False) -> None:
    if heavy:
        gc.collect()
    if device.type != "cuda" or not torch.cuda.is_available() or not torch.cuda.is_initialized():
        return
    if not heavy:
        return
    torch.cuda.synchronize(device)
    torch.cuda.empty_cache()
    if hasattr(torch.cuda, "ipc_collect"):
        torch.cuda.ipc_collect()


def _run_architecture_trace_after_training(
    *,
    args: argparse.Namespace,
    data_spec,
    run_dir: Path,
    checkpoints_dir: Path,
    image_size: int,
) -> Dict[str, object]:
    checkpoint_path = checkpoints_dir / "best.pt"
    output_dir = (
        Path(args.trace_architecture_output_dir)
        if args.trace_architecture_output_dir is not None
        else run_dir / "architecture_trace"
    )
    if not checkpoint_path.is_file():
        return {
            "enabled": True,
            "status": "skipped",
            "reason": "missing_best_checkpoint",
            "checkpoint": str(checkpoint_path),
            "output_dir": str(output_dir),
        }

    command = [
        sys.executable,
        "-m",
        "trkh.tools.trace_architecture",
        "--data",
        str(data_spec.data_yaml),
        "--output-dir",
        str(output_dir),
        "--seed",
        str(int(args.trace_architecture_seed)),
        "--device",
        str(args.trace_architecture_device),
        "--image-size",
        str(int(image_size)),
        "--checkpoint",
        str(checkpoint_path),
        "--class-name-mode",
        str(args.class_name_mode or "raw"),
        "--expected-num-classes",
        str(int(args.expected_num_classes or data_spec.num_classes)),
    ]
    start = time.time()
    print({"architecture_trace": "start", "output_dir": str(output_dir)}, flush=True)
    try:
        result = subprocess.run(
            command,
            cwd=str(Path(__file__).resolve().parents[2]),
            check=False,
            capture_output=True,
            text=True,
        )
    except Exception as exc:
        return {
            "enabled": True,
            "status": "failed",
            "error": repr(exc),
            "checkpoint": str(checkpoint_path),
            "output_dir": str(output_dir),
        }
    if result.stdout.strip():
        print(result.stdout.strip(), flush=True)
    if result.stderr.strip():
        print(result.stderr.strip(), flush=True)
    return {
        "enabled": True,
        "status": "completed" if int(result.returncode) == 0 else "failed",
        "returncode": int(result.returncode),
        "seconds": float(time.time() - start),
        "checkpoint": str(checkpoint_path),
        "output_dir": str(output_dir),
    }


def _device_memory_snapshot(device: torch.device) -> Dict[str, float]:
    if device.type != "cuda" or not torch.cuda.is_available() or not torch.cuda.is_initialized():
        return {
            "gpu_memory_allocated_mb": 0.0,
            "gpu_memory_reserved_mb": 0.0,
            "gpu_memory_max_allocated_mb": 0.0,
            "gpu_memory_max_reserved_mb": 0.0,
        }
    return {
        "gpu_memory_allocated_mb": float(torch.cuda.memory_allocated(device) / (1024 * 1024)),
        "gpu_memory_reserved_mb": float(torch.cuda.memory_reserved(device) / (1024 * 1024)),
        "gpu_memory_max_allocated_mb": float(torch.cuda.max_memory_allocated(device) / (1024 * 1024)),
        "gpu_memory_max_reserved_mb": float(torch.cuda.max_memory_reserved(device) / (1024 * 1024)),
    }


def _dataset_cache_snapshot(dataset, prefix: str) -> Dict[str, float]:
    stats_fn = getattr(dataset, "image_cache_stats", None)
    if not callable(stats_fn):
        return {
            f"{prefix}_image_cache_enabled": 0.0,
            f"{prefix}_image_cache_items": 0.0,
            f"{prefix}_image_cache_mb": 0.0,
            f"{prefix}_image_cache_hit_rate": 0.0,
        }
    stats = stats_fn()
    hits = int(stats.get("hits", 0) or 0)
    misses = int(stats.get("misses", 0) or 0)
    requests = max(1, hits + misses)
    return {
        f"{prefix}_image_cache_enabled": 1.0 if stats.get("enabled", False) else 0.0,
        f"{prefix}_image_cache_items": float(stats.get("items", 0) or 0),
        f"{prefix}_image_cache_mb": float(stats.get("bytes", 0) or 0) / (1024 * 1024),
        f"{prefix}_image_cache_hit_rate": float(hits) / float(requests),
    }


class MultiScaleDatasetWrapper:
    def __init__(
        self,
        dataset: MangoYOLOCropDataset,
        scale_controller: MultiScaleTransform,
        transform_factory: Callable[[int], Callable],
    ) -> None:
        self.dataset = dataset
        self.scale_controller = scale_controller
        self.transform_factory = transform_factory
        self.current_scale: Optional[int] = None
        self.current_transform: Optional[Callable] = None
        self.set_epoch(0)

    def __len__(self) -> int:
        return len(self.dataset)

    def __getattr__(self, name: str):
        if name.startswith("__") and name.endswith("__"):
            raise AttributeError(name)
        dataset = self.__dict__.get("dataset")
        if dataset is None:
            raise AttributeError(name)
        return getattr(dataset, name)

    @property
    def num_classes(self) -> Optional[int]:
        return self.dataset.num_classes

    def set_epoch(self, epoch: int) -> None:
        self.scale_controller.set_epoch(epoch)
        next_scale = int(self.scale_controller.get_current_scale())
        if self.current_scale == next_scale and self.current_transform is not None:
            return
        self.current_scale = next_scale
        self.current_transform = self.transform_factory(next_scale)
        self.dataset.transform = self.current_transform

    def get_current_scale(self) -> int:
        return int(self.scale_controller.get_current_scale())

    def __getitem__(self, index: int):
        if self.current_transform is not None and self.dataset.transform is not self.current_transform:
            self.dataset.transform = self.current_transform
        return self.dataset[index]


def make_train_transform(
    image_size: int,
    model_config: ModelConfig,
    augmentation_config: AugmentationConfig,
    random_erasing_probability: float,
    multi_scale_training: bool = False,
    mosaic_probability: float = 0.0,
    mixup_probability: float = 0.0,
    cutmix_probability: float = 0.0,
    copy_paste_probability: float = 0.0,
) -> Callable:
    enabled_augmentations = []
    if multi_scale_training:
        enabled_augmentations.append("multi_scale")
    if mosaic_probability > 0.0:
        enabled_augmentations.append("mosaic")
    if mixup_probability > 0.0:
        enabled_augmentations.append("mixup")
    if cutmix_probability > 0.0:
        enabled_augmentations.append("cutmix")
    if copy_paste_probability > 0.0:
        enabled_augmentations.append("copy_paste")
    if augmentation_config.color_jitter_brightness > 0.0:
        enabled_augmentations.append("brightness")
    if augmentation_config.color_jitter_contrast > 0.0:
        enabled_augmentations.append("contrast")
    if augmentation_config.color_jitter_saturation > 0.0:
        enabled_augmentations.append("saturation")
    if augmentation_config.color_jitter_hue > 0.0:
        enabled_augmentations.append("hue")
    if augmentation_config.random_affine_degrees > 0.0:
        enabled_augmentations.append("random_affine")
    if augmentation_config.horizontal_flip_probability > 0.0:
        enabled_augmentations.append("horizontal_flip")
    if augmentation_config.vertical_flip_probability > 0.0:
        enabled_augmentations.append("vertical_flip")
    if augmentation_config.rotate90_probability > 0.0:
        enabled_augmentations.append("rotate90")
    if augmentation_config.lighting_probability > 0.0:
        enabled_augmentations.append("lighting")
    if augmentation_config.illumination_normalization:
        enabled_augmentations.append("illumination_normalization")
    if augmentation_config.background_suppression_mode not in {"", "none", "off", "false"}:
        enabled_augmentations.append(f"background_{augmentation_config.background_suppression_mode}")
    if augmentation_config.local_exposure_probability > 0.0:
        enabled_augmentations.append("local_exposure")
    if augmentation_config.obstacle_probability > 0.0:
        enabled_augmentations.append("obstacle")
    if augmentation_config.class_aware_augmentation:
        enabled_augmentations.append("class_aware_intensity")
        if not augmentation_config.class_aware_photometric_augmentation:
            enabled_augmentations.append("color_preserving_class_scale")
    if augmentation_config.randaugment_num_ops > 0:
        enabled_augmentations.append("randaugment")
    if random_erasing_probability > 0.0:
        enabled_augmentations.append("random_erasing")

    logger.info(
        "Initializing train transform: image_size=%s enabled_augmentations=%s",
        image_size,
        enabled_augmentations or ["none"],
    )
    logger.debug(
        "Train transform parameters: resize_mode=%s scale_min=%.4f scale_crop_p=%.4f brightness=%.4f "
        "contrast=%.4f saturation=%.4f hue=%.4f random_erasing=%.4f "
        "affine_degrees=%.4f affine_translate=%.4f affine_scale_min=%.4f "
        "hflip=%.4f vflip=%.4f rotate90=%.4f lighting=%.4f "
        "class_aware=%s class_power=%.4f class_max_scale=%.4f "
        "class_photometric=%s "
        "randaugment_ops=%s randaugment_magnitude=%s illumination_norm=%s "
        "background_mode=%s background_prob=%.4f local_exposure=%.4f obstacle=%.4f temporal_frames=%s",
        augmentation_config.resize_mode,
        augmentation_config.random_resized_crop_scale_min,
        augmentation_config.random_resized_crop_probability,
        augmentation_config.color_jitter_brightness,
        augmentation_config.color_jitter_contrast,
        augmentation_config.color_jitter_saturation,
        augmentation_config.color_jitter_hue,
        random_erasing_probability,
        augmentation_config.random_affine_degrees,
        augmentation_config.random_affine_translate,
        augmentation_config.random_affine_scale_min,
        augmentation_config.horizontal_flip_probability,
        augmentation_config.vertical_flip_probability,
        augmentation_config.rotate90_probability,
        augmentation_config.lighting_probability,
        augmentation_config.class_aware_augmentation,
        augmentation_config.class_augmentation_power,
        augmentation_config.class_augmentation_max_scale,
        augmentation_config.class_aware_photometric_augmentation,
        augmentation_config.randaugment_num_ops,
        augmentation_config.randaugment_magnitude,
        augmentation_config.illumination_normalization,
        augmentation_config.background_suppression_mode,
        augmentation_config.background_suppression_probability,
        augmentation_config.local_exposure_probability,
        augmentation_config.obstacle_probability,
        model_config.temporal_frames,
    )

    base_transform = build_train_transform(
        image_size=int(image_size),
        resize_mode=augmentation_config.resize_mode,
        scale_min=augmentation_config.random_resized_crop_scale_min,
        scale_crop_probability=augmentation_config.random_resized_crop_probability,
        brightness=augmentation_config.color_jitter_brightness,
        contrast=augmentation_config.color_jitter_contrast,
        saturation=augmentation_config.color_jitter_saturation,
        hue=augmentation_config.color_jitter_hue,
        random_erasing_probability=random_erasing_probability,
        random_affine_degrees=augmentation_config.random_affine_degrees,
        random_affine_translate=augmentation_config.random_affine_translate,
        random_affine_scale_min=augmentation_config.random_affine_scale_min,
        horizontal_flip_probability=augmentation_config.horizontal_flip_probability,
        vertical_flip_probability=augmentation_config.vertical_flip_probability,
        rotate90_probability=augmentation_config.rotate90_probability,
        lighting_probability=augmentation_config.lighting_probability,
        randaugment_num_ops=augmentation_config.randaugment_num_ops,
        randaugment_magnitude=augmentation_config.randaugment_magnitude,
        illumination_normalization=augmentation_config.illumination_normalization,
        illumination_normalization_strength=augmentation_config.illumination_normalization_strength,
        background_suppression_mode=augmentation_config.background_suppression_mode,
        background_suppression_probability=augmentation_config.background_suppression_probability,
        background_suppression_margin=augmentation_config.background_suppression_margin,
        background_suppression_blur_radius=augmentation_config.background_suppression_blur_radius,
        local_exposure_probability=augmentation_config.local_exposure_probability,
        local_exposure_strength=augmentation_config.local_exposure_strength,
        obstacle_probability=augmentation_config.obstacle_probability,
        obstacle_max_area=augmentation_config.obstacle_max_area,
        scale_photometric_with_augmentation=augmentation_config.class_aware_photometric_augmentation,
    )
    if model_config.temporal_frames <= 1:
        return base_transform
    return PseudoVideoAugmenter(
        frame_transform=base_transform,
        temporal_frames=model_config.temporal_frames,
        deterministic=False,
    )


def main() -> None:
    args = parse_args()
    preloaded_resume_path, preloaded_resume_checkpoint = _preload_resume_checkpoint_for_config(args)
    if preloaded_resume_checkpoint is not None and not bool(args.resume_use_cli_config):
        (
            model_config,
            train_config,
            augmentation_config,
            resume_data_yaml,
        ) = _load_resume_configs_from_checkpoint(preloaded_resume_checkpoint)
        if resume_data_yaml is not None:
            args.data = resume_data_yaml
        print(
            {
                "resume_config": {
                    "source": str(preloaded_resume_path),
                    "data_yaml": str(args.data),
                    "model_config": "loaded_from_checkpoint",
                    "train_config": "loaded_from_checkpoint",
                    "augmentation_config": "loaded_from_checkpoint",
                }
            },
            flush=True,
        )
    else:
        model_config, train_config, augmentation_config = build_configs(args)
        if preloaded_resume_checkpoint is not None:
            print(
                {
                    "resume_config": {
                        "source": str(preloaded_resume_path),
                        "mode": "cli_config_override",
                        "model_config": "loaded_from_cli",
                        "train_config": "loaded_from_cli",
                        "augmentation_config": "loaded_from_cli",
                    }
                },
                flush=True,
            )
    detection_mode = model_config.model_type in DETECTION_MODEL_TYPES
    if detection_mode:
        train_config.balanced_epoch_sampling = False
    classification_object_crops = bool(not detection_mode and not args.disable_classification_object_crops)
    if not detection_mode:
        disabled_batch_composition = {
            "batch_mix_probability": train_config.batch_mix_probability,
            "mosaic_probability": train_config.mosaic_probability,
            "mixup_probability": train_config.mixup_probability,
            "cutmix_probability": train_config.cutmix_probability,
            "copy_paste_probability": train_config.copy_paste_probability,
            "targeted_copy_paste_probability": train_config.targeted_copy_paste_probability,
        }
        train_config.batch_mix_probability = 0.0
        train_config.mosaic_probability = 0.0
        train_config.mixup_probability = 0.0
        train_config.cutmix_probability = 0.0
        train_config.copy_paste_probability = 0.0
        train_config.targeted_copy_paste_probability = 0.0
        print(
            {
                "classification_only": True,
                "object_level_crops": classification_object_crops,
                "disabled_batch_composition": disabled_batch_composition,
                "reason": "classification-only dung crop theo bbox; mosaic/cutmix/copy-paste/mixup bi tat de giu nhan class ro rang",
            },
            flush=True,
        )
    if model_config.model_type in DETECTION_MODEL_TYPES and model_config.temporal_frames > 1:
        raise ValueError(
            "DETR-ViT-Registers hien yeu cau temporal_frames=1. "
            "Hay dung temporal smoothing trong stream_infer.py cho video."
        )

    data_spec = load_data_spec(
        args.data,
        class_name_mode=args.class_name_mode,
        expected_num_classes=args.expected_num_classes or None,
    )
    if detection_mode and data_spec.data_format == "classification_folder":
        raise ValueError(
            "Dataset format=classification_folder chi phu hop classification-only. "
            "Hay dung --model-type vit_registers/resnet50/mobilenet_v3_large/vit_b_16."
        )
    balance_auto_summary = apply_balance_file_auto_adjustment(
        data_spec=data_spec,
        train_config=train_config,
        augmentation_config=augmentation_config,
    )
    if balance_auto_summary.get("enabled"):
        print(
            {
                "dataset_balance_auto_config": balance_auto_summary,
                "reason": (
                    "canbang.yaml chi dung de doi chieu thong ke train; "
                    "balanced sampler moi quyet dinh exposure va khong doc val/test"
                ),
            },
            flush=True,
        )
    class_crop_margin_scales = (
        [float(value) for value in balance_auto_summary.get("auto_repeat_factors", [])]
        if balance_auto_summary.get("enabled")
        else None
    )
    set_seed(train_config.seed, deterministic=train_config.deterministic)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    if device.type == "cuda":
        torch.backends.cuda.matmul.allow_tf32 = True
        torch.backends.cudnn.allow_tf32 = True

    run_name = args.run_name or timestamp_run_name(f"{model_config.model_type}_mango")
    run_dir = ensure_dir(args.output_dir / run_name)
    checkpoints_dir = ensure_dir(run_dir / "checkpoints")

    train_random_erasing_probability = float(augmentation_config.random_erasing_probability)
    if detection_mode and train_random_erasing_probability > 0.0:
        print(
            {
                "detection_train_adjustment": "disable_random_erasing",
                "reason": "giu anh va bbox dong nhat khi train detector tu scratch",
                "requested_random_erasing_probability": train_random_erasing_probability,
                "effective_random_erasing_probability": 0.0,
            },
            flush=True,
        )
        train_random_erasing_probability = 0.0
        augmentation_config.random_erasing_probability = 0.0

    train_transform_factory = partial(
        make_train_transform,
        model_config=model_config,
        augmentation_config=augmentation_config,
        random_erasing_probability=train_random_erasing_probability,
        multi_scale_training=train_config.multi_scale_training,
        mosaic_probability=train_config.mosaic_probability,
        mixup_probability=train_config.mixup_probability,
        cutmix_probability=train_config.cutmix_probability,
        copy_paste_probability=train_config.copy_paste_probability,
    )
    base_train_transform = train_transform_factory(model_config.image_size)
    base_eval_transform = build_eval_transform(
        image_size=model_config.image_size,
        resize_mode=augmentation_config.resize_mode,
        illumination_normalization=augmentation_config.illumination_normalization,
        illumination_normalization_strength=augmentation_config.illumination_normalization_strength,
        background_suppression_mode=augmentation_config.background_suppression_mode,
        background_suppression_margin=augmentation_config.background_suppression_margin,
        background_suppression_blur_radius=augmentation_config.background_suppression_blur_radius,
    )
    if model_config.temporal_frames > 1:
        train_transform = base_train_transform
        eval_transform = PseudoVideoAugmenter(
            frame_transform=base_eval_transform,
            temporal_frames=model_config.temporal_frames,
            deterministic=True,
        )
    else:
        train_transform = base_train_transform
        eval_transform = base_eval_transform

    crop_to_primary_object = not bool(args.full_image_detection)
    if data_spec.data_format == "classification_folder":
        train_dataset = ClassificationFolderDataset.from_data_spec(
            data_spec=data_spec,
            split="train",
            transform=train_transform,
            class_aware_augmentation=augmentation_config.class_aware_augmentation,
            class_augmentation_power=augmentation_config.class_augmentation_power,
            class_augmentation_max_scale=augmentation_config.class_augmentation_max_scale,
            class_crop_margin_scales=class_crop_margin_scales,
        )
        val_dataset = ClassificationFolderDataset.from_data_spec(
            data_spec=data_spec,
            split="val",
            transform=eval_transform,
            class_aware_augmentation=False,
            class_crop_margin_scales=class_crop_margin_scales,
        )
        test_dataset = None
        if data_spec.has_test_split:
            test_dataset = ClassificationFolderDataset.from_data_spec(
                data_spec=data_spec,
                split="test",
                transform=eval_transform,
                class_aware_augmentation=False,
                class_crop_margin_scales=class_crop_margin_scales,
            )
        classification_object_crops = False
        crop_to_primary_object = False
    else:
        train_dataset = MangoYOLOCropDataset.from_data_spec(
            data_spec=data_spec,
            split="train",
            transform=train_transform,
            crop_margin_ratio=augmentation_config.crop_margin_ratio,
            crop_to_primary_object=crop_to_primary_object,
            classification_target=not detection_mode,
            classification_object_crops=classification_object_crops,
            class_aware_augmentation=augmentation_config.class_aware_augmentation,
            class_augmentation_power=augmentation_config.class_augmentation_power,
            class_augmentation_max_scale=augmentation_config.class_augmentation_max_scale,
            class_crop_margin_scale_threshold=augmentation_config.class_crop_margin_scale_threshold,
            class_crop_margin_max_ratio=augmentation_config.class_crop_margin_max_ratio,
            class_crop_margin_scales=class_crop_margin_scales,
        )
        val_dataset = MangoYOLOCropDataset.from_data_spec(
            data_spec=data_spec,
            split="val",
            transform=eval_transform,
            crop_margin_ratio=augmentation_config.crop_margin_ratio,
            crop_to_primary_object=crop_to_primary_object,
            classification_target=not detection_mode,
            classification_object_crops=classification_object_crops,
            class_aware_augmentation=False,
            class_crop_margin_scale_threshold=augmentation_config.class_crop_margin_scale_threshold,
            class_crop_margin_max_ratio=augmentation_config.class_crop_margin_max_ratio,
            class_crop_margin_scales=class_crop_margin_scales,
        )
        test_dataset = None
        if data_spec.has_test_split:
            test_dataset = MangoYOLOCropDataset.from_data_spec(
                data_spec=data_spec,
                split="test",
                transform=eval_transform,
                crop_margin_ratio=augmentation_config.crop_margin_ratio,
                crop_to_primary_object=crop_to_primary_object,
                classification_target=not detection_mode,
                classification_object_crops=classification_object_crops,
                class_aware_augmentation=False,
                class_crop_margin_scale_threshold=augmentation_config.class_crop_margin_scale_threshold,
                class_crop_margin_max_ratio=augmentation_config.class_crop_margin_max_ratio,
                class_crop_margin_scales=class_crop_margin_scales,
            )
    if len(train_dataset) == 0 or len(val_dataset) == 0:
        raise ValueError("Dataset train/val khong co sample hop le de huan luyen.")

    train_dataset_report = train_dataset.quality_report()
    val_dataset_report = val_dataset.quality_report()
    test_dataset_report = test_dataset.quality_report() if test_dataset is not None else None
    train_class_counts = train_dataset.class_counts(data_spec.num_classes)
    val_class_counts = val_dataset.class_counts(data_spec.num_classes)
    test_class_counts = test_dataset.class_counts(data_spec.num_classes) if test_dataset is not None else None
    if (
        balance_auto_summary.get("enabled")
        and balance_auto_summary.get("count_scope") == "train"
        and [int(value) for value in balance_auto_summary.get("class_counts", [])]
        != [int(value) for value in train_class_counts]
    ):
        raise ValueError(
            "Thong ke train trong canbang.yaml khong khop thu muc train. "
            f"canbang={balance_auto_summary.get('class_counts')}, actual={train_class_counts}"
        )
    total_class_counts = [
        int(train_class_counts[index])
        + int(val_class_counts[index])
        + int(test_class_counts[index] if test_class_counts is not None else 0)
        for index in range(data_spec.num_classes)
    ]
    imbalance_summary = resolve_imbalance_strategy(train_config, train_class_counts)

    print("Train audit:", _dataset_overview_payload(train_dataset), flush=True)
    print("Val audit:", _dataset_overview_payload(val_dataset), flush=True)
    if test_dataset is not None:
        print("Test audit:", _dataset_overview_payload(test_dataset), flush=True)
    if detection_mode and not crop_to_primary_object:
        max_labeled_objects = max(
            int(train_dataset_report.get("max_objects_per_image", 0) or 0),
            int(val_dataset_report.get("max_objects_per_image", 0) or 0),
            int((test_dataset_report or {}).get("max_objects_per_image", 0) or 0),
        )
        if max_labeled_objects > int(model_config.num_queries):
            print(
                "Warning: num_queries nho hon so object toi da trong label; "
                f"num_queries={model_config.num_queries}, max_objects_per_image={max_labeled_objects}.",
                flush=True,
            )
        if 0 < int(train_config.eval_max_detections_per_image) < max_labeled_objects:
            print(
                "Warning: eval_max_detections_per_image dang cat bot object tren anh nhieu trai; "
                f"eval_max_detections_per_image={train_config.eval_max_detections_per_image}, "
                f"max_objects_per_image={max_labeled_objects}.",
                flush=True,
            )
    print("Imbalance strategy:", imbalance_summary, flush=True)

    plot_dataset_overview(
        class_names=data_spec.class_names,
        train_class_counts=train_class_counts,
        val_class_counts=val_class_counts,
        train_bboxes=train_dataset.bboxes(),
        val_bboxes=val_dataset.bboxes(),
        train_audit=train_dataset_report,
        val_audit=val_dataset_report,
        output_path=run_dir / "labels.png",
    )
    color_audit_summary = plot_dataset_color_audit(
        class_names=data_spec.class_names,
        train_paths=train_dataset.sample_paths(),
        train_labels=train_dataset.labels(),
        val_paths=val_dataset.sample_paths(),
        val_labels=val_dataset.labels(),
        output_path=run_dir / "color_audit.png",
        summary_path=run_dir / "color_audit.json",
        seed=train_config.seed,
    )
    print(
        {
            "color_audit": {
                "plot": str(run_dir / "color_audit.png"),
                "summary": str(run_dir / "color_audit.json"),
                "max_train_val_lab_delta": max(
                    (
                        float(item.get("train_val_lab_delta", 0.0) or 0.0)
                        for item in color_audit_summary.get("classes", {}).values()
                        if isinstance(item, dict)
                    ),
                    default=0.0,
                ),
            }
        },
        flush=True,
    )

    multi_scale_controller = None
    if train_config.multi_scale_training:
        multi_scale_controller = MultiScaleTransform(
            scales=(224, 320, 416, 512, 640),
            update_interval=train_config.multi_scale_epochs,
            base_transform=None,
        )
        train_dataset = MultiScaleDatasetWrapper(
            dataset=train_dataset,
            scale_controller=multi_scale_controller,
            transform_factory=train_transform_factory,
        )
        print(
            {
                "multi_scale_training": True,
                "scales": list(multi_scale_controller.scales),
                "update_interval_epochs": train_config.multi_scale_epochs,
                "initial_scale": train_dataset.get_current_scale(),
            },
            flush=True,
        )

    offline_distillation_summary: Dict[str, object] = {"enabled": False}
    if str(train_config.distillation_teacher_csv or "").strip():
        offline_probabilities, offline_distillation_summary = _load_offline_distillation_probabilities(
            csv_path=Path(train_config.distillation_teacher_csv),
            num_classes=data_spec.num_classes,
            class_names=data_spec.class_names,
        )
        train_dataset = TeacherProbabilityDataset(
            train_dataset,
            offline_probabilities,
            num_classes=data_spec.num_classes,
        )
        offline_distillation_summary.update(train_dataset.teacher_probability_summary())
        offline_distillation_summary.update(
            {
                "loss_weight": float(train_config.distillation_weight),
                "temperature": float(train_config.distillation_temperature),
                "focus_class_index": int(train_config.distillation_focus_class_index),
                "focus_class_name": data_spec.class_names[
                    int(train_config.distillation_focus_class_index)
                ]
                if 0 <= int(train_config.distillation_focus_class_index) < data_spec.num_classes
                else "",
                "focus_class_weight": float(train_config.distillation_focus_class_weight),
            }
        )
        print({"offline_distillation": offline_distillation_summary}, flush=True)

    sample_weight_summary: Dict[str, object] = {"enabled": False}
    if str(train_config.sample_weight_manifest or "").strip():
        if detection_mode:
            raise ValueError("Sample-weight manifest hien chi ho tro classification-only train.")
        sample_weights, sample_weight_summary = _load_sample_weight_manifest(
            str(train_config.sample_weight_manifest),
            default_weight=float(train_config.sample_weight_factor),
            max_weight=float(train_config.sample_weight_max),
        )
        train_dataset = SampleWeightDataset(
            train_dataset,
            sample_weights,
            default_weight=1.0,
            max_weight=float(train_config.sample_weight_max),
        )
        sample_weight_summary.update(train_dataset.sample_weight_summary())
        print({"sample_weight_manifest": sample_weight_summary}, flush=True)

    if balance_auto_summary.get("enabled"):
        rare_class_repeat_factors = [
            float(value)
            for value in balance_auto_summary.get("auto_repeat_factors", [])
        ]
    else:
        rare_class_repeat_factors = build_rare_class_repeat_factors(
            train_class_counts,
            repeat_power=augmentation_config.rare_class_repeat_power,
            max_factor=augmentation_config.rare_class_repeat_max_factor,
            min_ratio=augmentation_config.rare_class_repeat_min_ratio,
        )
    rare_class_repeat_summary: Dict[str, object] = {
        "enabled": False,
        "class_repeat_factors": rare_class_repeat_factors,
        "source": "canbang.yaml" if balance_auto_summary.get("enabled") else "train_class_counts",
    }
    if (
        augmentation_config.class_aware_augmentation
        and augmentation_config.rare_class_repeat
        and not imbalance_summary["use_weighted_sampler"]
        and max(rare_class_repeat_factors, default=1.0) > 1.0
    ):
        train_dataset = RareClassRepeatDataset(
            train_dataset,
            class_repeat_factors=rare_class_repeat_factors,
            seed=train_config.seed,
        )
        rare_class_repeat_summary = {
            "enabled": True,
            "source": "canbang.yaml" if balance_auto_summary.get("enabled") else "train_class_counts",
            **train_dataset.repeat_summary(),
        }
        print(
            {
                "rare_class_repeat": rare_class_repeat_summary,
                "reason": "tang exposure cho lop it mau bang stochastic geometric augmentation, khong tang photometric neu khong bat co rieng",
                "class_aware_photometric_augmentation": bool(
                    augmentation_config.class_aware_photometric_augmentation
                ),
            },
            flush=True,
        )
    elif augmentation_config.rare_class_repeat and imbalance_summary["use_weighted_sampler"]:
        print(
            {
                "rare_class_repeat": "skipped",
                "reason": "strict balanced sampler da duoc bat, tranh oversample hai lan",
                "class_repeat_factors": rare_class_repeat_factors,
            },
            flush=True,
        )
        rare_class_repeat_summary = {
            "enabled": False,
            "skipped_reason": "strict_balanced_sampler_enabled",
            "class_repeat_factors": rare_class_repeat_factors,
            "source": "canbang.yaml" if balance_auto_summary.get("enabled") else "train_class_counts",
        }
    hard_sample_repeat_summary: Dict[str, object] = {
        "enabled": False,
        "manifest": str(train_config.hard_sample_manifest or ""),
        "repeat_factor": float(train_config.hard_sample_repeat_factor),
    }
    hard_sample_paths = _load_hard_sample_manifest(str(train_config.hard_sample_manifest or ""))
    if hard_sample_paths and float(train_config.hard_sample_repeat_factor) > 1.0:
        train_dataset = HardSampleRepeatDataset(
            train_dataset,
            hard_sample_paths=hard_sample_paths,
            repeat_factor=float(train_config.hard_sample_repeat_factor),
            seed=train_config.seed,
        )
        hard_sample_repeat_summary = {
            "enabled": True,
            "manifest": str(train_config.hard_sample_manifest),
            **train_dataset.repeat_summary(),
        }
        print(
            {
                "hard_sample_repeat": hard_sample_repeat_summary,
                "reason": (
                    "lap lai hard train samples da mine tu train split; "
                    "neu strict balanced sampler bat thi sampler se can bang lai exposure theo class"
                ),
            },
            flush=True,
        )
    elr_summary: Dict[str, object] = {"enabled": False}
    if float(train_config.elr_loss_weight) > 0.0:
        if detection_mode:
            raise ValueError("ELR hien chi ho tro classification-only train.")
        train_dataset = IndexedSampleDataset(train_dataset)
        elr_summary = {
            "enabled": True,
            "samples": int(len(train_dataset)),
            "num_classes": int(data_spec.num_classes),
            "loss_weight": float(train_config.elr_loss_weight),
            "beta": float(train_config.elr_beta),
            "start_epoch": int(train_config.elr_start_epoch),
            "source_split": "train_only_after_repeat_wrappers",
            "note": (
                "sample_index duoc tao sau rare/hard repeat va truoc sampler; "
                "khong dung val/test de cap nhat target history"
            ),
        }
        if float(train_config.batch_mix_probability) > 0.0:
            elr_summary["batch_mix_note"] = (
                "ELR metadata lam collator tra ve hard-label batch; "
                "nen dat --batch-mix-probability 0 cho run ELR de hanh vi ro rang"
            )
        print({"early_learning_regularization": elr_summary}, flush=True)
    class_target_scales = _combine_class_target_scales(
        data_spec.num_classes,
        getattr(train_dataset, "class_augmentation_scales", []),
        rare_class_repeat_factors,
        [
            float(value)
            for value in balance_auto_summary.get("raw_auto_repeat_factors", [])
        ],
    )
    class_crop_margin_summary = {
        "enabled": False,
        "base_ratio": float(augmentation_config.crop_margin_ratio),
        "max_ratio": float(augmentation_config.class_crop_margin_max_ratio),
        "scale_threshold": float(augmentation_config.class_crop_margin_scale_threshold),
        "class_target_scales": class_target_scales,
        "targeted_classes": [
            int(index)
            for index, value in enumerate(class_target_scales)
            if float(value) >= float(augmentation_config.class_crop_margin_scale_threshold)
            and float(augmentation_config.class_crop_margin_max_ratio)
            > float(augmentation_config.crop_margin_ratio)
        ],
        "effective_margin_ratios": [
            float(
                min(
                    float(augmentation_config.class_crop_margin_max_ratio),
                    float(augmentation_config.crop_margin_ratio) * float(value),
                )
                if float(value) >= float(augmentation_config.class_crop_margin_scale_threshold)
                and float(augmentation_config.class_crop_margin_max_ratio)
                > float(augmentation_config.crop_margin_ratio)
                else float(augmentation_config.crop_margin_ratio)
            )
            for value in class_target_scales
        ],
        "reason": "classification crop margin tu dong nham class co scale >= threshold",
    }
    class_crop_margin_summary["enabled"] = bool(class_crop_margin_summary["targeted_classes"])
    print({"class_crop_margin": class_crop_margin_summary}, flush=True)
    if detection_mode:
        targeted_copy_paste_summary = {
            "enabled": bool(train_config.copy_paste_probability > 0.0),
            "scale_threshold": float(train_config.targeted_copy_paste_scale_threshold),
            "target_probability": float(train_config.targeted_copy_paste_probability),
            "class_target_scales": class_target_scales,
            "targeted_classes": [
                int(index)
                for index, value in enumerate(class_target_scales)
                if float(value) >= float(train_config.targeted_copy_paste_scale_threshold)
            ],
            "photometric_scaled_for_target_classes": bool(
                augmentation_config.class_aware_photometric_augmentation
            ),
        }
    else:
        targeted_copy_paste_summary = {
            "enabled": False,
            "reason": "classification_only_uses_object_level_crops_instead",
            "class_target_scales": class_target_scales,
            "photometric_scaled_for_target_classes": False,
        }
    print({"targeted_copy_paste": targeted_copy_paste_summary}, flush=True)

    train_sampler = (
        build_weighted_sampler(
            train_dataset,
            batch_size=train_config.batch_size,
            num_classes=data_spec.num_classes,
            epoch_multiplier=float(imbalance_summary["weighted_sampler_epoch_multiplier"]),
        )
        if imbalance_summary["use_weighted_sampler"]
        else None
    )
    balanced_exposure_summary: Dict[str, object] = {"enabled": False}
    if isinstance(train_sampler, StrictBalancedBatchSampler):
        balanced_exposure_summary = {
            "enabled": True,
            "source": "train_split_labels_only",
            **train_sampler.exposure_summary(),
        }
        if (
            float(balanced_exposure_summary["relative_gap"])
            > float(train_config.balanced_epoch_tolerance) + 1e-12
        ):
            raise ValueError(
                "Balanced epoch sampler khong dat tolerance exposure: "
                f"gap={balanced_exposure_summary['relative_gap']:.6f}, "
                f"tolerance={train_config.balanced_epoch_tolerance:.6f}"
            )
        print({"balanced_epoch_exposure": balanced_exposure_summary}, flush=True)
    dataloader_kwargs, train_dataloader_summary = build_safe_dataloader_kwargs(
        requested_num_workers=train_config.num_workers,
        requested_pin_memory=device.type == "cuda",
        context="train",
        prefetch_factor=2,
        persistent_workers=not train_config.multi_scale_training,
        logger=logger,
    )
    train_cache_summary = maybe_enable_dataset_image_cache(
        train_dataset,
        enabled=int(train_dataloader_summary["effective_num_workers"]) == 0,
        context="train",
        max_megabytes=(
            int(train_config.train_image_cache_mb)
            if int(train_config.train_image_cache_mb) >= 0
            else None
        ),
        logger=logger,
    )

    train_collate_fn = build_train_collate_fn(
        num_classes=data_spec.num_classes,
        batch_mix_probability=train_config.batch_mix_probability,
        mosaic_probability=train_config.mosaic_probability,
        mosaic_min_split=train_config.mosaic_min_split,
        mosaic_max_split=train_config.mosaic_max_split,
        mixup_probability=train_config.mixup_probability,
        mixup_alpha=train_config.mixup_alpha,
        cutmix_probability=train_config.cutmix_probability,
        cutmix_alpha=train_config.cutmix_alpha,
        copy_paste_probability=train_config.copy_paste_probability,
        copy_paste_max_objects=train_config.copy_paste_max_objects,
        max_detection_objects=model_config.num_queries,
        class_aware_mix_probability_boost=(
            augmentation_config.class_aware_mix_probability_boost
            if augmentation_config.class_aware_augmentation
            else 0.0
        ),
        class_aware_mix_source_power=(
            augmentation_config.class_aware_mix_source_power
            if augmentation_config.class_aware_augmentation
            else 0.0
        ),
        targeted_copy_paste_class_scales=class_target_scales,
        targeted_copy_paste_scale_threshold=train_config.targeted_copy_paste_scale_threshold,
        targeted_copy_paste_probability=train_config.targeted_copy_paste_probability,
    )
    logger.info(
        "Creating train DataLoader: batch_size=%s requested_workers=%s "
        "effective_workers=%s pin_memory=%s drop_last=%s sampler=%s cache=%s",
        train_config.batch_size,
        train_config.num_workers,
        train_dataloader_summary["effective_num_workers"],
        train_dataloader_summary["effective_pin_memory"],
        getattr(train_sampler, "drop_last", False),
        type(train_sampler).__name__ if train_sampler is not None else None,
        train_cache_summary.get("enabled", False),
    )
    if train_sampler is not None:
        train_sampler.seed = int(train_config.seed)
        train_loader = DataLoader(
            train_dataset,
            batch_sampler=train_sampler,
            collate_fn=train_collate_fn,
            **dataloader_kwargs,
        )
    else:
        train_loader = DataLoader(
            train_dataset,
            batch_size=train_config.batch_size,
            shuffle=True,
            collate_fn=train_collate_fn,
            **dataloader_kwargs,
        )
    val_loader, val_dataloader_summary, val_cache_summary = _build_eval_loader(
        dataset=val_dataset,
        batch_size=train_config.batch_size,
        num_workers=_resolve_eval_num_workers(train_config),
        device=device,
        image_cache_mb=(
            int(train_config.eval_image_cache_mb)
            if int(train_config.eval_image_cache_mb) >= 0
            else None
        ),
    )
    print(
        "DataLoader setup:",
        {
            "train": train_dataloader_summary,
            "val": val_dataloader_summary,
            "image_cache": {
                "train": train_cache_summary,
                "val": val_cache_summary,
            },
        },
        flush=True,
    )
    effective_batch_size = train_config.batch_size * train_config.grad_accum_steps

    scheduler_total_epochs = _resolve_scheduler_total_epochs(train_config)
    model = create_model(
        num_classes=data_spec.num_classes,
        model_config=to_serializable(model_config),
    ).to(device)
    distillation_teacher: Optional[nn.Module] = None
    distillation_class_indices: Optional[Tensor] = None
    distillation_summary: Dict[str, object] = {"enabled": False}
    if train_config.pretrained_distillation:
        if detection_mode:
            raise ValueError(
                "Pretrained distillation hien chi ho tro classification-only TRKH."
            )
        if not 0 <= int(train_config.distillation_focus_class_index) < data_spec.num_classes:
            raise ValueError(
                "distillation_focus_class_index nam ngoai so class cua dataset."
            )
        if str(train_config.distillation_teacher_checkpoint or "").strip():
            distillation_teacher, distillation_class_indices, distillation_summary = (
                _build_pretrained_distillation_teacher(
                    checkpoint_path=Path(train_config.distillation_teacher_checkpoint),
                    target_class_names=data_spec.class_names,
                    device=device,
                )
            )
            distillation_summary.update(
                {
                    "loss_weight": float(train_config.distillation_weight),
                    "temperature": float(train_config.distillation_temperature),
                    "focus_class_index": int(
                        train_config.distillation_focus_class_index
                    ),
                    "focus_class_name": data_spec.class_names[
                        int(train_config.distillation_focus_class_index)
                    ],
                    "focus_class_weight": float(
                        train_config.distillation_focus_class_weight
                    ),
                }
            )
            print({"pretrained_distillation": distillation_summary}, flush=True)
        elif offline_distillation_summary.get("enabled"):
            distillation_summary = {
                "enabled": False,
                "reason": "offline_teacher_probability_csv_only",
            }
        else:
            raise ValueError(
                "Distillation da bat nhung khong co checkpoint hoac offline CSV."
            )
    elr_target_state: Optional[Tensor] = None
    if bool(elr_summary.get("enabled")):
        elr_target_state = torch.zeros(
            (int(elr_summary["samples"]), int(data_spec.num_classes)),
            device=device,
            dtype=torch.float32,
        )
        print(
            {
                "early_learning_regularization_state": {
                    "shape": list(elr_target_state.shape),
                    "device": str(elr_target_state.device),
                    "dtype": str(elr_target_state.dtype).replace("torch.", ""),
                }
            },
            flush=True,
        )
    if model_config.gradient_checkpointing:
        GradientCheckpointingEnabler.enable_gradient_checkpointing(model)
        print(
            {
                "gradient_checkpointing": True,
                "status": "enabled",
            },
            flush=True,
        )
    optimizer_param_groups = build_optimizer_param_groups(
        model,
        train_config.weight_decay,
        learning_rate=train_config.learning_rate,
        backbone_lr_scale=train_config.backbone_lr_scale,
    )
    if train_config.use_sam:
        optimizer = SAM(
            optimizer_param_groups,
            base_optimizer=optim.AdamW,
            lr=train_config.learning_rate,
            betas=(0.9, 0.999),
            rho=train_config.sam_rho,
            adaptive=train_config.sam_adaptive,
        )
    else:
        optimizer = optim.AdamW(
            optimizer_param_groups,
            lr=train_config.learning_rate,
            betas=(0.9, 0.999),
        )
    scheduler = build_warmup_decay_scheduler(
        optimizer=optimizer,
        warmup_epochs=train_config.warmup_epochs,
        warmup_start_factor=train_config.warmup_start_factor,
        total_epochs=scheduler_total_epochs,
        min_learning_rate=train_config.min_learning_rate,
        decay_style=train_config.lr_scheduler,
    )
    optimized_training_scheduler = None
    if train_config.optimized_scheduler:
        optimized_training_scheduler = TrainingScheduler(
            model=model,
            stage1_epochs=5,
            stage2_epochs=max(0, int(train_config.epochs) - 5),
            total_epochs=int(train_config.epochs),
        )
        print(
            {
                "optimized_scheduler": True,
                "stage1_epochs": 5,
                "stage2_progressive_unfreezing": "0-20 frozen, 20-40 last 2 layers, 40+ all",
            },
            flush=True,
        )

    class_weights = None
    if imbalance_summary["use_class_weights"]:
        class_weights = torch.tensor(
            imbalance_summary["class_weights"],
            dtype=torch.float32,
            device=device,
        )
    criterion_class_weights = class_weights.detach().cpu() if class_weights is not None else None

    if detection_mode:
        criterion = HybridDetectionClassificationLoss(
            num_classes=data_spec.num_classes,
            class_weights=criterion_class_weights,
            label_smoothing=train_config.label_smoothing,
            cls_weight=train_config.cls_loss_weight,
            bbox_l1_weight=train_config.bbox_l1_loss_weight,
            bbox_giou_weight=train_config.bbox_giou_loss_weight,
            background_weight=train_config.background_loss_weight,
            objectness_weight=train_config.objectness_loss_weight,
            objectness_focal_alpha=train_config.objectness_focal_alpha,
            objectness_focal_gamma=train_config.objectness_focal_gamma,
            matcher_class_cost=train_config.matcher_class_cost,
            matcher_objectness_cost=train_config.matcher_objectness_cost,
            cardinality_weight=train_config.cardinality_loss_weight,
            count_weight=train_config.count_loss_weight,
            quality_weight=train_config.quality_loss_weight,
            auxiliary_weight=train_config.auxiliary_loss_weight,
            count_objectness_consistency_weight=train_config.count_objectness_consistency_weight,
        )
        eval_criterion = HybridDetectionClassificationLoss(
            num_classes=data_spec.num_classes,
            class_weights=criterion_class_weights,
            label_smoothing=train_config.label_smoothing,
            cls_weight=train_config.cls_loss_weight,
            bbox_l1_weight=train_config.bbox_l1_loss_weight,
            bbox_giou_weight=train_config.bbox_giou_loss_weight,
            background_weight=train_config.background_loss_weight,
            objectness_weight=train_config.objectness_loss_weight,
            objectness_focal_alpha=train_config.objectness_focal_alpha,
            objectness_focal_gamma=train_config.objectness_focal_gamma,
            matcher_class_cost=train_config.matcher_class_cost,
            matcher_objectness_cost=train_config.matcher_objectness_cost,
            cardinality_weight=train_config.cardinality_loss_weight,
            count_weight=train_config.count_loss_weight,
            quality_weight=train_config.quality_loss_weight,
            auxiliary_weight=0.0,
            count_objectness_consistency_weight=train_config.count_objectness_consistency_weight,
        )
    else:
        classification_loss_name = str(train_config.classification_loss).strip().lower()
        if classification_loss_name == "balanced_softmax":
            criterion = BalancedSoftmaxFocalLoss(
                class_counts=train_class_counts,
                weight=class_weights,
                gamma=float(imbalance_summary["focal_loss_gamma"]),
                focal_mix=float(imbalance_summary["focal_loss_mix"]),
                label_smoothing=train_config.label_smoothing,
                prior_tau=train_config.balanced_softmax_tau,
            )
            eval_criterion = BalancedSoftmaxFocalLoss(
                class_counts=train_class_counts,
                weight=class_weights,
                gamma=float(imbalance_summary["focal_loss_gamma"]),
                focal_mix=float(imbalance_summary["focal_loss_mix"]),
                label_smoothing=train_config.label_smoothing,
                prior_tau=train_config.balanced_softmax_tau,
            )
        else:
            criterion = LDAMFocalLoss(
                class_counts=train_class_counts,
                weight=class_weights,
                gamma=float(imbalance_summary["focal_loss_gamma"]),
                focal_mix=float(imbalance_summary["focal_loss_mix"]),
                label_smoothing=train_config.label_smoothing,
                max_margin=float(imbalance_summary["ldam_max_margin"]),
                scale=train_config.ldam_scale if train_config.use_ldam else 1.0,
            )
            eval_criterion = LDAMFocalLoss(
                class_counts=train_class_counts,
                weight=class_weights,
                gamma=float(imbalance_summary["focal_loss_gamma"]),
                focal_mix=float(imbalance_summary["focal_loss_mix"]),
                label_smoothing=train_config.label_smoothing,
                max_margin=float(imbalance_summary["ldam_max_margin"]),
                scale=train_config.ldam_scale if train_config.use_ldam else 1.0,
            )
        print(
            {
                "classification_loss": {
                    "type": classification_loss_name,
                    "balanced_softmax_tau": (
                        float(train_config.balanced_softmax_tau)
                        if classification_loss_name == "balanced_softmax"
                        else None
                    ),
                    "class_weights": bool(class_weights is not None),
                    "focal_loss_gamma": float(imbalance_summary["focal_loss_gamma"]),
                    "focal_loss_mix": float(imbalance_summary["focal_loss_mix"]),
                }
            },
            flush=True,
        )
    metric_learning_criterion: Optional[nn.Module] = None
    if (not detection_mode) and float(train_config.metric_learning_loss_weight) > 0.0:
        metric_learning_criterion = SupervisedContrastiveLoss(
            temperature=train_config.metric_learning_temperature,
            class_balanced=train_config.metric_learning_class_balanced,
        ).to(device)
        print(
            {
                "metric_learning": {
                    "type": "supervised_contrastive",
                    "loss_weight": float(train_config.metric_learning_loss_weight),
                    "temperature": float(train_config.metric_learning_temperature),
                    "class_balanced": bool(train_config.metric_learning_class_balanced),
                    "sources": _parse_metric_learning_sources(train_config.metric_learning_sources),
                },
                "reason": "VFF-like cosine CSCL tren nhieu muc feature, khong dung pretrain",
            },
            flush=True,
        )
    if (not detection_mode) and float(train_config.boundary_contrastive_loss_weight) > 0.0:
        print(
            {
                "boundary_contrastive": {
                    "type": "in_batch_hard_boundary_contrastive",
                    "loss_weight": float(train_config.boundary_contrastive_loss_weight),
                    "pairs": _parse_boundary_contrastive_pairs(
                        train_config.boundary_contrastive_pairs,
                        data_spec.num_classes,
                    ),
                    "sources": _parse_metric_learning_sources(train_config.boundary_contrastive_sources),
                    "margin": float(train_config.boundary_contrastive_margin),
                    "temperature": float(train_config.boundary_contrastive_temperature),
                    "max_pairs": int(train_config.boundary_contrastive_max_pairs),
                },
                "reason": "Tap trung so sanh class de nham trong batch, khong tao pair dataset.",
            },
            flush=True,
        )
    train_amp = bool(train_config.amp)
    if train_config.use_sam and train_amp:
        train_amp = False
        print(
            "Warning: tat AMP trong train loop khi bat SAM de dam bao replay 2-pass va gradient accumulation on dinh.",
            flush=True,
        )
    effective_amp_dtype = resolve_amp_dtype(device) if train_amp and device.type == "cuda" else None
    scaler = (
        GradScaler(
            "cuda",
            enabled=device.type == "cuda" and train_amp and effective_amp_dtype != torch.bfloat16,
        )
        if not train_config.use_sam
        else None
    )
    resume_checkpoint_path = _resolve_resume_checkpoint_path(
        resume_path=None if args.disable_resume else (preloaded_resume_path or args.resume),
        auto_resume=bool(args.auto_resume and not args.disable_resume),
        run_dir=run_dir,
        disable_resume=bool(args.disable_resume),
    )
    resume_summary: Optional[Dict[str, object]] = None
    resume_checkpoint: Optional[Dict[str, object]] = None
    if resume_checkpoint_path is not None:
        resume_summary, resume_checkpoint = _load_training_checkpoint(
            resume_path=resume_checkpoint_path,
            model=model,
            optimizer=optimizer,
            scheduler=scheduler,
            scaler=scaler,
            device=device,
            class_names=data_spec.class_names,
            restore_optimizer=not bool(args.resume_reset_optimizer),
            restore_scheduler=not bool(args.resume_reset_scheduler),
            restore_scaler=not bool(args.resume_reset_scaler),
            allow_added_detection_heads=bool(args.resume_use_cli_config),
        )
        print({"resume": resume_summary}, flush=True)
    model_ema: Optional[ModelEMA] = None
    if train_config.model_ema:
        ema_updates = 0
        model_ema = ModelEMA(
            model,
            decay=train_config.model_ema_decay,
        )
        if resume_checkpoint is not None and isinstance(resume_checkpoint.get("ema_model_state"), dict):
            ema_partial_load_summary = _load_model_state_allowing_extensions(
                model_ema.module,
                resume_checkpoint["ema_model_state"],
                allow_extensions=bool(args.resume_use_cli_config),
            )
            if ema_partial_load_summary is not None:
                print({"model_ema_partial_load": ema_partial_load_summary}, flush=True)
            ema_updates = int(resume_checkpoint.get("ema_updates", 0) or 0)
            model_ema.updates = max(0, ema_updates)
        print(
            {
                "model_ema": {
                    "enabled": True,
                    "decay": float(train_config.model_ema_decay),
                    "updates": int(model_ema.updates),
                    "validation_weights": "ema",
                    "best_checkpoint_weights": "ema",
                }
            },
            flush=True,
        )
    print(
        "Optimizer setup:",
        {
            "name": "SAM(AdamW)" if train_config.use_sam else "AdamW",
            "sam_rho": train_config.sam_rho if train_config.use_sam else None,
            "sam_adaptive": train_config.sam_adaptive if train_config.use_sam else None,
            "effective_train_amp": train_amp,
            "effective_amp_dtype": str(effective_amp_dtype).replace("torch.", "") if effective_amp_dtype is not None else None,
            "grad_scaler_enabled": bool(scaler is not None and scaler.is_enabled()),
            "model_ema": bool(model_ema is not None),
            "model_ema_decay": (
                float(train_config.model_ema_decay) if model_ema is not None else None
            ),
        },
        flush=True,
    )
    print(
        "Scheduler setup:",
        {
            "type": train_config.lr_scheduler,
            "warmup_epochs": train_config.warmup_epochs,
            "total_epochs_for_decay": scheduler_total_epochs,
            "base_lr": train_config.learning_rate,
            "backbone_lr_scale": train_config.backbone_lr_scale,
            "min_lr": train_config.min_learning_rate,
        },
        flush=True,
    )

    data_summary = {
        "data_yaml": str(data_spec.data_yaml),
        "data_format": str(data_spec.data_format),
        "train_images": str(data_spec.train_images),
        "train_labels": str(data_spec.train_labels),
        "val_images": str(data_spec.val_images),
        "val_labels": str(data_spec.val_labels),
        "test_images": str(data_spec.test_images) if data_spec.test_images is not None else None,
        "test_labels": str(data_spec.test_labels) if data_spec.test_labels is not None else None,
        "class_names": data_spec.class_names,
        "crop_to_primary_object": crop_to_primary_object,
        "full_image_detection": not crop_to_primary_object,
        "classification_target": not detection_mode,
        "classification_object_crops": classification_object_crops,
        "dataset_balance_auto_config": balance_auto_summary,
        "balanced_epoch_exposure": balanced_exposure_summary,
        "train_class_counts": train_class_counts,
        "val_class_counts": val_class_counts,
        "test_class_counts": test_class_counts,
        "total_class_counts": total_class_counts,
        "train_dataset_report": train_dataset_report,
        "rare_class_repeat": rare_class_repeat_summary,
        "hard_sample_repeat": hard_sample_repeat_summary,
        "sample_weight_manifest": sample_weight_summary,
        "class_crop_margin": class_crop_margin_summary,
        "targeted_copy_paste": targeted_copy_paste_summary,
        "val_dataset_report": val_dataset_report,
        "test_dataset_report": test_dataset_report,
    }
    config_payload = {
        "run_name": run_name,
        "run_dir": str(run_dir),
        "data": data_summary,
        "model_config": to_serializable(model_config),
        "train_config": to_serializable(train_config),
        "augmentation_config": to_serializable(augmentation_config),
        "imbalance_summary": to_serializable(imbalance_summary),
        "dataloader": {
            "train": to_serializable(train_dataloader_summary),
            "val": to_serializable(val_dataloader_summary),
            "image_cache": {
                "train": to_serializable(train_cache_summary),
                "val": to_serializable(val_cache_summary),
            },
        },
        "optimizer_name": "SAM(AdamW)" if train_config.use_sam else "AdamW",
        "effective_train_amp": train_amp,
        "effective_amp_dtype": str(effective_amp_dtype).replace("torch.", "") if effective_amp_dtype is not None else None,
        "grad_scaler_enabled": bool(scaler is not None and scaler.is_enabled()),
        "resume": {
            "path": str(resume_checkpoint_path) if resume_checkpoint_path is not None else None,
            "loaded": resume_checkpoint is not None,
            "reset_epoch": bool(args.resume_reset_epoch),
            "reset_optimizer": bool(args.resume_reset_optimizer),
            "reset_scheduler": bool(args.resume_reset_scheduler),
            "reset_scaler": bool(args.resume_reset_scaler),
        },
        "effective_batch_size": effective_batch_size,
        "parameter_count": count_parameters(model),
        "pretrained_distillation": to_serializable(distillation_summary),
        "offline_distillation": to_serializable(offline_distillation_summary),
        "early_learning_regularization": to_serializable(elr_summary),
    }
    json_dump(run_dir / "resolved_config.json", config_payload)
    parameter_count = int(config_payload["parameter_count"])

    history_csv = run_dir / "history.csv"
    resume_progress = _initial_training_progress_from_resume(
        resume_summary,
        resume_checkpoint,
        reset_epoch=bool(args.resume_reset_epoch),
    )
    start_epoch = int(resume_progress["start_epoch"])
    best_macro_f1 = float(resume_progress["best_macro_f1"])
    best_selection_metric_name = str(resume_progress["best_selection_metric_name"])
    best_selection_metric_value = resume_progress["best_selection_metric_value"]
    best_selection_metric_higher_is_better = bool(resume_progress["best_selection_metric_higher_is_better"])
    best_epoch = int(resume_progress["best_epoch"])
    epochs_without_improvement = int(resume_progress["epochs_without_improvement"])
    stage1_auto_advance_epoch = resume_progress.get("stage1_auto_advance_epoch")
    if stage1_auto_advance_epoch is not None:
        stage1_auto_advance_epoch = int(stage1_auto_advance_epoch)
    stage1_best_macro_f1 = -1.0
    stage1_best_epoch = 0
    stage1_boundary_checkpoint_saved = False
    previous_val_metrics: Optional[Dict[str, object]] = (
        resume_progress["previous_val_metrics"]
        if isinstance(resume_progress["previous_val_metrics"], dict)
        else None
    )
    if resume_checkpoint is not None and bool(args.resume_reset_epoch):
        print({"resume_progress_reset": resume_progress}, flush=True)
    last_epoch = max(0, start_epoch - 1)
    stop_reason = "completed"

    overall_start = time.time()
    run_indefinitely = train_config.epochs <= 0 or train_config.epochs >= 9999
    epoch_iterator = count(start_epoch) if run_indefinitely else range(start_epoch, train_config.epochs + 1)
    epoch_total = None if run_indefinitely else max(0, int(train_config.epochs) - int(start_epoch) + 1)

    try:
        with tqdm(epoch_iterator, desc="Epochs", total=epoch_total, dynamic_ncols=True) as epoch_pbar:
            for epoch in epoch_pbar:
                last_epoch = int(epoch)
                epoch_start = time.time()
                train_seconds = 0.0
                val_seconds = 0.0
                checkpoint_seconds = 0.0
                artifact_seconds = 0.0
                cleanup_seconds = 0.0
                if device.type == "cuda":
                    torch.cuda.reset_peak_memory_stats(device)
                zero_based_epoch = int(epoch) - 1
                if train_config.multi_scale_training and hasattr(train_dataset, "set_epoch"):
                    train_dataset.set_epoch(zero_based_epoch)
                    print(
                        {
                            "epoch": epoch,
                            "multi_scale_image_size": train_dataset.get_current_scale(),
                        },
                        flush=True,
                    )

                if optimized_training_scheduler is not None:
                    optimized_training_scheduler.set_epoch(zero_based_epoch)
                    freeze_status = optimized_training_scheduler.apply_progressive_unfreezing()
                    print(
                        {
                            "epoch": epoch,
                            "optimized_scheduler_status": freeze_status,
                        },
                        flush=True,
                    )
                    if optimized_training_scheduler.get_current_stage() == 1:
                        stage_config = {
                            "stage_name": "stage1_cls_only",
                            "cls_weight": 1.0,
                            "bbox_l1_weight": 0.0,
                            "bbox_giou_weight": 0.0,
                            "matcher_class_cost": max(1.0, float(train_config.matcher_class_cost)),
                        }
                    else:
                        stage_config = {
                            "stage_name": "stage2_full_detection",
                            "cls_weight": float(train_config.cls_loss_weight),
                            "bbox_l1_weight": float(train_config.bbox_l1_loss_weight),
                            "bbox_giou_weight": float(train_config.bbox_giou_loss_weight),
                            "matcher_class_cost": float(train_config.matcher_class_cost),
                        }
                else:
                    stage_config = _resolve_detection_stage(
                        train_config=train_config,
                        epoch_index=int(epoch),
                        detection_mode=detection_mode,
                        stage1_auto_advance_epoch=stage1_auto_advance_epoch,
                    )
                classification_guard = _resolve_classification_overfit_guard(
                    train_config=train_config,
                    detection_mode=detection_mode,
                    stage_name=str(stage_config["stage_name"]),
                    previous_val_metrics=previous_val_metrics,
                )
                if classification_guard.get("active"):
                    base_cls_weight = float(stage_config["cls_weight"])
                    stage_config["cls_weight"] = base_cls_weight * float(
                        classification_guard.get("multiplier", 1.0)
                    )
                    print(
                        {
                            "epoch": epoch,
                            "classification_overfit_guard": classification_guard,
                            "base_cls_weight": base_cls_weight,
                            "active_cls_weight": stage_config["cls_weight"],
                        },
                        flush=True,
                    )
                stage_config["classification_overfit_guard"] = classification_guard
                detection_loss_adaptation = _resolve_adaptive_detection_loss(
                    train_config=train_config,
                    detection_mode=detection_mode,
                    stage_name=str(stage_config["stage_name"]),
                    previous_val_metrics=previous_val_metrics,
                )
                if detection_loss_adaptation.get("active"):
                    before_weights = {
                        "bbox_l1_weight": float(stage_config.get("bbox_l1_weight", 0.0) or 0.0),
                        "bbox_giou_weight": float(stage_config.get("bbox_giou_weight", 0.0) or 0.0),
                        "objectness_weight": float(stage_config.get("objectness_weight", 0.0) or 0.0),
                        "cardinality_weight": float(stage_config.get("cardinality_weight", 0.0) or 0.0),
                    }
                    _apply_adaptive_detection_loss(stage_config, detection_loss_adaptation)
                    print(
                        {
                            "epoch": epoch,
                            "adaptive_detection_loss": detection_loss_adaptation,
                            "base_detection_weights": before_weights,
                            "active_detection_weights": {
                                "bbox_l1_weight": float(stage_config.get("bbox_l1_weight", 0.0) or 0.0),
                                "bbox_giou_weight": float(stage_config.get("bbox_giou_weight", 0.0) or 0.0),
                                "objectness_weight": float(stage_config.get("objectness_weight", 0.0) or 0.0),
                                "cardinality_weight": float(stage_config.get("cardinality_weight", 0.0) or 0.0),
                            },
                        },
                        flush=True,
                    )
                stage_config["adaptive_detection_loss"] = detection_loss_adaptation
                rare_class_recall_guard = _resolve_rare_class_recall_guard(
                    train_config=train_config,
                    detection_mode=detection_mode,
                    stage_name=str(stage_config["stage_name"]),
                    previous_val_metrics=previous_val_metrics,
                    class_target_scales=class_target_scales,
                )
                if rare_class_recall_guard.get("active"):
                    print(
                        {
                            "epoch": epoch,
                            "rare_class_recall_guard": rare_class_recall_guard,
                        },
                        flush=True,
                    )
                stage_config["rare_class_recall_guard"] = rare_class_recall_guard
                if detection_mode:
                    _configure_detection_criterion(criterion, stage_config)
                    _configure_detection_criterion(eval_criterion, stage_config)
                if hasattr(criterion, "set_class_weight_multipliers"):
                    criterion.set_class_weight_multipliers(
                        torch.tensor(
                            rare_class_recall_guard.get("multipliers", [1.0] * data_spec.num_classes),
                            dtype=torch.float32,
                            device=device,
                        )
                    )
                if eval_criterion is not criterion and hasattr(eval_criterion, "reset_class_weight_multipliers"):
                    eval_criterion.reset_class_weight_multipliers()

                train_phase_start = time.time()
                train_loss, train_artifact_stats, current_lr = train_one_epoch(
                    model=model,
                    dataloader=train_loader,
                    criterion=criterion,
                    optimizer=optimizer,
                    scheduler=scheduler,
                    scaler=scaler,
                    device=device,
                    amp=train_amp,
                    grad_clip_norm=train_config.grad_clip_norm,
                    epoch_index=int(epoch),
                    metric_learning_criterion=metric_learning_criterion,
                    metric_learning_loss_weight=train_config.metric_learning_loss_weight,
                    metric_learning_sources=train_config.metric_learning_sources,
                    boundary_contrastive_loss_weight=train_config.boundary_contrastive_loss_weight,
                    boundary_contrastive_pairs=train_config.boundary_contrastive_pairs,
                    boundary_contrastive_sources=train_config.boundary_contrastive_sources,
                    boundary_contrastive_margin=train_config.boundary_contrastive_margin,
                    boundary_contrastive_temperature=train_config.boundary_contrastive_temperature,
                    boundary_contrastive_max_pairs=train_config.boundary_contrastive_max_pairs,
                    foreground_consistency_loss_weight=train_config.foreground_consistency_loss_weight,
                    foreground_consistency_margin=train_config.foreground_consistency_margin,
                    attention_view_loss_weight=train_config.attention_view_loss_weight,
                    attention_crop_probability=train_config.attention_crop_probability,
                    attention_drop_probability=train_config.attention_drop_probability,
                    attention_view_start_epoch=train_config.attention_view_start_epoch,
                    attention_crop_threshold=train_config.attention_crop_threshold,
                    attention_drop_threshold=train_config.attention_drop_threshold,
                    attention_crop_padding_ratio=train_config.attention_crop_padding_ratio,
                    attention_crop_min_area_ratio=train_config.attention_crop_min_area_ratio,
                    attention_view_foreground_weight=train_config.attention_view_foreground_weight,
                    attention_view_score_source=train_config.attention_view_score_source,
                    attention_drop_blur_kernel=train_config.attention_drop_blur_kernel,
                    attention_drop_dilation_kernel=train_config.attention_drop_dilation_kernel,
                    attention_drop_min_area_ratio=train_config.attention_drop_min_area_ratio,
                    attention_drop_max_area_ratio=train_config.attention_drop_max_area_ratio,
                    register_diversity_loss_weight=train_config.register_diversity_loss_weight,
                    pairwise_margin_loss_weight=train_config.pairwise_margin_loss_weight,
                    ordinal_maturity_loss_weight=train_config.ordinal_maturity_loss_weight,
                    distillation_teacher=distillation_teacher,
                    distillation_class_indices=distillation_class_indices,
                    distillation_loss_weight=train_config.distillation_weight,
                    distillation_temperature=train_config.distillation_temperature,
                    distillation_focus_class_index=train_config.distillation_focus_class_index,
                    distillation_focus_class_weight=train_config.distillation_focus_class_weight,
                    elr_target_state=elr_target_state,
                    elr_loss_weight=train_config.elr_loss_weight,
                    elr_beta=train_config.elr_beta,
                    elr_start_epoch=train_config.elr_start_epoch,
                    use_sam=train_config.use_sam,
                    grad_accum_steps=train_config.grad_accum_steps,
                    max_nonfinite_grad_steps=train_config.max_nonfinite_grad_steps,
                    max_batches=train_config.max_train_batches,
                    debug_bbox=(args.debug_loss or args.debug_iou),
                    model_ema=model_ema,
                )
                train_seconds = time.time() - train_phase_start
                val_phase_start = time.time()
                validation_model = model_ema.module if model_ema is not None else model
                val_metrics = evaluate_model(
                    model=validation_model,
                    dataloader=val_loader,
                    device=device,
                    class_names=data_spec.class_names,
                    criterion=eval_criterion,
                    amp=train_config.amp,
                    max_batches=train_config.max_val_batches or None,
                    collect_artifact_stats=train_config.log_artifact_stats,
                    tta=train_config.eval_tta,
                    tta_brightness_delta=train_config.tta_brightness_delta,
                    confidence_threshold=None,
                    detection_nms_iou_threshold=train_config.eval_detection_nms_iou_threshold,
                    max_detections_per_image=(
                        train_config.eval_max_detections_per_image
                        if train_config.eval_max_detections_per_image > 0
                        else None
                    ),
                    require_foreground_argmax=train_config.eval_require_foreground_argmax,
                    detection_score_mode=train_config.eval_detection_score_mode,
                    adaptive_max_detections=train_config.eval_adaptive_max_detections,
                    adaptive_count_source=train_config.eval_adaptive_count_source,
                    adaptive_count_margin=train_config.eval_adaptive_count_margin,
                    adaptive_min_detections=train_config.eval_adaptive_min_detections,
                )
                val_seconds = time.time() - val_phase_start
                val_timing = val_metrics.get("timing", {}) if isinstance(val_metrics.get("timing", {}), dict) else {}
                current_macro_f1 = float(val_metrics["macro_f1"])
                if current_macro_f1 > best_macro_f1:
                    best_macro_f1 = current_macro_f1
                selection_metric_name, selection_metric_value, selection_metric_higher_is_better = (
                    _resolve_checkpoint_selection(
                        model_config=model_config,
                        train_config=train_config,
                        metrics=val_metrics,
                    )
                )
                eligible_for_best = _is_best_checkpoint_candidate(
                    model_config=model_config,
                    stage_name=str(stage_config["stage_name"]),
                )
                if not eligible_for_best:
                    is_improved = False
                elif best_selection_metric_value is None:
                    is_improved = True
                elif selection_metric_higher_is_better:
                    is_improved = selection_metric_value > float(best_selection_metric_value)
                else:
                    is_improved = selection_metric_value < float(best_selection_metric_value)
                is_stage1_epoch = detection_mode and str(stage_config["stage_name"]) == "stage1_cls_only"
                stage1_auto_advance_triggered = False
                stage1_auto_threshold = float(
                    getattr(train_config, "stage1_auto_advance_macro_f1_threshold", 0.0) or 0.0
                )
                stage1_auto_min_epochs = max(
                    1,
                    int(getattr(train_config, "stage1_auto_advance_min_epochs", 1) or 1),
                )
                if (
                    is_stage1_epoch
                    and stage1_auto_advance_epoch is None
                    and stage1_auto_threshold > 0.0
                    and int(epoch) >= stage1_auto_min_epochs
                    and current_macro_f1 >= stage1_auto_threshold
                ):
                    stage1_auto_advance_epoch = int(epoch)
                    stage1_auto_advance_triggered = True
                    print(
                        {
                            "stage1_auto_advance": "triggered",
                            "epoch": int(epoch),
                            "macro_f1": round(float(current_macro_f1), 6),
                            "threshold": float(stage1_auto_threshold),
                            "next_stage": "stage2_full_detection",
                        },
                        flush=True,
                    )
                stage1_boundary_reached = bool(
                    is_stage1_epoch
                    and (
                        int(epoch) >= int(train_config.stage1_epochs)
                        or stage1_auto_advance_triggered
                    )
                )
                val_detection = val_metrics.get("detection", {})
                val_detection_curve = val_metrics.get("detection_confidence_curve", {})
                row = {
                    "epoch": epoch,
                    "learning_rate": current_lr,
                    "train_stage": stage_config["stage_name"],
                    "train_loss": train_loss,
                    "train_cls_loss": train_artifact_stats.get("cls_loss", 0.0),
                    "train_metric_learning_loss": train_artifact_stats.get("metric_learning_loss", 0.0),
                    "train_boundary_contrastive_loss": train_artifact_stats.get(
                        "boundary_contrastive_loss",
                        0.0,
                    ),
                    "train_boundary_contrastive_terms": train_artifact_stats.get(
                        "boundary_contrastive_terms",
                        0.0,
                    ),
                    "train_foreground_consistency_loss": train_artifact_stats.get(
                        "foreground_consistency_loss",
                        0.0,
                    ),
                    "train_attention_view_loss": train_artifact_stats.get(
                        "attention_view_loss",
                        0.0,
                    ),
                    "train_attention_view_fraction": train_artifact_stats.get(
                        "attention_view_fraction",
                        0.0,
                    ),
                    "train_attention_crop_fraction": train_artifact_stats.get(
                        "attention_crop_fraction",
                        0.0,
                    ),
                    "train_attention_drop_fraction": train_artifact_stats.get(
                        "attention_drop_fraction",
                        0.0,
                    ),
                    "train_attention_drop_area_fraction": train_artifact_stats.get(
                        "attention_drop_area_fraction",
                        0.0,
                    ),
                    "train_register_diversity_loss": train_artifact_stats.get(
                        "register_diversity_loss",
                        0.0,
                    ),
                    "train_pairwise_margin_loss": train_artifact_stats.get("pairwise_margin_loss", 0.0),
                    "train_ordinal_maturity_loss": train_artifact_stats.get(
                        "ordinal_maturity_loss",
                        0.0,
                    ),
                    "train_distillation_loss": train_artifact_stats.get(
                        "distillation_loss",
                        0.0,
                    ),
                    "train_elr_loss": train_artifact_stats.get("elr_loss", 0.0),
                    "train_sample_weight_mean": train_artifact_stats.get("sample_weight_mean", 1.0),
                    "train_objectness_loss": train_artifact_stats.get("objectness_loss", 0.0),
                    "train_bbox_l1_loss": train_artifact_stats.get("bbox_l1_loss", 0.0),
                    "train_bbox_giou_loss": train_artifact_stats.get("bbox_giou_loss", 0.0),
                    "train_cardinality_loss": train_artifact_stats.get("cardinality_loss", 0.0),
                    "train_count_loss": train_artifact_stats.get("count_loss", 0.0),
                    "train_quality_loss": train_artifact_stats.get("quality_loss", 0.0),
                    "train_count_objectness_consistency_loss": train_artifact_stats.get(
                        "count_objectness_consistency_loss",
                        0.0,
                    ),
                    "train_auxiliary_loss": train_artifact_stats.get("auxiliary_loss", 0.0),
                    "train_seconds": train_seconds,
                    "val_seconds": val_seconds,
                    "val_loop_seconds": float(val_timing.get("eval_loop_seconds", 0.0) or 0.0),
                    "val_metric_seconds": float(val_timing.get("metric_seconds", 0.0) or 0.0),
                    "val_detection_prepare_seconds": float(
                        val_timing.get("detection_prepare_seconds", 0.0) or 0.0
                    ),
                    "val_detection_threshold_seconds": float(
                        val_timing.get("detection_threshold_seconds", 0.0) or 0.0
                    ),
                    "val_detection_curve_seconds": float(
                        val_timing.get("detection_curve_seconds", 0.0) or 0.0
                    ),
                    "checkpoint_seconds": checkpoint_seconds,
                    "artifact_seconds": artifact_seconds,
                    "cleanup_seconds": cleanup_seconds,
                    "post_epoch_seconds": 0.0,
                    "train_cls_weight_active": float(stage_config["cls_weight"]),
                    "train_bbox_l1_weight_active": float(stage_config["bbox_l1_weight"]),
                    "train_bbox_giou_weight_active": float(stage_config["bbox_giou_weight"]),
                    "classification_overfit_guard_multiplier": float(
                        classification_guard.get("multiplier", 1.0)
                    ),
                    "classification_overfit_guard_active": int(bool(classification_guard.get("active"))),
                    "adaptive_detection_loss_multiplier": float(
                        detection_loss_adaptation.get("multiplier", 1.0)
                    ),
                    "adaptive_detection_loss_active": int(bool(detection_loss_adaptation.get("active"))),
                    "rare_class_recall_guard_active": int(bool(rare_class_recall_guard.get("active"))),
                    "rare_class_recall_guard_class_count": int(
                        rare_class_recall_guard.get("active_class_count", 0) or 0
                    ),
                    "rare_class_recall_guard_max_multiplier": float(
                        rare_class_recall_guard.get("max_multiplier", 1.0) or 1.0
                    ),
                    "stage1_auto_advance_triggered": int(bool(stage1_auto_advance_triggered)),
                    "stage1_auto_advance_epoch": int(stage1_auto_advance_epoch or 0),
                    "val_loss": val_metrics["loss"],
                    "val_cls_loss": val_metrics.get("loss_components", {}).get("cls_loss", val_metrics["loss"]),
                    "val_objectness_loss": val_metrics.get("loss_components", {}).get("objectness_loss", 0.0),
                    "val_bbox_l1_loss": val_metrics.get("loss_components", {}).get("bbox_l1_loss", 0.0),
                    "val_bbox_giou_loss": val_metrics.get("loss_components", {}).get("bbox_giou_loss", 0.0),
                    "val_cardinality_loss": val_metrics.get("loss_components", {}).get("cardinality_loss", 0.0),
                    "val_count_loss": val_metrics.get("loss_components", {}).get("count_loss", 0.0),
                    "val_quality_loss": val_metrics.get("loss_components", {}).get("quality_loss", 0.0),
                    "val_count_objectness_consistency_loss": val_metrics.get("loss_components", {}).get(
                        "count_objectness_consistency_loss",
                        0.0,
                    ),
                    "val_auxiliary_loss": val_metrics.get("loss_components", {}).get("auxiliary_loss", 0.0),
                    "val_count_mae": val_metrics.get("count", {}).get("mae", 0.0),
                    "val_count_rmse": val_metrics.get("count", {}).get("rmse", 0.0),
                    "val_accuracy": val_metrics["accuracy"],
                    "val_macro_f1": val_metrics["macro_f1"],
                    "val_weighted_f1": val_metrics["weighted_f1"],
                    "val_bbox_iou": val_metrics.get("bbox", {}).get("mean_iou", 0.0),
                    "val_bbox_giou": val_metrics.get("bbox", {}).get("mean_giou", 0.0),
                    "val_detection_precision_50": val_detection.get("precision_50", 0.0),
                    "val_detection_recall_50": val_detection.get("recall_50", 0.0),
                    "val_detection_f1_50": val_detection.get("f1_50", 0.0),
                    "val_best_detection_f1_50": val_detection_curve.get("best_f1_50", 0.0),
                    "val_best_detection_confidence": val_detection_curve.get("best_f1_50_confidence", 0.0),
                    "selection_metric": selection_metric_value,
                    "val_best_confidence": val_metrics.get("confidence_curves", {}).get(
                        "best_macro_f1_confidence",
                        0.0,
                    ),
                    "train_register_patch_ratio": train_artifact_stats["register_to_patch_ratio"],
                    "val_register_patch_ratio": val_metrics.get("artifact_stats", {}).get(
                        "register_to_patch_ratio",
                        0.0,
                    ),
                    "val_high_norm_patch_fraction": val_metrics.get("artifact_stats", {}).get(
                        "high_norm_patch_fraction",
                        0.0,
                    ),
                    "epoch_seconds": time.time() - epoch_start,
                }
                val_background_aware = val_metrics.get("background_aware_classification", {})
                row["val_background_prediction_rate"] = val_background_aware.get(
                    "background_prediction_rate",
                    0.0,
                )
                val_per_class = val_metrics.get("per_class", [])
                val_bgaware_per_class = val_background_aware.get("per_class", [])
                class_fieldnames: List[str] = []
                for class_index, class_name in enumerate(data_spec.class_names):
                    class_metrics = next(
                        (
                            item
                            for item in val_per_class
                            if int(item.get("class_index", class_index)) == class_index
                        ),
                        {},
                    )
                    bg_metrics = next(
                        (
                            item
                            for item in val_bgaware_per_class
                            if int(item.get("class_index", class_index)) == class_index
                        ),
                        {},
                    )
                    prefix = f"val_class_{class_index}"
                    bg_prefix = f"val_bgaware_class_{class_index}"
                    row[f"{prefix}_precision"] = class_metrics.get("precision", 0.0)
                    row[f"{prefix}_recall"] = class_metrics.get("recall", 0.0)
                    row[f"{prefix}_accuracy"] = class_metrics.get("recall", 0.0)
                    row[f"{prefix}_f1"] = class_metrics.get("f1", 0.0)
                    row[f"{prefix}_support"] = class_metrics.get("support", 0)
                    row[f"{bg_prefix}_precision"] = bg_metrics.get("precision", 0.0)
                    row[f"{bg_prefix}_recall"] = bg_metrics.get("recall", 0.0)
                    row[f"{bg_prefix}_f1"] = bg_metrics.get("f1", 0.0)
                    row[f"{bg_prefix}_support"] = bg_metrics.get("support", 0)
                    class_fieldnames.extend(
                        [
                            f"{prefix}_precision",
                            f"{prefix}_recall",
                            f"{prefix}_accuracy",
                            f"{prefix}_f1",
                            f"{prefix}_support",
                            f"{bg_prefix}_precision",
                            f"{bg_prefix}_recall",
                            f"{bg_prefix}_f1",
                            f"{bg_prefix}_support",
                        ]
                    )
                class_aug_scales = getattr(train_dataset, "class_augmentation_scales", [])
                class_aug_fieldnames: List[str] = []
                for class_index in range(len(data_spec.class_names)):
                    field_name = f"train_class_{class_index}_augmentation_scale"
                    row[field_name] = (
                        float(class_aug_scales[class_index])
                        if class_index < len(class_aug_scales)
                        else 1.0
                    )
                    class_aug_fieldnames.append(field_name)
                checkpoint_payload = make_checkpoint_payload(
                    model=model,
                    optimizer=optimizer,
                    epoch=int(epoch),
                    best_macro_f1=best_macro_f1,
                    model_config=model_config,
                    train_config=train_config,
                    augmentation_config=augmentation_config,
                    data_yaml=data_spec.data_yaml,
                    class_names=data_spec.class_names,
                    metrics=val_metrics,
                    data_summary=data_summary,
                    imbalance_summary=imbalance_summary,
                )
                checkpoint_payload["train_stage"] = stage_config["stage_name"]
                checkpoint_payload["classification_overfit_guard"] = classification_guard
                checkpoint_payload["adaptive_detection_loss"] = detection_loss_adaptation
                checkpoint_payload["rare_class_recall_guard"] = rare_class_recall_guard
                checkpoint_payload["selection_metric"] = {
                    "name": selection_metric_name,
                    "value": selection_metric_value,
                    "higher_is_better": selection_metric_higher_is_better,
                }
                checkpoint_payload["best_epoch"] = best_epoch
                checkpoint_payload["best_selection_metric"] = {
                    "name": best_selection_metric_name,
                    "value": best_selection_metric_value,
                    "higher_is_better": best_selection_metric_higher_is_better,
                }
                checkpoint_payload["epochs_without_improvement"] = epochs_without_improvement
                checkpoint_payload["scheduler_state"] = (
                    scheduler.state_dict() if hasattr(scheduler, "state_dict") else None
                )
                checkpoint_payload["scaler_state"] = scaler.state_dict() if scaler is not None else None
                checkpoint_payload["resume_state"] = {
                    "last_completed_epoch": int(epoch),
                    "next_epoch": int(epoch) + 1,
                    "checkpoint_kind": "last",
                }
                checkpoint_payload["stage1_auto_advance_epoch"] = (
                    int(stage1_auto_advance_epoch) if stage1_auto_advance_epoch is not None else None
                )
                checkpoint_payload["stage1_auto_advance"] = {
                    "enabled": bool(stage1_auto_threshold > 0.0),
                    "threshold": float(stage1_auto_threshold),
                    "min_epochs": int(stage1_auto_min_epochs),
                    "triggered_this_epoch": bool(stage1_auto_advance_triggered),
                    "advance_epoch": (
                        int(stage1_auto_advance_epoch) if stage1_auto_advance_epoch is not None else None
                    ),
                }
                checkpoint_payload["validation_weight_source"] = (
                    "ema" if model_ema is not None else "train"
                )
                if model_ema is not None:
                    checkpoint_payload["ema_model_state"] = model_ema.state_dict()
                    checkpoint_payload["ema_updates"] = int(model_ema.updates)
                    checkpoint_payload["ema_decay"] = float(model_ema.decay)
                if is_stage1_epoch:
                    stage1_improved = current_macro_f1 > float(stage1_best_macro_f1)
                    if stage1_improved:
                        stage1_best_macro_f1 = current_macro_f1
                        stage1_best_epoch = int(epoch)
                        stage1_best_payload = dict(checkpoint_payload)
                        stage1_best_payload["optimizer_state"] = None
                        stage1_best_payload["resume_state"] = {
                            "last_completed_epoch": int(epoch),
                            "next_epoch": int(epoch) + 1,
                            "checkpoint_kind": "stage1_best",
                        }
                        stage1_best_payload["stage1_checkpoint"] = {
                            "kind": "stage1_best",
                            "epoch": int(epoch),
                            "macro_f1": float(current_macro_f1),
                            "next_stage": "stage2_full_detection"
                            if stage1_boundary_reached
                            else "stage1_cls_only",
                        }
                        checkpoint_start = time.time()
                        save_checkpoint(checkpoints_dir / "stage1_best.pt", stage1_best_payload)
                        checkpoint_seconds += time.time() - checkpoint_start
                        artifact_start = time.time()
                        save_evaluation_artifacts(val_metrics, data_spec.class_names, run_dir / "stage1_best_val")
                        json_dump(run_dir / "stage1_best_metrics.json", to_serializable(val_metrics))
                        artifact_seconds += time.time() - artifact_start
                        print(
                            {
                                "stage1_checkpoint": "stage1_best",
                                "path": str(checkpoints_dir / "stage1_best.pt"),
                                "epoch": int(epoch),
                                "macro_f1": round(float(current_macro_f1), 4),
                            },
                            flush=True,
                        )
                        del stage1_best_payload
                    if stage1_boundary_reached and not stage1_boundary_checkpoint_saved:
                        stage1_boundary_checkpoint_saved = True
                        stage1_payload = dict(checkpoint_payload)
                        stage1_payload["resume_state"] = {
                            "last_completed_epoch": int(epoch),
                            "next_epoch": int(epoch) + 1,
                            "checkpoint_kind": "stage1_boundary",
                        }
                        stage1_payload["stage1_checkpoint"] = {
                            "kind": "stage1_boundary",
                            "epoch": int(epoch),
                            "macro_f1": float(current_macro_f1),
                            "best_stage1_epoch": int(stage1_best_epoch),
                            "best_stage1_macro_f1": float(stage1_best_macro_f1),
                            "auto_advance_triggered": bool(stage1_auto_advance_triggered),
                            "next_stage": "stage2_full_detection",
                        }
                        checkpoint_start = time.time()
                        save_checkpoint(checkpoints_dir / "stage1.pt", stage1_payload)
                        checkpoint_seconds += time.time() - checkpoint_start
                        json_dump(run_dir / "stage1_metrics.json", to_serializable(val_metrics))
                        print(
                            {
                                "stage1_checkpoint": "stage1_boundary",
                                "path": str(checkpoints_dir / "stage1.pt"),
                                "epoch": int(epoch),
                                "next_epoch": int(epoch) + 1,
                                "next_stage": "stage2_full_detection",
                                "best_stage1_epoch": int(stage1_best_epoch),
                                "best_stage1_macro_f1": round(float(stage1_best_macro_f1), 4),
                            },
                            flush=True,
                        )
                        del stage1_payload
                count_early_stopping = not is_stage1_epoch
                if is_improved:
                    best_selection_metric_name = selection_metric_name
                    best_selection_metric_value = float(selection_metric_value)
                    best_selection_metric_higher_is_better = bool(selection_metric_higher_is_better)
                    best_epoch = int(epoch)
                    epochs_without_improvement = 0
                    best_payload = dict(checkpoint_payload)
                    best_payload["best_macro_f1"] = best_macro_f1
                    best_payload["best_epoch"] = best_epoch
                    best_payload["best_selection_metric"] = {
                        "name": best_selection_metric_name,
                        "value": best_selection_metric_value,
                        "higher_is_better": best_selection_metric_higher_is_better,
                    }
                    best_payload["optimizer_state"] = None
                    if model_ema is not None:
                        best_payload["train_model_state"] = best_payload["model_state"]
                        best_payload["model_state"] = model_ema.state_dict()
                        best_payload["checkpoint_weight_source"] = "ema"
                    checkpoint_start = time.time()
                    save_checkpoint(checkpoints_dir / "best.pt", best_payload)
                    checkpoint_seconds += time.time() - checkpoint_start
                    artifact_start = time.time()
                    save_evaluation_artifacts(val_metrics, data_spec.class_names, run_dir / "best_val")
                    json_dump(run_dir / "best_metrics.json", to_serializable(val_metrics))
                    artifact_seconds += time.time() - artifact_start
                    del best_payload
                else:
                    if count_early_stopping:
                        epochs_without_improvement += 1

                checkpoint_payload["best_epoch"] = best_epoch
                checkpoint_payload["best_macro_f1"] = best_macro_f1
                checkpoint_payload["best_selection_metric"] = {
                    "name": best_selection_metric_name,
                    "value": best_selection_metric_value,
                    "higher_is_better": best_selection_metric_higher_is_better,
                }
                checkpoint_payload["epochs_without_improvement"] = epochs_without_improvement
                if train_config.save_last_checkpoint:
                    checkpoint_start = time.time()
                    save_checkpoint(checkpoints_dir / "last.pt", checkpoint_payload)
                    checkpoint_seconds += time.time() - checkpoint_start

                cleanup_start = time.time()
                _cleanup_device_memory(device)
                cleanup_seconds += time.time() - cleanup_start
                epoch_memory = _device_memory_snapshot(device)
                row.update(epoch_memory)
                row.update(_dataset_cache_snapshot(train_dataset, "train"))
                row.update(_dataset_cache_snapshot(val_dataset, "val"))
                epoch_elapsed = time.time() - epoch_start
                row.update(
                    {
                        "train_seconds": train_seconds,
                        "val_seconds": val_seconds,
                        "val_loop_seconds": float(val_timing.get("eval_loop_seconds", 0.0) or 0.0),
                        "val_metric_seconds": float(val_timing.get("metric_seconds", 0.0) or 0.0),
                        "val_detection_prepare_seconds": float(
                            val_timing.get("detection_prepare_seconds", 0.0) or 0.0
                        ),
                        "val_detection_threshold_seconds": float(
                            val_timing.get("detection_threshold_seconds", 0.0) or 0.0
                        ),
                        "val_detection_curve_seconds": float(
                            val_timing.get("detection_curve_seconds", 0.0) or 0.0
                        ),
                        "checkpoint_seconds": checkpoint_seconds,
                        "artifact_seconds": artifact_seconds,
                        "cleanup_seconds": cleanup_seconds,
                        "post_epoch_seconds": max(0.0, epoch_elapsed - train_seconds - val_seconds),
                        "epoch_seconds": epoch_elapsed,
                    }
                )
                memory_fieldnames = [
                    "gpu_memory_allocated_mb",
                    "gpu_memory_reserved_mb",
                    "gpu_memory_max_allocated_mb",
                    "gpu_memory_max_reserved_mb",
                ]
                cache_fieldnames = [
                    "train_image_cache_enabled",
                    "train_image_cache_items",
                    "train_image_cache_mb",
                    "train_image_cache_hit_rate",
                    "val_image_cache_enabled",
                    "val_image_cache_items",
                    "val_image_cache_mb",
                    "val_image_cache_hit_rate",
                ]
                append_csv_row(
                    history_csv,
                    row=row,
                    fieldnames=[
                        "epoch",
                        "learning_rate",
                        "train_stage",
                        "train_loss",
                        "train_cls_loss",
                        "train_metric_learning_loss",
                        "train_boundary_contrastive_loss",
                        "train_boundary_contrastive_terms",
                        "train_foreground_consistency_loss",
                        "train_attention_view_loss",
                        "train_attention_view_fraction",
                        "train_attention_crop_fraction",
                        "train_attention_drop_fraction",
                        "train_attention_drop_area_fraction",
                        "train_register_diversity_loss",
                        "train_pairwise_margin_loss",
                        "train_ordinal_maturity_loss",
                        "train_distillation_loss",
                        "train_elr_loss",
                        "train_sample_weight_mean",
                        "train_objectness_loss",
                        "train_bbox_l1_loss",
                        "train_bbox_giou_loss",
                        "train_cardinality_loss",
                        "train_count_loss",
                        "train_seconds",
                        "val_seconds",
                        "val_loop_seconds",
                        "val_metric_seconds",
                        "val_detection_prepare_seconds",
                        "val_detection_threshold_seconds",
                        "val_detection_curve_seconds",
                        "checkpoint_seconds",
                        "artifact_seconds",
                        "cleanup_seconds",
                        "post_epoch_seconds",
                        "train_cls_weight_active",
                        "train_bbox_l1_weight_active",
                        "train_bbox_giou_weight_active",
                        "classification_overfit_guard_multiplier",
                        "classification_overfit_guard_active",
                        "adaptive_detection_loss_multiplier",
                        "adaptive_detection_loss_active",
                        "rare_class_recall_guard_active",
                        "rare_class_recall_guard_class_count",
                        "rare_class_recall_guard_max_multiplier",
                        "stage1_auto_advance_triggered",
                        "stage1_auto_advance_epoch",
                        "val_loss",
                        "val_cls_loss",
                        "val_objectness_loss",
                        "val_bbox_l1_loss",
                        "val_bbox_giou_loss",
                        "val_cardinality_loss",
                        "val_count_loss",
                        "val_count_mae",
                        "val_count_rmse",
                        "val_accuracy",
                        "val_macro_f1",
                        "val_weighted_f1",
                        "val_bbox_iou",
                        "val_bbox_giou",
                        "val_detection_precision_50",
                        "val_detection_recall_50",
                        "val_detection_f1_50",
                        "val_best_detection_f1_50",
                        "val_best_detection_confidence",
                        "selection_metric",
                        "val_best_confidence",
                        "val_background_prediction_rate",
                        "train_register_patch_ratio",
                        "val_register_patch_ratio",
                        "val_high_norm_patch_fraction",
                        "epoch_seconds",
                    ]
                    + memory_fieldnames
                    + cache_fieldnames
                    + class_fieldnames
                    + class_aug_fieldnames,
                )

                epoch_pbar.set_postfix(
                    train_loss=f"{train_loss:.4f}",
                    val_loss=f"{float(val_metrics['loss']):.4f}",
                    val_f1=f"{current_macro_f1:.4f}",
                )
                print(
                    {
                        "epoch": epoch,
                        "stage": stage_config["stage_name"],
                        "train_loss": round(train_loss, 4),
                        "val_loss": round(float(val_metrics["loss"]), 4),
                        "val_acc": round(float(val_metrics["accuracy"]), 4),
                        "val_macro_f1": round(current_macro_f1, 4),
                        "val_bbox_iou": round(float(val_metrics.get("bbox", {}).get("mean_iou", 0.0)), 4),
                        "val_detection_f1_50": round(float(val_detection.get("f1_50", 0.0)), 4),
                        "val_best_detection_f1_50": round(
                            float(val_detection_curve.get("best_f1_50", 0.0)),
                            4,
                        ),
                        "val_count_mae": round(float(val_metrics.get("count", {}).get("mae", 0.0)), 4),
                        "selection_metric_name": selection_metric_name,
                        "selection_metric_value": round(float(selection_metric_value), 4),
                        "eligible_for_best": eligible_for_best,
                        "classification_overfit_guard": classification_guard,
                        "rare_class_recall_guard": rare_class_recall_guard,
                        "val_best_conf": round(
                            float(
                                val_metrics.get("confidence_curves", {}).get(
                                    "best_macro_f1_confidence",
                                    0.0,
                                )
                            ),
                            4,
                        ),
                        "val_reg_patch_ratio": round(
                            float(
                                val_metrics.get("artifact_stats", {}).get(
                                    "register_to_patch_ratio",
                                    0.0,
                                )
                            ),
                            4,
                        ),
                        "best_epoch": best_epoch,
                        "best_macro_f1": round(best_macro_f1, 4),
                        "best_selection_metric_name": best_selection_metric_name,
                        "best_selection_metric_value": round(float(best_selection_metric_value), 4)
                        if best_selection_metric_value is not None
                        else None,
                        "effective_batch_size": effective_batch_size,
                        "lr": current_lr,
                        "timing": {
                            "train": round(float(train_seconds), 2),
                            "val": round(float(val_seconds), 2),
                            "val_loop": round(float(val_timing.get("eval_loop_seconds", 0.0) or 0.0), 2),
                            "val_metric": round(float(val_timing.get("metric_seconds", 0.0) or 0.0), 2),
                            "val_detection_prepare": round(
                                float(val_timing.get("detection_prepare_seconds", 0.0) or 0.0),
                                2,
                            ),
                            "val_detection_curve": round(
                                float(val_timing.get("detection_curve_seconds", 0.0) or 0.0),
                                2,
                            ),
                            "checkpoint": round(float(checkpoint_seconds), 2),
                            "artifact": round(float(artifact_seconds), 2),
                            "cleanup": round(float(cleanup_seconds), 2),
                            "post_epoch": round(float(row.get("post_epoch_seconds", 0.0)), 2),
                        },
                        "image_cache": {
                            "train_hit_rate": round(float(row.get("train_image_cache_hit_rate", 0.0)), 4),
                            "val_hit_rate": round(float(row.get("val_image_cache_hit_rate", 0.0)), 4),
                            "train_mb": round(float(row.get("train_image_cache_mb", 0.0)), 1),
                            "val_mb": round(float(row.get("val_image_cache_mb", 0.0)), 1),
                        },
                        "elapsed": format_seconds(time.time() - overall_start),
                    },
                    flush=True,
                )

                previous_val_metrics = val_metrics
                del val_metrics
                del checkpoint_payload
                del row

                if (
                    train_config.early_stopping_patience > 0
                    and count_early_stopping
                    and epochs_without_improvement >= train_config.early_stopping_patience
                ):
                    model.zero_grad(set_to_none=True)
                    optimizer.zero_grad(set_to_none=True)
                    _cleanup_device_memory(device, heavy=True)
                    stop_reason = "early_stopping"
                    break
    except KeyboardInterrupt:
        stop_reason = "keyboard_interrupt"
        if last_epoch > 0:
            try:
                bad_parameter = _first_nonfinite_parameter(model)
                if bad_parameter is not None:
                    print(
                        {
                            "keyboard_interrupt_checkpoint": "skipped",
                            "reason": "model_has_nonfinite_parameters",
                            "first_bad_parameter": bad_parameter[0],
                            "invalid_entries": bad_parameter[1],
                            "recommendation": "resume from checkpoints/best.pt, not last.pt or interrupt.pt",
                        },
                        flush=True,
                    )
                    raise RuntimeError("Skip interrupt checkpoint because model parameters are non-finite.")
                interrupt_path = checkpoints_dir / "interrupt.pt"
                interrupt_payload = _save_interrupt_checkpoint(
                    path=interrupt_path,
                    model=model,
                    optimizer=optimizer,
                    scheduler=scheduler,
                    scaler=scaler,
                    epoch=int(last_epoch),
                    model_config=model_config,
                    train_config=train_config,
                    augmentation_config=augmentation_config,
                    data_yaml=data_spec.data_yaml,
                    class_names=data_spec.class_names,
                    metrics=previous_val_metrics,
                    data_summary=data_summary,
                    imbalance_summary=imbalance_summary,
                    best_macro_f1=best_macro_f1,
                    best_epoch=best_epoch,
                    best_selection_metric_name=best_selection_metric_name,
                    best_selection_metric_value=best_selection_metric_value,
                    best_selection_metric_higher_is_better=best_selection_metric_higher_is_better,
                    epochs_without_improvement=epochs_without_improvement,
                    stage1_auto_advance_epoch=stage1_auto_advance_epoch,
                    model_ema=model_ema,
                )
                save_checkpoint(checkpoints_dir / "last.pt", interrupt_payload)
                print(
                    {
                        "keyboard_interrupt_checkpoint": str(interrupt_path.resolve()),
                        "last_checkpoint_updated": str((checkpoints_dir / "last.pt").resolve()),
                        "resume_next_epoch": int(last_epoch) + 1,
                    },
                    flush=True,
                )
            except Exception as exc:
                print(
                    {
                        "keyboard_interrupt_checkpoint": "failed",
                        "error": repr(exc),
                    },
                    flush=True,
                )

    model.zero_grad(set_to_none=True)
    optimizer.zero_grad(set_to_none=True)
    del optimizer
    del scheduler
    del scaler
    del eval_criterion
    del criterion
    del class_weights
    del train_loader
    del val_loader
    del model_ema
    del model
    _cleanup_device_memory(device, heavy=True)

    plot_training_history(history_csv, run_dir / "training_curves.png")
    plot_training_history(history_csv, run_dir / "results.png")
    plot_all_training_metrics(history_csv, run_dir / "all_training_metrics.png")
    plot_per_class_training_metrics(
        history_csv,
        data_spec.class_names,
        run_dir / "per_class_training_metrics.png",
    )
    plot_per_class_validation_metric(
        history_csv,
        data_spec.class_names,
        run_dir / "per_class_val_f1.png",
        metric="f1",
        title="Per-Class Validation F1",
        ylabel="F1",
        target=0.97,
    )
    plot_per_class_validation_metric(
        history_csv,
        data_spec.class_names,
        run_dir / "per_class_val_accuracy.png",
        metric="accuracy",
        title="Per-Class Validation Accuracy",
        ylabel="Accuracy / TP-rate",
        target=0.97,
    )
    plot_detection_training_metrics(history_csv, run_dir / "detection_training_metrics.png")
    plot_validation_convergence(history_csv, run_dir / "validation_convergence.png")

    test_summary = None
    if (not args.skip_final_test) and data_spec.has_test_split and (checkpoints_dir / "best.pt").exists():
        _cleanup_device_memory(device, heavy=True)
        best_checkpoint = load_checkpoint(checkpoints_dir / "best.pt", map_location="cpu")
        best_model = build_model_from_checkpoint(
            checkpoint=best_checkpoint,
            num_classes=data_spec.num_classes,
        )
        best_model.to(device)
        best_model.eval()
        test_loader, test_dataloader_summary, test_cache_summary = _build_eval_loader(
            dataset=test_dataset,
            batch_size=train_config.batch_size,
            num_workers=_resolve_eval_num_workers(train_config),
            device=device,
            image_cache_mb=(
                int(train_config.eval_image_cache_mb)
                if int(train_config.eval_image_cache_mb) >= 0
                else None
            ),
        )
        print(
            "Test DataLoader setup:",
            {
                "test": test_dataloader_summary,
                "image_cache": test_cache_summary,
            },
            flush=True,
        )
        calibration = best_checkpoint.get("calibration", {})
        confidence_threshold = calibration.get(
            "best_detection_f1_confidence",
            calibration.get("best_macro_f1_confidence"),
        )
        if best_checkpoint.get("model_config", {}).get("model_type") in DETECTION_MODEL_TYPES:
            test_criterion = HybridDetectionClassificationLoss(
                num_classes=data_spec.num_classes,
                label_smoothing=float(best_checkpoint.get("train_config", {}).get("label_smoothing", 0.0)),
                cls_weight=float(best_checkpoint.get("train_config", {}).get("cls_loss_weight", 1.0)),
                bbox_l1_weight=float(best_checkpoint.get("train_config", {}).get("bbox_l1_loss_weight", 1.0)),
                bbox_giou_weight=float(best_checkpoint.get("train_config", {}).get("bbox_giou_loss_weight", 0.5)),
                background_weight=float(best_checkpoint.get("train_config", {}).get("background_loss_weight", 0.3)),
                objectness_weight=float(best_checkpoint.get("train_config", {}).get("objectness_loss_weight", 5.0)),
                objectness_focal_alpha=float(best_checkpoint.get("train_config", {}).get("objectness_focal_alpha", 0.75)),
                objectness_focal_gamma=float(best_checkpoint.get("train_config", {}).get("objectness_focal_gamma", 0.5)),
                matcher_class_cost=float(best_checkpoint.get("train_config", {}).get("matcher_class_cost", 1.0)),
                matcher_objectness_cost=float(best_checkpoint.get("train_config", {}).get("matcher_objectness_cost", 1.0)),
                cardinality_weight=float(best_checkpoint.get("train_config", {}).get("cardinality_loss_weight", 0.0)),
                count_weight=float(best_checkpoint.get("train_config", {}).get("count_loss_weight", 0.0)),
                quality_weight=float(best_checkpoint.get("train_config", {}).get("quality_loss_weight", 0.0)),
                auxiliary_weight=0.0,
                count_objectness_consistency_weight=float(
                    best_checkpoint.get("train_config", {}).get(
                        "count_objectness_consistency_weight",
                        0.0,
                    )
                ),
            )
        else:
            test_criterion = nn.CrossEntropyLoss()
        test_metrics = evaluate_model(
            model=best_model,
            dataloader=test_loader,
            device=device,
            class_names=data_spec.class_names,
            criterion=test_criterion,
            amp=train_config.amp,
            max_batches=None,
            collect_artifact_stats=train_config.log_artifact_stats,
            tta=train_config.eval_tta,
            tta_brightness_delta=train_config.tta_brightness_delta,
            confidence_threshold=float(confidence_threshold) if confidence_threshold is not None else None,
            detection_nms_iou_threshold=float(
                best_checkpoint.get("train_config", {}).get(
                    "eval_detection_nms_iou_threshold",
                    train_config.eval_detection_nms_iou_threshold,
                )
            ),
            max_detections_per_image=(
                int(
                    best_checkpoint.get("train_config", {}).get(
                        "eval_max_detections_per_image",
                        train_config.eval_max_detections_per_image,
                    )
                )
                if int(
                    best_checkpoint.get("train_config", {}).get(
                        "eval_max_detections_per_image",
                        train_config.eval_max_detections_per_image,
                    )
                )
                > 0
                else None
            ),
            require_foreground_argmax=bool(
                best_checkpoint.get("train_config", {}).get(
                    "eval_require_foreground_argmax",
                    train_config.eval_require_foreground_argmax,
                )
            ),
            detection_score_mode=str(
                best_checkpoint.get("train_config", {}).get(
                    "eval_detection_score_mode",
                    train_config.eval_detection_score_mode,
                )
            ),
            adaptive_max_detections=bool(
                best_checkpoint.get("train_config", {}).get(
                    "eval_adaptive_max_detections",
                    train_config.eval_adaptive_max_detections,
                )
            ),
            adaptive_count_source=str(
                best_checkpoint.get("train_config", {}).get(
                    "eval_adaptive_count_source",
                    train_config.eval_adaptive_count_source,
                )
            ),
            adaptive_count_margin=int(
                best_checkpoint.get("train_config", {}).get(
                    "eval_adaptive_count_margin",
                    train_config.eval_adaptive_count_margin,
                )
            ),
            adaptive_min_detections=int(
                best_checkpoint.get("train_config", {}).get(
                    "eval_adaptive_min_detections",
                    train_config.eval_adaptive_min_detections,
                )
            ),
        )
        save_evaluation_artifacts(test_metrics, data_spec.class_names, run_dir / "final_test")
        plot_train_val_final_test_metrics(
            history_csv,
            run_dir / "final_test" / "metrics.json",
            run_dir / "train_val_final_test_curves.png",
            data_spec.class_names,
        )
        test_summary = {
            "macro_f1": float(test_metrics["macro_f1"]),
            "weighted_f1": float(test_metrics["weighted_f1"]),
            "accuracy": float(test_metrics["accuracy"]),
            "confidence_threshold": confidence_threshold,
        }
        if "bbox" in test_metrics:
            test_summary["bbox"] = test_metrics["bbox"]
        if "detection" in test_metrics:
            test_summary["detection"] = test_metrics["detection"]
        if "detection_best_threshold" in test_metrics:
            test_summary["detection_best_threshold"] = test_metrics["detection_best_threshold"]
        if "detection_calibrated" in test_metrics:
            test_summary["detection_calibrated"] = test_metrics["detection_calibrated"]
        if "count" in test_metrics:
            test_summary["count"] = test_metrics["count"]
        del best_model
        del best_checkpoint
        del test_loader
        del test_criterion
        _cleanup_device_memory(device, heavy=True)

    architecture_trace_summary = {"enabled": False}
    if args.trace_architecture:
        _cleanup_device_memory(device, heavy=True)
        architecture_trace_summary = _run_architecture_trace_after_training(
            args=args,
            data_spec=data_spec,
            run_dir=run_dir,
            checkpoints_dir=checkpoints_dir,
            image_size=int(model_config.image_size),
        )

    summary = {
        "best_epoch": best_epoch,
        "best_macro_f1": best_macro_f1,
        "best_selection_metric_name": best_selection_metric_name,
        "best_selection_metric_value": best_selection_metric_value,
        "best_selection_metric_higher_is_better": best_selection_metric_higher_is_better,
        "last_epoch": last_epoch,
        "stop_reason": stop_reason,
        "total_seconds": time.time() - overall_start,
        "parameter_count": parameter_count,
        "run_dir": str(run_dir),
        "test_summary": test_summary,
        "architecture_trace": architecture_trace_summary,
    }
    json_dump(run_dir / "summary.json", to_serializable(summary))
    print(summary, flush=True)


if __name__ == "__main__":
    main()
