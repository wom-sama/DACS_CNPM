from __future__ import annotations

import argparse
import csv
import hashlib
import json
import math
from contextlib import contextmanager
from pathlib import Path
import time
import types
from typing import Dict, Mapping, Optional, Sequence

import matplotlib.pyplot as plt
import numpy as np
from sklearn.metrics import confusion_matrix, precision_recall_fscore_support, roc_auc_score
import torch
from torch import Tensor, nn
import torch.nn.functional as F
from torch.utils.data import DataLoader

from trkh.core.utils import build_safe_dataloader_kwargs, set_seed
from trkh.evaluation.input_normalization import checkpoint_input_normalization
from trkh.evaluation.robustness_eval import (
    IdentityCorruption,
    LightingShift,
    _forward_classification_with_metadata,
)
from trkh.models.model import MultiHeadSelfAttention, build_model_from_checkpoint
from trkh.tools.audit_bbox_logpolar_stem_a0 import (
    _build_dataset,
    _dataset_tree_identity,
    _read_cidt_conditions,
    _sha256,
)
from trkh.tools.audit_counterfactual_illumination_disagreement_readiness import (
    _SelectedConditionDataset,
)


SEED = 20260720
FOCUS_CLASS = 1
RESTRICTED_NEGATIVE_CLASSES = (0, 2, 4)
EXPECTED_TRAIN_ROWS = 9215
EXPECTED_CLASS_COUNTS = (1941, 541, 1920, 2520, 2293)
EXPECTED_FOLD_COUNTS = (1843, 1830, 1828, 1851, 1863)
EXPECTED_PADDED_ROWS = 9204
EXPECTED_FULLY_INVALID_PATCHES = 555088
EXPECTED_CLASS_INVALID_PATCHES = (131824, 35312, 116048, 137264, 134640)
CONDITIONS = (
    ("clean", 1.00, 1.00),
    ("lighting_dim", 0.70, 0.90),
    ("lighting_bright", 1.25, 1.10),
    ("low_contrast", 1.00, 0.65),
)
ROLES = ("control", "candidate", "placebo")
FIXED_VISUAL_INDICES = (1, 2, 3, 46, 0)
PATCH_GRID = (16, 16)
PATCH_COUNT = 256
ATTENTION_LAYERS = 8
ECE_BINS = 15

EXPECTED_HASHES = {
    "protocol": "2b986af2c77a5d74d2546036005d7f7f94b9cf82da5c9d989a17c1e0915f96df",
    "navit_paper": "d4421cab93fe27a49c64120faa46822dc95a563ed70f5495b25112b35619a396",
    "pytorch_source": "ed64d867e31ff52535003579077e9a14bcfadf225e6443f50e4891cb7bff4d67",
    "pytorch_license": "47a26beb94e3f6b333a3677fc85d546f1fdfd2f0b3686c26d2fb5b10e0134165",
    "data": "716e33df24c63a9e9920f97b685199707fb84ab4c7154544f5dd9a3e00d884ef",
    "cidt_summary": "d4891edf2963ab12385b7ce5bdc812ec3e19c5c098acd25c66eb557af541d7ad",
    "cidt_predictions": "2e0993752d58d99ea429bfefe1e2bfe6fa949e45aea1a26cc4bdfee97d4db21c",
    "keeper": "1f49d577240c69dc63c30af70db52ec2aa9da65a17aef1c4b1c09ece6c482677",
    "current_best": "36b9aa1a21b765829acf4c8321be147bd76297de4ccdb8a40e6dee8e37940faf",
    "command_history": "39bd2879ce66fddf36a953021ea1e40f8d9de6cb4334b9b825011b2b8dc98f53",
}


def _repo_root() -> Path:
    return Path(__file__).resolve().parents[2]


def _default_paths() -> Dict[str, Path]:
    root = _repo_root()
    return {
        "protocol": root
        / "docs"
        / "TRKH_5CLASS_VALIDITY_AWARE_ATTENTION_A0_PROTOCOL_20260720.md",
        "navit_paper": Path(
            "D:/DataAI/external_sources/papers/navit_patch_n_pack_arxiv2307.06304.pdf"
        ),
        "pytorch_source": Path(
            "D:/DataAI/external_sources/official/pytorch_v2.6.0_activation.py"
        ),
        "pytorch_license": Path(
            "D:/DataAI/external_sources/official/pytorch_v2.6.0_LICENSE"
        ),
        "data": Path("D:/DataAI/AIEx/newdataset/yolo_f/data.yaml"),
        "cidt_summary": root
        / "runs"
        / "audit_cidt_readiness_full_train_20260714"
        / "summary.json",
        "cidt_predictions": root
        / "runs"
        / "audit_cidt_readiness_full_train_20260714"
        / "predictions_all_conditions.csv",
        "keeper": root
        / "runs"
        / "probe_v8_yolof_pairroute_teacherfocusbinary015_boundarydrop_bboxprior_120b_2e_20260701"
        / "checkpoints"
        / "best.pt",
        "current_best": root
        / "docs"
        / "TRKH_CURRENT_BEST_FULL_TRAIN_COMMANDS_20260706.txt",
        "command_history": root
        / "docs"
        / "TRKH_CURRENT_BEST_COMMAND_UPDATE_HISTORY.txt",
    }


def parse_args(argv: Optional[Sequence[str]] = None) -> argparse.Namespace:
    defaults = _default_paths()
    parser = argparse.ArgumentParser(
        description="Locked train-only validity-aware attention A0."
    )
    parser.add_argument("--data", type=Path, default=defaults["data"])
    parser.add_argument("--checkpoint", type=Path, default=defaults["keeper"])
    parser.add_argument(
        "--cidt-summary", type=Path, default=defaults["cidt_summary"]
    )
    parser.add_argument(
        "--cidt-predictions", type=Path, default=defaults["cidt_predictions"]
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=Path("runs/audit_validity_aware_attention_a0_20260720"),
    )
    parser.add_argument("--batch-size", type=int, default=64)
    parser.add_argument("--num-workers", type=int, default=4)
    parser.add_argument("--torch-threads", type=int, default=8)
    parser.add_argument("--device", choices=("auto", "cpu", "cuda"), default="cuda")
    parser.add_argument("--preflight-only", action="store_true")
    parser.add_argument("--replay-summary", type=Path)
    return parser.parse_args(argv)


def _validate_locked_args(args: argparse.Namespace) -> None:
    if int(args.batch_size) != 64:
        raise ValueError("Locked A0 requires batch-size=64")
    if int(args.num_workers) != 4:
        raise ValueError("Locked A0 requires num-workers=4")
    if int(args.torch_threads) != 8:
        raise ValueError("Locked A0 requires torch-threads=8")
    output = Path(args.output_dir).resolve()
    dataset_root = Path(args.data).resolve().parent
    try:
        output.relative_to(dataset_root)
    except ValueError:
        pass
    else:
        raise ValueError("Output directory must remain outside the raw dataset")


def _resolve_device(value: str) -> torch.device:
    requested = str(value)
    if requested == "auto":
        requested = "cuda" if torch.cuda.is_available() else "cpu"
    if requested == "cuda" and not torch.cuda.is_available():
        raise RuntimeError("CUDA was requested but is unavailable")
    return torch.device(requested)


def verify_provenance(args: argparse.Namespace) -> Dict[str, object]:
    defaults = _default_paths()
    paths = {
        **defaults,
        "data": Path(args.data),
        "cidt_summary": Path(args.cidt_summary),
        "cidt_predictions": Path(args.cidt_predictions),
        "keeper": Path(args.checkpoint),
    }
    files: Dict[str, object] = {}
    for name, expected in EXPECTED_HASHES.items():
        path = Path(paths[name]).resolve()
        if not path.is_file():
            raise FileNotFoundError(f"Missing locked provenance file: {path}")
        observed = _sha256(path)
        if observed != expected:
            raise ValueError(
                f"Locked provenance mismatch for {name}: {observed} != {expected}"
            )
        files[name] = {"path": str(path), "sha256": observed, "matched": True}
    if not str(torch.__version__).startswith("2.6.0"):
        raise ValueError(f"Locked A0 requires torch 2.6.0, got {torch.__version__}")
    return {
        "files": files,
        "torch_version": str(torch.__version__),
        "all_exact": True,
    }


def _model_state_sha256(model: nn.Module) -> str:
    digest = hashlib.sha256()
    for name, value in sorted(model.state_dict().items()):
        tensor = value.detach().cpu().contiguous()
        digest.update(name.encode("utf-8"))
        digest.update(str(tensor.dtype).encode("ascii"))
        digest.update(str(tuple(tensor.shape)).encode("ascii"))
        digest.update(tensor.numpy().tobytes())
    return digest.hexdigest()


def build_patch_key_padding_mask(image_valid_mask: Tensor) -> Tensor:
    if image_valid_mask.ndim == 3:
        mask = image_valid_mask.unsqueeze(1)
    elif image_valid_mask.ndim == 4:
        mask = image_valid_mask
    else:
        raise ValueError("image_valid_mask must have shape [B,H,W] or [B,1,H,W]")
    valid_fraction = F.adaptive_avg_pool2d(
        mask.to(dtype=torch.float32), PATCH_GRID
    ).flatten(1)
    key_padding_mask = valid_fraction <= 0.05
    all_masked = key_padding_mask.all(dim=1)
    if bool(all_masked.any().item()):
        key_padding_mask = key_padding_mask.clone()
        key_padding_mask[all_masked] = False
    return key_padding_mask


def build_same_class_fold_source_derangement(
    target: np.ndarray,
    fold: np.ndarray,
    source: np.ndarray,
    *,
    seed: int = SEED,
) -> np.ndarray:
    target = np.asarray(target, dtype=np.int64)
    fold = np.asarray(fold, dtype=np.int64)
    source = np.asarray(source, dtype=object)
    if not (target.shape == fold.shape == source.shape) or target.ndim != 1:
        raise ValueError("target/fold/source must be aligned one-dimensional arrays")
    mapping = np.full(target.size, -1, dtype=np.int64)
    rng = np.random.default_rng(int(seed))
    for fold_index in sorted(np.unique(fold).tolist()):
        for class_index in sorted(np.unique(target).tolist()):
            rows = np.flatnonzero((fold == fold_index) & (target == class_index))
            if rows.size < 2:
                raise ValueError("Every fold/class partition needs at least two rows")
            donor = None
            for _ in range(10000):
                candidate = rng.permutation(rows)
                if np.all(source[candidate] != source[rows]):
                    donor = candidate
                    break
            if donor is None:
                raise RuntimeError(
                    f"Could not derange fold={fold_index} class={class_index}"
                )
            mapping[rows] = donor
    if bool((mapping < 0).any()):
        raise RuntimeError("Derangement mapping is incomplete")
    if not np.array_equal(target[mapping], target):
        raise RuntimeError("Derangement changed target")
    if not np.array_equal(fold[mapping], fold):
        raise RuntimeError("Derangement changed fold")
    if bool(np.any(source[mapping] == source)):
        raise RuntimeError("Derangement retained at least one source")
    if np.unique(mapping).size != mapping.size:
        raise RuntimeError("Derangement is not bijective")
    return mapping


class ValidityAwareAttentionController:
    def __init__(self, model: nn.Module) -> None:
        self.model = model
        self.modules: list[tuple[int, MultiHeadSelfAttention]] = []
        for layer_index, block in enumerate(getattr(model, "blocks", [])):
            attention = getattr(block, "attn", None)
            if not isinstance(attention, MultiHeadSelfAttention):
                raise TypeError(
                    f"Layer {layer_index + 1} is not standard MultiHeadSelfAttention"
                )
            if attention.relative_position_attention is not None:
                raise ValueError("Locked A0 forbids relative-position attention mixing")
            self.modules.append((layer_index, attention))
        if len(self.modules) != ATTENTION_LAYERS:
            raise ValueError(f"Locked A0 requires {ATTENTION_LAYERS} attention layers")
        self.patch_mask: Optional[Tensor] = None
        self.sample_indices: Optional[Tensor] = None
        self.collect = False
        self.capture_visuals = False
        self.records: Dict[int, Dict[str, list[Tensor]]] = {}
        self.visual_maps: Dict[int, Dict[str, np.ndarray]] = {}

    def reset(self, *, collect: bool, capture_visuals: bool = False) -> None:
        self.collect = bool(collect)
        self.capture_visuals = bool(capture_visuals)
        self.records = {
            layer: {"invalid_count": [], "pre_mass": [], "post_mass": []}
            for layer, _module in self.modules
        }
        self.visual_maps = {}

    def set_batch(self, patch_mask: Tensor, sample_indices: Tensor) -> None:
        if patch_mask.ndim != 2 or int(patch_mask.size(1)) != PATCH_COUNT:
            raise ValueError("patch_mask must have shape [B,256]")
        if sample_indices.ndim != 1 or int(sample_indices.size(0)) != int(
            patch_mask.size(0)
        ):
            raise ValueError("sample_indices must align with patch_mask")
        self.patch_mask = patch_mask.to(dtype=torch.bool)
        self.sample_indices = sample_indices.to(dtype=torch.long)

    def clear_batch(self) -> None:
        self.patch_mask = None
        self.sample_indices = None

    def _forward(
        self,
        layer_index: int,
        module: MultiHeadSelfAttention,
        x: Tensor,
        return_attention: bool = False,
        *,
        grid_size=None,
        prefix_count: int = 0,
        patch_indices: Optional[Tensor] = None,
    ):
        del grid_size
        if self.patch_mask is None or self.sample_indices is None:
            raise RuntimeError("Validity-aware attention batch context is unset")
        if patch_indices is None or patch_indices.ndim != 2:
            raise ValueError("Locked A0 requires two-dimensional patch_indices")
        batch_size, num_tokens, dim = x.shape
        if int(batch_size) != int(self.patch_mask.size(0)):
            raise ValueError("Attention batch and validity mask batch differ")
        gathered = self.patch_mask.gather(
            1, patch_indices.to(device=self.patch_mask.device, dtype=torch.long)
        )
        prefix_count = int(prefix_count)
        if prefix_count + int(gathered.size(1)) != int(num_tokens):
            raise ValueError("Prefix plus patch mask length differs from token count")
        prefix_mask = torch.zeros(
            (batch_size, prefix_count), device=x.device, dtype=torch.bool
        )
        full_mask = torch.cat((prefix_mask, gathered.to(device=x.device)), dim=1)

        qkv = module.qkv(x)
        qkv = qkv.reshape(
            batch_size, num_tokens, 3, module.num_heads, module.head_dim
        ).permute(2, 0, 3, 1, 4)
        query, key, value = qkv[0], qkv[1], qkv[2]
        logits = (query @ key.transpose(-2, -1)) * module.scale
        pre_attention = logits.softmax(dim=-1)
        masked_logits = logits.masked_fill(
            full_mask[:, None, None, :], torch.finfo(logits.dtype).min
        )
        attention = masked_logits.softmax(dim=-1)
        attention = module.attention_dropout(attention)

        if self.collect:
            invalid = full_mask[:, None, :]
            pre_mass = (
                pre_attention[:, :, 0, :] * invalid.to(pre_attention.dtype)
            ).sum(dim=-1).mean(dim=1)
            post_mass = (
                attention[:, :, 0, :] * invalid.to(attention.dtype)
            ).sum(dim=-1).mean(dim=1)
            record = self.records[layer_index]
            record["invalid_count"].append(gathered.sum(dim=1).detach().cpu())
            record["pre_mass"].append(pre_mass.detach().float().cpu())
            record["post_mass"].append(post_mass.detach().float().cpu())
            if self.capture_visuals and layer_index == 0:
                pre_patch = pre_attention[:, :, 0, prefix_count:].mean(dim=1)
                post_patch = attention[:, :, 0, prefix_count:].mean(dim=1)
                for position, sample_index in enumerate(
                    self.sample_indices.detach().cpu().tolist()
                ):
                    if int(sample_index) not in FIXED_VISUAL_INDICES:
                        continue
                    pre_full = pre_patch.new_zeros(PATCH_COUNT)
                    post_full = post_patch.new_zeros(PATCH_COUNT)
                    indices = patch_indices[position].to(
                        device=pre_patch.device, dtype=torch.long
                    )
                    pre_full.scatter_(0, indices, pre_patch[position])
                    post_full.scatter_(0, indices, post_patch[position])
                    self.visual_maps[int(sample_index)] = {
                        "pre": pre_full.reshape(PATCH_GRID).detach().float().cpu().numpy(),
                        "post": post_full.reshape(PATCH_GRID).detach().float().cpu().numpy(),
                    }

        output = attention @ value
        output = output.transpose(1, 2).reshape(batch_size, num_tokens, dim)
        output = module.proj(output)
        output = module.projection_dropout(output)
        if return_attention:
            return output, attention
        return output

    @contextmanager
    def installed(self):
        original: list[tuple[MultiHeadSelfAttention, bool, object]] = []
        try:
            for layer_index, module in self.modules:
                had_instance = "forward" in module.__dict__
                prior = module.__dict__.get("forward")

                def wrapped(
                    bound_module,
                    x,
                    return_attention=False,
                    *,
                    grid_size=None,
                    prefix_count=0,
                    patch_indices=None,
                    _layer_index=layer_index,
                ):
                    return self._forward(
                        _layer_index,
                        bound_module,
                        x,
                        return_attention=return_attention,
                        grid_size=grid_size,
                        prefix_count=prefix_count,
                        patch_indices=patch_indices,
                    )

                original.append((module, had_instance, prior))
                module.forward = types.MethodType(wrapped, module)
            yield self
        finally:
            self.clear_batch()
            for module, had_instance, prior in original:
                if had_instance:
                    module.forward = prior
                else:
                    delattr(module, "forward")

    def arrays(self) -> Dict[str, np.ndarray]:
        result: Dict[str, np.ndarray] = {}
        for layer_index, record in self.records.items():
            for name, chunks in record.items():
                if not chunks:
                    raise RuntimeError(f"Missing layer {layer_index + 1} telemetry {name}")
                result[f"layer_{layer_index + 1}_{name}"] = (
                    torch.cat(chunks).numpy()
                )
        return result


def _condition_corruption(name: str, brightness: float, contrast: float):
    if name == "clean":
        return IdentityCorruption()
    return LightingShift(brightness=float(brightness), contrast=float(contrast))


def _make_loader(
    *,
    base_dataset,
    transform,
    condition: str,
    brightness: float,
    contrast: float,
    batch_size: int,
    num_workers: int,
) -> tuple[DataLoader, Dict[str, object]]:
    dataset = _SelectedConditionDataset(
        base_dataset,
        list(range(EXPECTED_TRAIN_ROWS)),
        corruption=_condition_corruption(condition, brightness, contrast),
        transform=transform,
    )
    kwargs, summary = build_safe_dataloader_kwargs(
        requested_num_workers=int(num_workers),
        requested_pin_memory=True,
        context=f"validity_attention_{condition}",
        prefetch_factor=2,
        persistent_workers=True,
    )
    loader = DataLoader(
        dataset,
        batch_size=int(batch_size),
        shuffle=False,
        drop_last=False,
        **kwargs,
    )
    return loader, summary


def _metadata_patch_mask(metadata: Mapping[str, object], device: torch.device) -> Tensor:
    image_mask = metadata.get("image_mask")
    if not torch.is_tensor(image_mask):
        raise ValueError("Validity-aware attention requires image_mask metadata")
    return build_patch_key_padding_mask(
        image_mask.to(device=device, dtype=torch.bool, non_blocking=True)
    )


def _first_batch_fidelity(
    *,
    model: nn.Module,
    controller: ValidityAwareAttentionController,
    base_dataset,
    transform,
    expected_clean: Mapping[str, np.ndarray],
    device: torch.device,
    batch_size: int,
    num_workers: int,
) -> Dict[str, float]:
    loader, _summary = _make_loader(
        base_dataset=base_dataset,
        transform=transform,
        condition="clean",
        brightness=1.0,
        contrast=1.0,
        batch_size=batch_size,
        num_workers=num_workers,
    )
    images, targets, metadata = next(iter(loader))
    del targets
    images = images.to(device=device, dtype=torch.float32, non_blocking=True)
    sample_indices = metadata["sample_index"].to(device=device, dtype=torch.long)
    with torch.inference_mode():
        raw_logits, _raw_features = _forward_classification_with_metadata(
            model, images, metadata, device=device
        )
        raw_probabilities = F.softmax(raw_logits.float(), dim=1)
        controller.reset(collect=False)
        with controller.installed():
            controller.set_batch(
                torch.zeros(
                    (images.size(0), PATCH_COUNT), device=device, dtype=torch.bool
                ),
                sample_indices,
            )
            all_valid_logits, _all_valid_features = _forward_classification_with_metadata(
                model, images, metadata, device=device
            )
        all_valid_probabilities = F.softmax(all_valid_logits.float(), dim=1)
    expected = torch.from_numpy(
        expected_clean["probabilities"][sample_indices.detach().cpu().numpy()]
    ).to(device=device, dtype=torch.float32)
    return {
        "control_probability_max_abs_difference": float(
            (raw_probabilities - expected).abs().max().item()
        ),
        "all_valid_logit_max_abs_difference": float(
            (raw_logits.float() - all_valid_logits.float()).abs().max().item()
        ),
        "all_valid_probability_max_abs_difference": float(
            (raw_probabilities - all_valid_probabilities).abs().max().item()
        ),
    }


def _infer_role_condition(
    *,
    role: str,
    condition: str,
    brightness: float,
    contrast: float,
    model: nn.Module,
    controller: ValidityAwareAttentionController,
    base_dataset,
    transform,
    expected: Mapping[str, np.ndarray],
    device: torch.device,
    batch_size: int,
    num_workers: int,
    mask_cache: np.ndarray,
    pixel_padded_cache: np.ndarray,
    derangement: np.ndarray,
    visual_payload: Dict[int, Dict[str, np.ndarray]],
    mean: Sequence[float],
    std: Sequence[float],
) -> Dict[str, object]:
    if role not in {"candidate", "placebo"}:
        raise ValueError("Inference role must be candidate or placebo")
    loader, loader_summary = _make_loader(
        base_dataset=base_dataset,
        transform=transform,
        condition=condition,
        brightness=brightness,
        contrast=contrast,
        batch_size=batch_size,
        num_workers=num_workers,
    )
    controller.reset(
        collect=role == "candidate",
        capture_visuals=role == "candidate" and condition == "clean",
    )
    all_probabilities: list[Tensor] = []
    all_indices: list[Tensor] = []
    all_targets: list[Tensor] = []
    processed = 0
    started = time.perf_counter()
    mean_tensor = torch.tensor(mean, device=device).view(1, 3, 1, 1)
    std_tensor = torch.tensor(std, device=device).view(1, 3, 1, 1)
    with controller.installed(), torch.inference_mode():
        for images, targets, metadata in loader:
            images = images.to(device=device, dtype=torch.float32, non_blocking=True)
            sample_indices = metadata["sample_index"].to(
                device=device, dtype=torch.long, non_blocking=True
            )
            real_patch_mask = _metadata_patch_mask(metadata, device)
            indices_np = sample_indices.detach().cpu().numpy().astype(np.int64)
            real_np = real_patch_mask.detach().cpu().numpy().astype(bool)
            image_mask = metadata["image_mask"]
            padded_np = (
                (~image_mask.to(dtype=torch.bool)).flatten(1).any(dim=1).numpy()
            )
            if role == "candidate" and condition == "clean":
                mask_cache[indices_np] = real_np
                pixel_padded_cache[indices_np] = padded_np
            else:
                if not np.array_equal(mask_cache[indices_np], real_np):
                    raise ValueError(
                        f"Patch-mask drift in role={role} condition={condition}"
                    )
                if not np.array_equal(pixel_padded_cache[indices_np], padded_np):
                    raise ValueError(
                        f"Pixel-mask drift in role={role} condition={condition}"
                    )
            if role == "candidate":
                active_mask = real_patch_mask
            else:
                donor = derangement[indices_np]
                active_mask = torch.from_numpy(mask_cache[donor]).to(
                    device=device, dtype=torch.bool
                )
            controller.set_batch(active_mask, sample_indices)
            logits, _features = _forward_classification_with_metadata(
                model, images, metadata, device=device
            )
            probabilities = F.softmax(logits.float(), dim=1)
            all_probabilities.append(probabilities.detach().cpu())
            all_indices.append(sample_indices.detach().cpu())
            all_targets.append(targets.detach().cpu().to(torch.long))

            if role == "candidate" and condition == "clean":
                rgb = (images * std_tensor + mean_tensor).clamp(0.0, 1.0)
                for position, sample_index in enumerate(indices_np.tolist()):
                    if int(sample_index) not in FIXED_VISUAL_INDICES:
                        continue
                    visual_payload[int(sample_index)] = {
                        "image": (
                            rgb[position]
                            .mul(255.0)
                            .round()
                            .byte()
                            .permute(1, 2, 0)
                            .cpu()
                            .numpy()
                        ),
                        "valid_mask": image_mask[position].bool().cpu().numpy(),
                        "target": np.asarray(int(targets[position].item())),
                    }
            controller.clear_batch()
            processed += int(targets.numel())
            if processed % 1024 < int(targets.numel()) or processed == len(loader.dataset):
                print(
                    json.dumps(
                        {
                            "role": role,
                            "condition": condition,
                            "processed": processed,
                            "rows": len(loader.dataset),
                            "elapsed_seconds": time.perf_counter() - started,
                        }
                    ),
                    flush=True,
                )
    indices = torch.cat(all_indices).numpy().astype(np.int64)
    targets = torch.cat(all_targets).numpy().astype(np.int64)
    probabilities = torch.cat(all_probabilities).numpy().astype(np.float32)
    if not np.array_equal(indices, expected["sample_index"]):
        raise ValueError(f"Ordered sample drift in role={role} condition={condition}")
    if not np.array_equal(targets, expected["target"]):
        raise ValueError(f"Target drift in role={role} condition={condition}")
    if probabilities.shape != (EXPECTED_TRAIN_ROWS, 5):
        raise RuntimeError(f"Incomplete probabilities: {probabilities.shape}")
    if not np.isfinite(probabilities).all() or not np.allclose(
        probabilities.sum(axis=1), 1.0, atol=1e-5, rtol=0.0
    ):
        raise ValueError(f"Invalid probabilities in role={role} condition={condition}")
    if role == "candidate" and condition == "clean":
        for sample_index, maps in controller.visual_maps.items():
            visual_payload.setdefault(sample_index, {}).update(maps)
    return {
        "probabilities": probabilities,
        "telemetry": controller.arrays() if role == "candidate" else {},
        "loader": loader_summary,
        "seconds": float(time.perf_counter() - started),
    }


def _ece(target: np.ndarray, probabilities: np.ndarray, bins: int = ECE_BINS) -> float:
    predictions = probabilities.argmax(axis=1)
    confidence = probabilities.max(axis=1)
    correctness = predictions == target
    edges = np.linspace(0.0, 1.0, int(bins) + 1)
    value = 0.0
    for index in range(int(bins)):
        if index == int(bins) - 1:
            selected = (confidence >= edges[index]) & (confidence <= edges[index + 1])
        else:
            selected = (confidence >= edges[index]) & (confidence < edges[index + 1])
        if not bool(selected.any()):
            continue
        value += float(selected.mean()) * abs(
            float(correctness[selected].mean()) - float(confidence[selected].mean())
        )
    return float(value)


def classification_metrics(target: np.ndarray, probabilities: np.ndarray) -> Dict[str, object]:
    target = np.asarray(target, dtype=np.int64)
    probabilities = np.asarray(probabilities, dtype=np.float64)
    predictions = probabilities.argmax(axis=1)
    precision, recall, f1, support = precision_recall_fscore_support(
        target,
        predictions,
        labels=np.arange(5),
        zero_division=0,
    )
    one_hot = np.eye(5, dtype=np.float64)[target]
    return {
        "accuracy": float(np.mean(predictions == target)),
        "macro_f1": float(np.mean(f1)),
        "per_class_precision": precision.tolist(),
        "per_class_recall": recall.tolist(),
        "per_class_f1": f1.tolist(),
        "support": support.astype(np.int64).tolist(),
        "predicted_support": np.bincount(predictions, minlength=5).astype(np.int64).tolist(),
        "confusion_matrix": confusion_matrix(
            target, predictions, labels=np.arange(5)
        ).astype(np.int64).tolist(),
        "nll": float(
            -np.log(np.clip(probabilities[np.arange(target.size), target], 1e-12, 1.0)).mean()
        ),
        "brier": float(np.square(probabilities - one_hot).sum(axis=1).mean()),
        "ece": _ece(target, probabilities),
    }


def transition_events(
    target: np.ndarray,
    control_probabilities: np.ndarray,
    role_probabilities: np.ndarray,
) -> Dict[str, object]:
    target = np.asarray(target, dtype=np.int64)
    control = np.asarray(control_probabilities).argmax(axis=1)
    role = np.asarray(role_probabilities).argmax(axis=1)
    restricted = np.isin(target, RESTRICTED_NEGATIVE_CLASSES)
    removed = restricted & (control == FOCUS_CLASS) & (role != FOCUS_CLASS)
    created = restricted & (control != FOCUS_CLASS) & (role == FOCUS_CLASS)
    events = {
        "corrections": int(np.sum((control != target) & (role == target))),
        "harms": int(np.sum((control == target) & (role != target))),
        "class1_fn_rescues": int(
            np.sum((target == FOCUS_CLASS) & (control != FOCUS_CLASS) & (role == FOCUS_CLASS))
        ),
        "class1_tp_breaks": int(
            np.sum((target == FOCUS_CLASS) & (control == FOCUS_CLASS) & (role != FOCUS_CLASS))
        ),
        "restricted_fp_removals": int(removed.sum()),
        "restricted_fp_creations": int(created.sum()),
        "net_restricted_fp_removals": int(removed.sum() - created.sum()),
        "changed_predictions": int(np.sum(control != role)),
        "transition_matrix": confusion_matrix(
            control, role, labels=np.arange(5)
        ).astype(np.int64).tolist(),
    }
    return events


def _metric_deltas(candidate: Mapping[str, object], control: Mapping[str, object]) -> Dict[str, float]:
    return {
        "macro_f1": float(candidate["macro_f1"] - control["macro_f1"]),
        "class1_f1": float(
            candidate["per_class_f1"][FOCUS_CLASS]
            - control["per_class_f1"][FOCUS_CLASS]
        ),
        "class1_precision": float(
            candidate["per_class_precision"][FOCUS_CLASS]
            - control["per_class_precision"][FOCUS_CLASS]
        ),
        "class1_recall": float(
            candidate["per_class_recall"][FOCUS_CLASS]
            - control["per_class_recall"][FOCUS_CLASS]
        ),
        "accuracy": float(candidate["accuracy"] - control["accuracy"]),
        "nll": float(candidate["nll"] - control["nll"]),
        "brier": float(candidate["brier"] - control["brier"]),
        "ece": float(candidate["ece"] - control["ece"]),
    }


def _distribution(values: np.ndarray) -> Dict[str, object]:
    values = np.asarray(values, dtype=np.float64)
    if values.size == 0:
        return {"count": 0, "mean": 0.0, "median": 0.0, "q10": 0.0, "q90": 0.0}
    return {
        "count": int(values.size),
        "mean": float(values.mean()),
        "median": float(np.median(values)),
        "q10": float(np.quantile(values, 0.10)),
        "q90": float(np.quantile(values, 0.90)),
    }


def direction_summary(
    target: np.ndarray,
    control_probabilities: np.ndarray,
    role_probabilities: np.ndarray,
) -> Dict[str, object]:
    control_predictions = control_probabilities.argmax(axis=1)
    delta = role_probabilities[:, FOCUS_CLASS] - control_probabilities[:, FOCUS_CLASS]
    true_class1 = target == FOCUS_CLASS
    restricted_fp = (
        np.isin(target, RESTRICTED_NEGATIVE_CLASSES)
        & (control_predictions == FOCUS_CLASS)
    )
    selected = true_class1 | restricted_fp
    auc_target = true_class1[selected].astype(np.int64)
    auc = float(roc_auc_score(auc_target, delta[selected]))
    cohorts = {
        "true_class1_keeper_tp": true_class1 & (control_predictions == FOCUS_CLASS),
        "true_class1_keeper_fn": true_class1 & (control_predictions != FOCUS_CLASS),
        "restricted_keeper_fp": restricted_fp,
        "remaining_restricted_negatives": (
            np.isin(target, RESTRICTED_NEGATIVE_CLASSES)
            & (control_predictions != FOCUS_CLASS)
        ),
    }
    return {
        "direction_auc": auc,
        "cohorts": {name: _distribution(delta[mask]) for name, mask in cohorts.items()},
    }


def build_analysis(
    *,
    target: np.ndarray,
    fold: np.ndarray,
    probabilities: Mapping[str, Mapping[str, np.ndarray]],
) -> Dict[str, object]:
    conditions: Dict[str, object] = {}
    for condition, _brightness, _contrast in CONDITIONS:
        role_metrics = {
            role: classification_metrics(target, probabilities[condition][role])
            for role in ROLES
        }
        conditions[condition] = {
            "roles": role_metrics,
            "candidate_vs_control": {
                "deltas": _metric_deltas(
                    role_metrics["candidate"], role_metrics["control"]
                ),
                "events": transition_events(
                    target,
                    probabilities[condition]["control"],
                    probabilities[condition]["candidate"],
                ),
            },
            "placebo_vs_control": {
                "deltas": _metric_deltas(
                    role_metrics["placebo"], role_metrics["control"]
                ),
                "events": transition_events(
                    target,
                    probabilities[condition]["control"],
                    probabilities[condition]["placebo"],
                ),
            },
        }

    clean_probabilities = probabilities["clean"]
    clean_folds = []
    for fold_index in range(5):
        selected = fold == fold_index
        role_metrics = {
            role: classification_metrics(target[selected], values[selected])
            for role, values in clean_probabilities.items()
        }
        clean_folds.append(
            {
                "fold": int(fold_index),
                "rows": int(selected.sum()),
                "roles": role_metrics,
                "candidate_vs_control": _metric_deltas(
                    role_metrics["candidate"], role_metrics["control"]
                ),
                "placebo_vs_control": _metric_deltas(
                    role_metrics["placebo"], role_metrics["control"]
                ),
            }
        )
    direction = {
        role: direction_summary(
            target, clean_probabilities["control"], clean_probabilities[role]
        )
        for role in ("candidate", "placebo")
    }
    return {
        "conditions": conditions,
        "clean_folds": clean_folds,
        "direction": direction,
    }


def summarize_attention_telemetry(
    telemetry: Mapping[str, Mapping[str, np.ndarray]],
) -> Dict[str, object]:
    conditions: Dict[str, object] = {}
    maximum_post_mass = 0.0
    for condition, _brightness, _contrast in CONDITIONS:
        layers = []
        payload = telemetry[condition]
        for layer_index in range(1, ATTENTION_LAYERS + 1):
            invalid = np.asarray(payload[f"layer_{layer_index}_invalid_count"])
            pre = np.asarray(payload[f"layer_{layer_index}_pre_mass"])
            post = np.asarray(payload[f"layer_{layer_index}_post_mass"])
            maximum_post_mass = max(maximum_post_mass, float(post.max(initial=0.0)))
            layers.append(
                {
                    "layer": layer_index,
                    "invalid_count": {
                        "mean": float(invalid.mean()),
                        "median": float(np.median(invalid)),
                        "q10": float(np.quantile(invalid, 0.10)),
                        "q90": float(np.quantile(invalid, 0.90)),
                        "maximum": int(invalid.max()),
                    },
                    "pre_mask_class_attention_mass": _distribution(pre),
                    "post_mask_class_attention_mass": _distribution(post),
                    "post_mask_maximum": float(post.max(initial=0.0)),
                }
            )
        conditions[condition] = {"layers": layers}
    return {
        "conditions": conditions,
        "maximum_post_mask_invalid_attention_mass": float(maximum_post_mass),
    }


def assess_validity_aware_attention_a0(
    *,
    structural: Mapping[str, object],
    analysis: Mapping[str, object],
) -> Dict[str, object]:
    clean = analysis["conditions"]["clean"]
    delta = clean["candidate_vs_control"]["deltas"]
    events = clean["candidate_vs_control"]["events"]
    candidate_clean = clean["roles"]["candidate"]
    placebo_clean = clean["roles"]["placebo"]
    candidate_placebo = _metric_deltas(candidate_clean, placebo_clean)
    placebo_events = clean["placebo_vs_control"]["events"]
    folds = analysis["clean_folds"]
    direction = analysis["direction"]
    shifted = [
        analysis["conditions"][name]["candidate_vs_control"]
        for name in ("lighting_dim", "lighting_bright", "low_contrast")
    ]

    structural_checks = {
        "provenance_exact": bool(structural["provenance_exact"]),
        "ordered_rows_complete": bool(structural["ordered_rows_complete"]),
        "class_fold_counts_exact": bool(structural["class_fold_counts_exact"]),
        "derangement_exact": bool(structural["derangement_exact"]),
        "geometry_census_exact": bool(structural["geometry_census_exact"]),
        "no_all_masked_fallback": bool(structural["no_all_masked_fallback"]),
        "standard_attention_contract": bool(structural["standard_attention_contract"]),
        "control_replay_fidelity": float(
            structural["control_probability_max_abs_difference"]
        )
        <= 2e-6,
        "all_valid_logit_fidelity": float(
            structural["all_valid_logit_max_abs_difference"]
        )
        <= 2e-6,
        "all_valid_probability_fidelity": float(
            structural["all_valid_probability_max_abs_difference"]
        )
        <= 2e-6,
        "candidate_differs_from_keeper": bool(
            structural["candidate_differs_from_keeper"]
        ),
        "post_mask_attention_zero": float(
            structural["maximum_post_mask_invalid_attention_mass"]
        )
        <= 1e-7,
        "model_state_unchanged": bool(structural["model_state_unchanged"]),
        "raw_dataset_unchanged": bool(structural["raw_dataset_unchanged"]),
        "validation_data_unused": bool(structural["validation_data_unused"]),
        "test_data_unused": bool(structural["test_data_unused"]),
        "model_checkpoint_writes_false": bool(
            structural["model_checkpoint_writes_false"]
        ),
        "replay_verified": bool(structural["replay_verified"]),
    }
    clean_checks = {
        "macro_f1_delta_ge_neg002": delta["macro_f1"] >= -0.002,
        "class1_f1_delta_ge_005": delta["class1_f1"] >= 0.005,
        "class1_precision_delta_ge_010": delta["class1_precision"] >= 0.010,
        "class1_recall_delta_ge_neg010": delta["class1_recall"] >= -0.010,
        "net_restricted_fp_removals_ge_10": events[
            "net_restricted_fp_removals"
        ]
        >= 10,
        "corrections_ge_harms": events["corrections"] >= events["harms"],
        "class1_tp_break_safety": events["class1_tp_breaks"]
        <= min(5, events["class1_fn_rescues"] + 3),
        "precision_nonworse_folds_ge_4": sum(
            item["candidate_vs_control"]["class1_precision"] >= 0.0
            for item in folds
        )
        >= 4,
        "f1_nonworse_folds_ge_3": sum(
            item["candidate_vs_control"]["class1_f1"] >= 0.0 for item in folds
        )
        >= 3,
        "worst_fold_class1_f1_delta_ge_neg010": min(
            item["candidate_vs_control"]["class1_f1"] for item in folds
        )
        >= -0.010,
        "direction_auc_ge_060": direction["candidate"]["direction_auc"] >= 0.60,
    }
    alignment_checks = {
        "macro_f1_vs_placebo_ge_002": candidate_placebo["macro_f1"] >= 0.002,
        "class1_f1_vs_placebo_ge_005": candidate_placebo["class1_f1"] >= 0.005,
        "class1_precision_vs_placebo_ge_008": candidate_placebo[
            "class1_precision"
        ]
        >= 0.008,
        "direction_auc_advantage_ge_003": direction["candidate"]["direction_auc"]
        - direction["placebo"]["direction_auc"]
        >= 0.03,
        "net_fp_removal_advantage_ge_5": events["net_restricted_fp_removals"]
        - placebo_events["net_restricted_fp_removals"]
        >= 5,
        "tp_breaks_no_more_than_placebo": events["class1_tp_breaks"]
        <= placebo_events["class1_tp_breaks"],
    }
    shifted_precision = [item["deltas"]["class1_precision"] for item in shifted]
    robustness_checks = {
        "shifted_precision_nonnegative_all": min(shifted_precision) >= 0.0,
        "shifted_precision_ge_005_two": sum(value >= 0.005 for value in shifted_precision)
        >= 2,
        "worst_shifted_class1_f1_delta_ge_neg010": min(
            item["deltas"]["class1_f1"] for item in shifted
        )
        >= -0.010,
        "worst_shifted_class1_recall_delta_ge_neg020": min(
            item["deltas"]["class1_recall"] for item in shifted
        )
        >= -0.020,
        "worst_shifted_macro_f1_delta_ge_neg005": min(
            item["deltas"]["macro_f1"] for item in shifted
        )
        >= -0.005,
        "aggregate_shifted_net_fp_removals_ge_15": sum(
            item["events"]["net_restricted_fp_removals"] for item in shifted
        )
        >= 15,
        "aggregate_shifted_tp_breaks_le_rescues": sum(
            item["events"]["class1_tp_breaks"] for item in shifted
        )
        <= sum(item["events"]["class1_fn_rescues"] for item in shifted),
    }
    groups = {
        "structural": structural_checks,
        "clean": clean_checks,
        "alignment": alignment_checks,
        "robustness": robustness_checks,
    }
    failed = [
        f"{group}.{name}"
        for group, checks in groups.items()
        for name, passed in checks.items()
        if not bool(passed)
    ]
    return {
        "groups": groups,
        "failed": failed,
        "passed_count": int(
            sum(bool(value) for checks in groups.values() for value in checks.values())
        ),
        "total_count": int(sum(len(checks) for checks in groups.values())),
        "all_passed": not failed,
        "status": (
            "passed_for_model_integration_review"
            if not failed
            else "rejected_before_model_integration"
        ),
    }


def _recursive_max_difference(left, right) -> float:
    if isinstance(left, Mapping) and isinstance(right, Mapping):
        if set(left) != set(right):
            return math.inf
        return max(
            (_recursive_max_difference(left[key], right[key]) for key in left),
            default=0.0,
        )
    if isinstance(left, list) and isinstance(right, list):
        if len(left) != len(right):
            return math.inf
        return max(
            (_recursive_max_difference(a, b) for a, b in zip(left, right)),
            default=0.0,
        )
    if isinstance(left, (bool, str)) or isinstance(right, (bool, str)):
        return 0.0 if left == right else math.inf
    if left is None or right is None:
        return 0.0 if left is right else math.inf
    try:
        return abs(float(left) - float(right))
    except (TypeError, ValueError):
        return 0.0 if left == right else math.inf


def _render_mechanism_sheet(
    output_path: Path,
    visuals: Mapping[int, Mapping[str, np.ndarray]],
) -> None:
    if set(visuals) != set(FIXED_VISUAL_INDICES):
        raise ValueError("Fixed mechanism visual cohort is incomplete")
    figure, axes = plt.subplots(5, 5, figsize=(15, 15), constrained_layout=True)
    titles = (
        "Input",
        "Validity overlay",
        "Block-1 pre-mask",
        "Block-1 post-mask",
        "Absolute change",
    )
    for column, title in enumerate(titles):
        axes[0, column].set_title(title)
    for row, sample_index in enumerate(FIXED_VISUAL_INDICES):
        payload = visuals[int(sample_index)]
        image = payload["image"]
        valid = payload["valid_mask"].astype(bool)
        pre = payload["pre"]
        post = payload["post"]
        overlay = image.astype(np.float32).copy()
        overlay[~valid] = 0.45 * overlay[~valid] + 0.55 * np.asarray(
            [255.0, 40.0, 40.0]
        )
        panels = (image, overlay.astype(np.uint8), pre, post, np.abs(post - pre))
        for column, panel in enumerate(panels):
            if column < 2:
                axes[row, column].imshow(panel)
            else:
                axes[row, column].imshow(panel, cmap="magma", interpolation="nearest")
            axes[row, column].axis("off")
        axes[row, 0].set_ylabel(
            f"class {int(payload['target'])}\nindex {sample_index}", fontsize=10
        )
    figure.savefig(output_path, dpi=150)
    plt.close(figure)


def _write_predictions_csv(
    path: Path,
    *,
    target: np.ndarray,
    fold: np.ndarray,
    source: np.ndarray,
    probabilities: Mapping[str, Mapping[str, np.ndarray]],
    mask_cache: np.ndarray,
    derangement: np.ndarray,
) -> None:
    fields = [
        "condition",
        "sample_index",
        "target",
        "fold",
        "source",
        "donor_index",
        "donor_source",
        "fully_invalid_patches",
    ]
    for role in ROLES:
        fields.extend([f"{role}_pred", *[f"{role}_p{index}" for index in range(5)]])
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields)
        writer.writeheader()
        for condition, _brightness, _contrast in CONDITIONS:
            role_predictions = {
                role: values.argmax(axis=1)
                for role, values in probabilities[condition].items()
            }
            for index in range(EXPECTED_TRAIN_ROWS):
                donor = int(derangement[index])
                row = {
                    "condition": condition,
                    "sample_index": index,
                    "target": int(target[index]),
                    "fold": int(fold[index]),
                    "source": str(source[index]),
                    "donor_index": donor,
                    "donor_source": str(source[donor]),
                    "fully_invalid_patches": int(mask_cache[index].sum()),
                }
                for role in ROLES:
                    row[f"{role}_pred"] = int(role_predictions[role][index])
                    for class_index in range(5):
                        row[f"{role}_p{class_index}"] = format(
                            float(probabilities[condition][role][index, class_index]),
                            ".10g",
                        )
                writer.writerow(row)


def _json_dump(path: Path, payload: Mapping[str, object]) -> None:
    path.write_text(json.dumps(payload, indent=2, allow_nan=False), encoding="utf-8")


def _artifact_hashes(output_dir: Path, names: Sequence[str]) -> Dict[str, object]:
    result: Dict[str, object] = {}
    for name in names:
        path = output_dir / name
        result[name] = {
            "sha256": _sha256(path),
            "bytes": int(path.stat().st_size),
        }
    return result


def _load_persisted_arrays(output_dir: Path):
    probability_cache = np.load(output_dir / "probabilities.npz", allow_pickle=False)
    telemetry_cache = np.load(output_dir / "attention_telemetry.npz", allow_pickle=False)
    target = probability_cache["target"].astype(np.int64)
    fold = probability_cache["fold"].astype(np.int64)
    probabilities = {
        condition: {
            role: probability_cache[f"{condition}__{role}"] for role in ROLES
        }
        for condition, _brightness, _contrast in CONDITIONS
    }
    telemetry = {
        condition: {
            f"layer_{layer}_{name}": telemetry_cache[
                f"{condition}__layer_{layer}_{name}"
            ]
            for layer in range(1, ATTENTION_LAYERS + 1)
            for name in ("invalid_count", "pre_mass", "post_mass")
        }
        for condition, _brightness, _contrast in CONDITIONS
    }
    return probability_cache, telemetry_cache, target, fold, probabilities, telemetry


def replay_output(summary_path: Path) -> Dict[str, object]:
    summary_path = Path(summary_path).resolve()
    summary = json.loads(summary_path.read_text(encoding="utf-8"))
    output_dir = summary_path.parent
    (
        probability_cache,
        telemetry_cache,
        target,
        fold,
        probabilities,
        telemetry,
    ) = _load_persisted_arrays(output_dir)
    try:
        analysis = build_analysis(target=target, fold=fold, probabilities=probabilities)
        telemetry_summary = summarize_attention_telemetry(telemetry)
    finally:
        probability_cache.close()
        telemetry_cache.close()
    analysis_difference = _recursive_max_difference(summary["analysis"], analysis)
    telemetry_difference = _recursive_max_difference(
        summary["attention_telemetry"], telemetry_summary
    )
    structural = dict(summary["structural"])
    structural["replay_verified"] = True
    gate = assess_validity_aware_attention_a0(
        structural=structural, analysis=analysis
    )
    gate_difference = _recursive_max_difference(summary["gate"], gate)
    result = {
        "analysis_max_abs_difference": float(analysis_difference),
        "telemetry_max_abs_difference": float(telemetry_difference),
        "gate_max_abs_difference": float(gate_difference),
        "probability_roundtrip_max_abs_difference": 0.0,
        "all_exact": bool(
            analysis_difference <= 1e-12
            and telemetry_difference <= 1e-12
            and gate_difference <= 1e-12
        ),
    }
    if not result["all_exact"]:
        raise ValueError(f"Validity-aware attention replay failed: {result}")
    return result


def run_audit(args: argparse.Namespace) -> Dict[str, object]:
    _validate_locked_args(args)
    provenance = verify_provenance(args)
    torch.set_num_threads(int(args.torch_threads))
    set_seed(SEED, deterministic=True)
    device = _resolve_device(args.device)
    if device.type == "cuda":
        torch.backends.cuda.matmul.allow_tf32 = False
        torch.backends.cudnn.allow_tf32 = True
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False

    paths = _default_paths()
    condition_rows = _read_cidt_conditions(Path(args.cidt_predictions))
    checkpoint = torch.load(Path(args.checkpoint), map_location="cpu", weights_only=False)
    if not isinstance(checkpoint, Mapping):
        raise ValueError("Invalid keeper checkpoint")
    model = build_model_from_checkpoint(dict(checkpoint)).to(device).eval()
    controller = ValidityAwareAttentionController(model)
    base_dataset, transform, class_names = _build_dataset(
        data_path=Path(args.data),
        checkpoint=checkpoint,
        condition_rows=condition_rows,
    )
    fidelity = _first_batch_fidelity(
        model=model,
        controller=controller,
        base_dataset=base_dataset,
        transform=transform,
        expected_clean=condition_rows["clean"],
        device=device,
        batch_size=int(args.batch_size),
        num_workers=int(args.num_workers),
    )
    if args.preflight_only:
        return {
            "mode": "preflight",
            "provenance": provenance,
            "class_names": class_names,
            "fidelity": fidelity,
            "attention_layers": len(controller.modules),
        }

    output_dir = Path(args.output_dir).resolve()
    if output_dir.exists() and any(output_dir.iterdir()):
        raise FileExistsError(f"Formal output already exists: {output_dir}")
    output_dir.mkdir(parents=True, exist_ok=True)
    started = time.perf_counter()
    raw_before = _dataset_tree_identity(Path(args.data))
    state_before = _model_state_sha256(model)
    clean = condition_rows["clean"]
    target = clean["target"].astype(np.int64)
    fold = clean["fold"].astype(np.int64)
    source = clean["source"].astype(object)
    derangement = build_same_class_fold_source_derangement(target, fold, source)
    mask_cache = np.zeros((EXPECTED_TRAIN_ROWS, PATCH_COUNT), dtype=bool)
    pixel_padded_cache = np.zeros(EXPECTED_TRAIN_ROWS, dtype=bool)
    visual_payload: Dict[int, Dict[str, np.ndarray]] = {}
    mean, std = checkpoint_input_normalization(checkpoint)

    probabilities: Dict[str, Dict[str, np.ndarray]] = {}
    telemetry: Dict[str, Dict[str, np.ndarray]] = {}
    runtime_records: Dict[str, object] = {}
    for condition, brightness, contrast in CONDITIONS:
        probabilities[condition] = {
            "control": condition_rows[condition]["probabilities"].astype(
                np.float64, copy=True
            )
        }
        candidate = _infer_role_condition(
            role="candidate",
            condition=condition,
            brightness=brightness,
            contrast=contrast,
            model=model,
            controller=controller,
            base_dataset=base_dataset,
            transform=transform,
            expected=condition_rows[condition],
            device=device,
            batch_size=int(args.batch_size),
            num_workers=int(args.num_workers),
            mask_cache=mask_cache,
            pixel_padded_cache=pixel_padded_cache,
            derangement=derangement,
            visual_payload=visual_payload,
            mean=mean,
            std=std,
        )
        placebo = _infer_role_condition(
            role="placebo",
            condition=condition,
            brightness=brightness,
            contrast=contrast,
            model=model,
            controller=controller,
            base_dataset=base_dataset,
            transform=transform,
            expected=condition_rows[condition],
            device=device,
            batch_size=int(args.batch_size),
            num_workers=int(args.num_workers),
            mask_cache=mask_cache,
            pixel_padded_cache=pixel_padded_cache,
            derangement=derangement,
            visual_payload=visual_payload,
            mean=mean,
            std=std,
        )
        probabilities[condition]["candidate"] = candidate["probabilities"]
        probabilities[condition]["placebo"] = placebo["probabilities"]
        telemetry[condition] = candidate["telemetry"]
        runtime_records[condition] = {
            "candidate_seconds": candidate["seconds"],
            "placebo_seconds": placebo["seconds"],
            "candidate_loader": candidate["loader"],
            "placebo_loader": placebo["loader"],
        }

    state_after = _model_state_sha256(model)
    raw_after = _dataset_tree_identity(Path(args.data))
    model.to("cpu")
    del model
    if device.type == "cuda":
        torch.cuda.empty_cache()

    analysis = build_analysis(target=target, fold=fold, probabilities=probabilities)
    telemetry_summary = summarize_attention_telemetry(telemetry)
    class_invalid = tuple(
        int(mask_cache[target == class_index].sum()) for class_index in range(5)
    )
    geometry = {
        "padded_rows": int(pixel_padded_cache.sum()),
        "square_rows": int((~pixel_padded_cache).sum()),
        "fully_invalid_patch_total": int(mask_cache.sum()),
        "fully_invalid_patch_mean": float(mask_cache.sum(axis=1).mean()),
        "class_fully_invalid_patch_totals": list(class_invalid),
        "all_masked_rows": int(mask_cache.all(axis=1).sum()),
    }
    maximum_candidate_difference = max(
        float(
            np.max(
                np.abs(
                    probabilities[condition]["candidate"].astype(np.float64)
                    - probabilities[condition]["control"]
                )
            )
        )
        for condition, _brightness, _contrast in CONDITIONS
    )
    structural = {
        "provenance_exact": bool(provenance["all_exact"]),
        "ordered_rows_complete": all(
            np.array_equal(
                condition_rows[condition]["sample_index"],
                np.arange(EXPECTED_TRAIN_ROWS),
            )
            for condition, _brightness, _contrast in CONDITIONS
        ),
        "class_fold_counts_exact": bool(
            np.bincount(target, minlength=5).tolist() == list(EXPECTED_CLASS_COUNTS)
            and np.bincount(fold, minlength=5).tolist() == list(EXPECTED_FOLD_COUNTS)
        ),
        "derangement_exact": bool(
            np.unique(derangement).size == EXPECTED_TRAIN_ROWS
            and np.array_equal(target[derangement], target)
            and np.array_equal(fold[derangement], fold)
            and np.all(source[derangement] != source)
        ),
        "geometry_census_exact": bool(
            geometry["padded_rows"] == EXPECTED_PADDED_ROWS
            and geometry["fully_invalid_patch_total"]
            == EXPECTED_FULLY_INVALID_PATCHES
            and class_invalid == EXPECTED_CLASS_INVALID_PATCHES
        ),
        "no_all_masked_fallback": geometry["all_masked_rows"] == 0,
        "standard_attention_contract": len(controller.modules) == ATTENTION_LAYERS,
        **fidelity,
        "candidate_differs_from_keeper": maximum_candidate_difference > 1e-6,
        "candidate_keeper_max_abs_difference": maximum_candidate_difference,
        "maximum_post_mask_invalid_attention_mass": telemetry_summary[
            "maximum_post_mask_invalid_attention_mass"
        ],
        "model_state_before": state_before,
        "model_state_after": state_after,
        "model_state_unchanged": state_before == state_after,
        "raw_dataset_before": raw_before,
        "raw_dataset_after": raw_after,
        "raw_dataset_unchanged": raw_before == raw_after,
        "validation_data_unused": True,
        "test_data_unused": True,
        "model_checkpoint_writes_false": True,
        "replay_verified": False,
    }

    probability_payload = {
        "sample_index": np.arange(EXPECTED_TRAIN_ROWS, dtype=np.int64),
        "target": target,
        "fold": fold,
        "derangement": derangement,
        "patch_mask": mask_cache,
        "pixel_padded": pixel_padded_cache,
        **{
            f"{condition}__{role}": probabilities[condition][role]
            for condition, _brightness, _contrast in CONDITIONS
            for role in ROLES
        },
    }
    np.savez_compressed(output_dir / "probabilities.npz", **probability_payload)
    np.savez_compressed(
        output_dir / "attention_telemetry.npz",
        **{
            f"{condition}__{name}": values
            for condition, payload in telemetry.items()
            for name, values in payload.items()
        },
    )
    _write_predictions_csv(
        output_dir / "predictions_all_conditions.csv",
        target=target,
        fold=fold,
        source=source,
        probabilities=probabilities,
        mask_cache=mask_cache,
        derangement=derangement,
    )
    _render_mechanism_sheet(
        output_dir / "fixed_validity_attention_mechanism_sheet.png",
        visual_payload,
    )

    (
        probability_cache,
        telemetry_cache,
        replay_target,
        replay_fold,
        replay_probabilities,
        replay_telemetry,
    ) = _load_persisted_arrays(output_dir)
    try:
        replay_analysis = build_analysis(
            target=replay_target,
            fold=replay_fold,
            probabilities=replay_probabilities,
        )
        replay_telemetry_summary = summarize_attention_telemetry(replay_telemetry)
    finally:
        probability_cache.close()
        telemetry_cache.close()
    replay_record = {
        "probability_roundtrip_max_abs_difference": 0.0,
        "analysis_max_abs_difference": float(
            _recursive_max_difference(analysis, replay_analysis)
        ),
        "telemetry_max_abs_difference": float(
            _recursive_max_difference(telemetry_summary, replay_telemetry_summary)
        ),
    }
    replay_record["all_exact"] = bool(
        replay_record["analysis_max_abs_difference"] <= 1e-12
        and replay_record["telemetry_max_abs_difference"] <= 1e-12
    )
    structural["replay_verified"] = replay_record["all_exact"]
    gate = assess_validity_aware_attention_a0(
        structural=structural, analysis=analysis
    )
    summary = {
        "mode": "validity_aware_attention_a0",
        "status": gate["status"],
        "test_data_used": False,
        "validation_predictions_used": False,
        "split": "train",
        "rows": EXPECTED_TRAIN_ROWS,
        "class_names": class_names,
        "provenance": provenance,
        "runtime": {
            "device": str(device),
            "gpu_name": (
                torch.cuda.get_device_name(device) if device.type == "cuda" else None
            ),
            "batch_size": int(args.batch_size),
            "requested_num_workers": int(args.num_workers),
            "torch_threads": int(args.torch_threads),
            "condition_records": runtime_records,
            "elapsed_seconds": float(time.perf_counter() - started),
        },
        "geometry": geometry,
        "structural": structural,
        "analysis": analysis,
        "attention_telemetry": telemetry_summary,
        "replay": replay_record,
        "gate": gate,
        "decision": (
            "Manual mechanism review and one model integration are authorized."
            if gate["all_passed"]
            else "Reject before model integration, validation, test, smoke, probe, or full train."
        ),
        "current_best_commands_updated": False,
    }
    _json_dump(output_dir / "summary.json", summary)
    external_replay = replay_output(output_dir / "summary.json")
    _json_dump(output_dir / "replay_summary.json", external_replay)
    artifact_names = (
        "probabilities.npz",
        "attention_telemetry.npz",
        "predictions_all_conditions.csv",
        "fixed_validity_attention_mechanism_sheet.png",
        "summary.json",
        "replay_summary.json",
    )
    manifest = {
        "mode": "validity_aware_attention_a0_manifest",
        "status": gate["status"],
        "artifacts": _artifact_hashes(output_dir, artifact_names),
        "raw_dataset_unchanged": structural["raw_dataset_unchanged"],
        "model_state_unchanged": structural["model_state_unchanged"],
        "validation_data_used": False,
        "test_data_used": False,
        "model_checkpoint_written": False,
    }
    _json_dump(output_dir / "manifest.json", manifest)
    return summary


def main(argv: Optional[Sequence[str]] = None) -> int:
    args = parse_args(argv)
    if args.replay_summary is not None:
        result = replay_output(args.replay_summary)
    else:
        result = run_audit(args)
    print(json.dumps(result, indent=2, allow_nan=False))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
