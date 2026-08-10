from __future__ import annotations

import argparse
import csv
import hashlib
import json
import math
from pathlib import Path
from typing import Callable, Dict, Iterable, List, Mapping, Optional, Sequence, Tuple

import numpy as np
import torch
import torch.nn.functional as F
from sklearn.model_selection import StratifiedGroupKFold
from torch import Tensor, nn
from torch.utils.data import DataLoader, Dataset
from tqdm import tqdm

from trkh.core.config import load_data_spec
from trkh.core.utils import (
    build_safe_dataloader_kwargs,
    ensure_dir,
    json_dump,
    load_checkpoint,
    set_seed,
)
from trkh.evaluation.input_normalization import checkpoint_input_normalization
from trkh.evaluation.metrics import build_metrics
from trkh.models.model import build_model_from_checkpoint
from trkh.training.friendly_adversarial import (
    FriendlyAttackResult,
    build_eroded_bbox_mask as _build_eroded_bbox_mask,
    class1_attack_margin as _class1_attack_margin,
    generate_friendly_adversarial_examples as _generate_friendly_adversarial_examples,
)
from trkh.tools.evaluate_paired_view_fusion import (
    _build_classification_dataset,
    _build_eval_transform_from_checkpoint,
    _forward_logits,
    _select_bbox_token_prior,
)
from trkh.tools.probe_api_pairwise_interaction_readiness import _write_artifact_manifest
from trkh.tools.probe_embedding_prototypes import _collate_classification


SEED = 20260712
FOCUS_CLASS = 1
NEGATIVE_CLASSES = (0, 2, 4)
ATTACK_EPSILON = 2.0 / 255.0
ATTACK_STEP_SIZE = 1.0 / 255.0
ATTACK_STEPS = 2
BBOX_ERODE_RATIO = 0.10
GRADIENT_SAMPLE_CAP = 32
GRADIENT_BATCH_SIZE = 8
FOLDS = 5
EXPECTED_TRAIN_COUNT = 9215
EXPECTED_VAL_COUNT = 2606
EXPECTED_KEEPER_VAL_MACRO = 0.882925
EXPECTED_KEEPER_VAL_CLASS1 = 0.678261
EXPECTED_CHECKPOINT_SHA256 = (
    "1f49d577240c69dc63c30af70db52ec2aa9da65a17aef1c4b1c09ece6c482677"
)

FAT_URL = "https://proceedings.mlr.press/v119/zhang20z.html"
ADVPROP_URL = (
    "https://openaccess.thecvf.com/content_CVPR_2020/html/"
    "Xie_Adversarial_Examples_Improve_Image_Recognition_CVPR_2020_paper.html"
)
LBGAT_URL = (
    "https://openaccess.thecvf.com/content/ICCV2021/html/"
    "Cui_Learnable_Boundary_Guided_Adversarial_Training_ICCV_2021_paper.html"
)


class IndexedSubsetDataset(Dataset):
    def __init__(self, dataset: Dataset, indices: Sequence[int]) -> None:
        self.dataset = dataset
        self.indices = [int(index) for index in indices]
        sample_paths = getattr(dataset, "sample_paths", None)
        if not callable(sample_paths):
            raise TypeError("classification dataset must expose sample_paths()")
        self.paths = [str(path) for path in sample_paths()]
        if len(self.paths) != len(dataset):
            raise ValueError("classification sample_paths coverage is incomplete")

    def __len__(self) -> int:
        return len(self.indices)

    def __getitem__(self, index: int):
        sample_index = self.indices[int(index)]
        item = self.dataset[sample_index]
        if len(item) == 2:
            image, label = item
            metadata: Dict[str, object] = {}
        elif len(item) == 3:
            image, label, raw_metadata = item
            metadata = dict(raw_metadata) if isinstance(raw_metadata, Mapping) else {}
        else:
            raise ValueError("classification item must have two or three fields")
        metadata["audit_index"] = torch.tensor(sample_index, dtype=torch.long)
        metadata["image_path"] = self.paths[sample_index]
        return image, label, metadata


def parse_args(argv: Optional[Sequence[str]] = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Locked no-test readiness audit for class-1 friendly foreground "
            "adversarial supervision. It never writes a model or modifies data."
        )
    )
    parser.add_argument("--checkpoint", type=Path, required=True)
    parser.add_argument("--data", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--batch-size", type=int, default=32)
    parser.add_argument("--attack-batch-size", type=int, default=8)
    parser.add_argument("--num-workers", type=int, default=4)
    parser.add_argument("--device", type=str, default="cuda")
    parser.add_argument("--seed", type=int, default=SEED)
    parser.add_argument("--max-train-samples", type=int, default=0)
    parser.add_argument("--max-val-samples", type=int, default=0)
    parser.add_argument(
        "--allow-preflight",
        action="store_true",
        default=False,
        help="Allow capped split prefixes for implementation checks; gate remains closed.",
    )
    return parser.parse_args(argv)


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with Path(path).open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _is_relative_to(path: Path, parent: Path) -> bool:
    try:
        path.relative_to(parent)
        return True
    except ValueError:
        return False


def _resolve_device(requested: str) -> torch.device:
    normalized = str(requested or "").strip().lower()
    if normalized.startswith("cuda") and not torch.cuda.is_available():
        raise RuntimeError("CUDA was requested but is unavailable")
    return torch.device(normalized or ("cuda" if torch.cuda.is_available() else "cpu"))


def build_eroded_bbox_mask(
    crop_bboxes: Tensor,
    image_valid_mask: Tensor,
    *,
    height: int,
    width: int,
    erode_ratio: float = BBOX_ERODE_RATIO,
) -> Tensor:
    return _build_eroded_bbox_mask(
        crop_bboxes,
        image_valid_mask,
        height=height,
        width=width,
        erode_ratio=erode_ratio,
    )


def class1_attack_margin(
    logits: Tensor,
    labels: Tensor,
    *,
    direction: str,
    focus_class: int = FOCUS_CLASS,
    negative_classes: Sequence[int] = NEGATIVE_CLASSES,
) -> Tensor:
    return _class1_attack_margin(
        logits,
        labels,
        direction=direction,
        focus_class=focus_class,
        negative_classes=negative_classes,
    )


def generate_friendly_adversarial_examples(
    *,
    images: Tensor,
    labels: Tensor,
    attack_mask: Tensor,
    forward_logits: Callable[[Tensor], Tensor],
    mean: Sequence[float],
    std: Sequence[float],
    direction: str,
    epsilon: float = ATTACK_EPSILON,
    step_size: float = ATTACK_STEP_SIZE,
    steps: int = ATTACK_STEPS,
) -> FriendlyAttackResult:
    return _generate_friendly_adversarial_examples(
        images=images,
        labels=labels,
        attack_mask=attack_mask,
        forward_logits=forward_logits,
        mean=mean,
        std=std,
        direction=direction,
        focus_class=FOCUS_CLASS,
        negative_classes=NEGATIVE_CLASSES,
        epsilon=epsilon,
        step_size=step_size,
        steps=steps,
    )


def classify_row_cohort(row: Mapping[str, object]) -> str:
    target = int(row["target_index"])
    prediction = int(row["prediction_index"])
    top2 = {int(row["top1_index"]), int(row["top2_index"])}
    if target == FOCUS_CLASS:
        return "protect_correct" if prediction == FOCUS_CLASS else "positive_error"
    if target in NEGATIVE_CLASSES and prediction == FOCUS_CLASS:
        return "negative_error"
    if target in NEGATIVE_CLASSES and prediction == target and FOCUS_CLASS in top2:
        return "suppress_correct"
    return "other"


def boundary_margin_from_row(row: Mapping[str, object]) -> float:
    probabilities = [float(row[f"prob_{index}"]) for index in range(5)]
    target = int(row["target_index"])
    if target == FOCUS_CLASS:
        rival = max(probabilities[index] for index in NEGATIVE_CLASSES)
        return float(probabilities[FOCUS_CLASS] - rival)
    return float(probabilities[target] - probabilities[FOCUS_CLASS])


def assign_source_grouped_folds(
    rows: Sequence[Mapping[str, object]],
    *,
    folds: int = FOLDS,
    seed: int = SEED,
) -> Dict[int, int]:
    if int(folds) < 2:
        raise ValueError("folds must be at least two")
    sample_indices = np.asarray([int(row["sample_index"]) for row in rows], dtype=np.int64)
    if len(set(sample_indices.tolist())) != len(sample_indices):
        raise ValueError("sample indices must be unique")
    groups = np.asarray([str(row["source_stem"]) for row in rows], dtype=object)
    if any(not str(group) for group in groups.tolist()):
        raise ValueError("source groups must be non-empty")
    cohort_to_label = {
        "protect_correct": 0,
        "positive_error": 1,
        "suppress_correct": 2,
        "negative_error": 3,
        "other": 4,
    }
    strata = np.asarray(
        [cohort_to_label[classify_row_cohort(row)] for row in rows],
        dtype=np.int64,
    )
    splitter = StratifiedGroupKFold(
        n_splits=int(folds),
        shuffle=True,
        random_state=int(seed),
    )
    assignments: Dict[int, int] = {}
    for fold_index, (_, held_indices) in enumerate(
        splitter.split(np.zeros(len(rows), dtype=np.float32), strata, groups)
    ):
        for row_index in held_indices.tolist():
            assignments[int(sample_indices[row_index])] = int(fold_index)
    if len(assignments) != len(rows):
        raise RuntimeError("source-grouped fold assignment is incomplete")
    source_folds: Dict[str, set[int]] = {}
    for row in rows:
        source_folds.setdefault(str(row["source_stem"]), set()).add(
            assignments[int(row["sample_index"])]
        )
    if any(len(values) != 1 for values in source_folds.values()):
        raise RuntimeError("a source group crosses readiness folds")
    return assignments


def select_cohort_indices(
    rows: Sequence[Mapping[str, object]],
    *,
    cohort: str,
    cap: int = GRADIENT_SAMPLE_CAP,
    fold_assignments: Optional[Mapping[int, int]] = None,
    include_fold: Optional[int] = None,
    exclude_fold: Optional[int] = None,
) -> List[int]:
    if include_fold is not None and exclude_fold is not None:
        raise ValueError("include_fold and exclude_fold are mutually exclusive")
    selected: List[Tuple[float, int]] = []
    for row in rows:
        if classify_row_cohort(row) != str(cohort):
            continue
        sample_index = int(row["sample_index"])
        if fold_assignments is not None:
            fold = int(fold_assignments[sample_index])
            if include_fold is not None and fold != int(include_fold):
                continue
            if exclude_fold is not None and fold == int(exclude_fold):
                continue
        selected.append((abs(boundary_margin_from_row(row)), sample_index))
    selected.sort(key=lambda item: (item[0], item[1]))
    if int(cap) > 0:
        selected = selected[: int(cap)]
    return [sample_index for _, sample_index in selected]


def _metadata_tensors(
    metadata: Mapping[str, object],
    *,
    device: torch.device,
) -> Tuple[Tensor, Tensor, Tensor, Tensor]:
    required = ("bbox", "crop_bbox", "image_mask", "audit_index")
    if any(not torch.is_tensor(metadata.get(key)) for key in required):
        raise ValueError("bbox/crop_bbox/image_mask/audit_index metadata is required")
    bbox = metadata["bbox"].to(device=device, dtype=torch.float32, non_blocking=True)
    crop_bbox = metadata["crop_bbox"].to(
        device=device, dtype=torch.float32, non_blocking=True
    )
    image_valid_mask = metadata["image_mask"].to(
        device=device, dtype=torch.bool, non_blocking=True
    )
    if image_valid_mask.ndim == 4 and int(image_valid_mask.size(1)) == 1:
        image_valid_mask = image_valid_mask[:, 0]
    audit_indices = metadata["audit_index"].to(device=device, dtype=torch.long).view(-1)
    return bbox, crop_bbox, image_valid_mask, audit_indices


def _model_logits(
    *,
    model: nn.Module,
    images: Tensor,
    image_valid_mask: Tensor,
    bbox: Tensor,
    crop_bbox: Tensor,
    bbox_token_prior_source: str,
    device: torch.device,
) -> Tensor:
    bbox_token_prior = _select_bbox_token_prior(
        bbox=bbox,
        crop_bbox=crop_bbox,
        source=bbox_token_prior_source,
    )
    logits, _ = _forward_logits(
        model=model,
        images=images,
        image_valid_mask=image_valid_mask,
        bbox_metadata=bbox,
        bbox_token_prior=bbox_token_prior,
        amp=False,
        device=device,
    )
    return logits.float()


def _loader_for_indices(
    *,
    dataset: Dataset,
    indices: Sequence[int],
    batch_size: int,
    num_workers: int,
    device: torch.device,
    context: str,
) -> Tuple[DataLoader, Dict[str, object]]:
    loader_kwargs, loader_summary = build_safe_dataloader_kwargs(
        requested_num_workers=int(num_workers),
        requested_pin_memory=device.type == "cuda",
        context=context,
        prefetch_factor=2,
        persistent_workers=True,
    )
    loader = DataLoader(
        IndexedSubsetDataset(dataset, indices),
        batch_size=int(batch_size),
        shuffle=False,
        collate_fn=_collate_classification,
        **loader_kwargs,
    )
    return loader, loader_summary


def collect_clean_split(
    *,
    split: str,
    dataset: Dataset,
    indices: Sequence[int],
    model: nn.Module,
    class_names: Sequence[str],
    bbox_token_prior_source: str,
    device: torch.device,
    batch_size: int,
    num_workers: int,
) -> Dict[str, object]:
    loader, loader_summary = _loader_for_indices(
        dataset=dataset,
        indices=indices,
        batch_size=int(batch_size),
        num_workers=int(num_workers),
        device=device,
        context=f"friendly_adversarial_clean_{split}",
    )
    rows: List[Dict[str, object]] = []
    all_targets: List[Tensor] = []
    all_probabilities: List[Tensor] = []
    with torch.inference_mode():
        for images, targets, metadata in tqdm(
            loader, desc=f"friendly-clean:{split}", unit="batch"
        ):
            if not isinstance(metadata, Mapping):
                raise ValueError("classification metadata is required")
            images = images.to(device=device, dtype=torch.float32, non_blocking=True)
            targets = targets.to(device=device, dtype=torch.long, non_blocking=True)
            bbox, crop_bbox, image_valid_mask, audit_indices = _metadata_tensors(
                metadata, device=device
            )
            logits = _model_logits(
                model=model,
                images=images,
                image_valid_mask=image_valid_mask,
                bbox=bbox,
                crop_bbox=crop_bbox,
                bbox_token_prior_source=bbox_token_prior_source,
                device=device,
            )
            probabilities = torch.softmax(logits.float(), dim=1)
            top2 = probabilities.topk(k=2, dim=1, largest=True, sorted=True).indices
            predictions = top2[:, 0]
            paths = metadata.get("paths")
            if not isinstance(paths, list) or len(paths) != int(images.size(0)):
                raise ValueError("classification paths are incomplete")
            for batch_index in range(int(images.size(0))):
                sample_index = int(audit_indices[batch_index].item())
                path = str(paths[batch_index])
                if not path:
                    raise ValueError(f"sample {sample_index} has no source path")
                row: Dict[str, object] = {
                    "split": split,
                    "sample_index": sample_index,
                    "image_path": path,
                    "source_stem": Path(path).stem.casefold(),
                    "target_index": int(targets[batch_index].item()),
                    "prediction_index": int(predictions[batch_index].item()),
                    "top1_index": int(top2[batch_index, 0].item()),
                    "top2_index": int(top2[batch_index, 1].item()),
                }
                for class_index in range(len(class_names)):
                    row[f"prob_{class_index}"] = float(
                        probabilities[batch_index, class_index].item()
                    )
                row["cohort"] = classify_row_cohort(row)
                row["boundary_margin"] = boundary_margin_from_row(row)
                rows.append(row)
            all_targets.append(targets.detach().cpu())
            all_probabilities.append(probabilities.detach().cpu())
    target_tensor = torch.cat(all_targets, dim=0)
    probability_tensor = torch.cat(all_probabilities, dim=0)
    prediction_tensor = probability_tensor.argmax(dim=1)
    metrics = build_metrics(target_tensor, prediction_tensor, class_names)
    cohort_counts: Dict[str, int] = {}
    for row in rows:
        name = str(row["cohort"])
        cohort_counts[name] = cohort_counts.get(name, 0) + 1
    return {
        "split": split,
        "rows": rows,
        "metrics": metrics,
        "cohort_counts": cohort_counts,
        "loader": loader_summary,
        "support": int(len(rows)),
        "source_group_count": int(len({str(row["source_stem"]) for row in rows})),
    }


def _write_csv(path: Path, rows: Sequence[Mapping[str, object]]) -> None:
    if not rows:
        path.write_text("", encoding="utf-8")
        return
    fieldnames: List[str] = []
    seen = set()
    for row in rows:
        for key in row:
            if key not in seen:
                seen.add(key)
                fieldnames.append(str(key))
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)


def audit_attack_pool(
    *,
    split: str,
    direction: str,
    dataset: Dataset,
    rows: Sequence[Mapping[str, object]],
    model: nn.Module,
    mean: Sequence[float],
    std: Sequence[float],
    bbox_token_prior_source: str,
    device: torch.device,
    batch_size: int,
    num_workers: int,
) -> Dict[str, object]:
    cohort = "protect_correct" if str(direction) == "protect" else "suppress_correct"
    indices = select_cohort_indices(rows, cohort=cohort, cap=0)
    if not indices:
        return {
            "split": split,
            "direction": direction,
            "eligible_count": 0,
            "crossed_count": 0,
            "crossing_rate": 0.0,
            "empty_mask_count": 0,
            "max_delta_rgb": 0.0,
            "max_outside_mask_delta": 0.0,
            "mean_active_delta_rgb": 0.0,
            "rows": [],
        }
    loader, loader_summary = _loader_for_indices(
        dataset=dataset,
        indices=indices,
        batch_size=int(batch_size),
        num_workers=int(num_workers),
        device=device,
        context=f"friendly_adversarial_attack_{split}_{direction}",
    )
    output_rows: List[Dict[str, object]] = []
    crossed_count = 0
    empty_mask_count = 0
    max_delta = 0.0
    max_outside = 0.0
    active_delta_sum = 0.0
    active_delta_count = 0.0
    for images, labels, metadata in tqdm(
        loader,
        desc=f"friendly-attack:{split}:{direction}",
        unit="batch",
    ):
        if not isinstance(metadata, Mapping):
            raise ValueError("classification metadata is required")
        images = images.to(device=device, dtype=torch.float32, non_blocking=True)
        labels = labels.to(device=device, dtype=torch.long, non_blocking=True)
        bbox, crop_bbox, image_valid_mask, audit_indices = _metadata_tensors(
            metadata, device=device
        )
        attack_mask = build_eroded_bbox_mask(
            crop_bbox,
            image_valid_mask,
            height=int(images.size(2)),
            width=int(images.size(3)),
        )

        def forward_logits(candidate_images: Tensor) -> Tensor:
            return _model_logits(
                model=model,
                images=candidate_images,
                image_valid_mask=image_valid_mask,
                bbox=bbox,
                crop_bbox=crop_bbox,
                bbox_token_prior_source=bbox_token_prior_source,
                device=device,
            )

        result = generate_friendly_adversarial_examples(
            images=images,
            labels=labels,
            attack_mask=attack_mask,
            forward_logits=forward_logits,
            mean=mean,
            std=std,
            direction=direction,
        )
        mask_float = result.attack_mask.to(dtype=result.delta_rgb.dtype)
        outside_delta = result.delta_rgb * (1.0 - mask_float)
        mask_counts = result.attack_mask.flatten(1).sum(dim=1)
        empty_mask_count += int((mask_counts == 0).sum().item())
        crossed_count += int(result.crossed.sum().item())
        max_delta = max(max_delta, float(result.delta_rgb.abs().max().item()))
        max_outside = max(max_outside, float(outside_delta.abs().max().item()))
        active_delta_sum += float((result.delta_rgb.abs() * mask_float).sum().item())
        active_delta_count += float(mask_float.sum().item() * int(images.size(1)))
        paths = metadata.get("paths")
        if not isinstance(paths, list) or len(paths) != int(images.size(0)):
            raise ValueError("classification paths are incomplete")
        for row_index in range(int(images.size(0))):
            output_rows.append(
                {
                    "split": split,
                    "direction": direction,
                    "sample_index": int(audit_indices[row_index].item()),
                    "image_path": str(paths[row_index]),
                    "target_index": int(labels[row_index].item()),
                    "crossed": bool(result.crossed[row_index].item()),
                    "crossing_step": int(result.crossing_step[row_index].item()),
                    "clean_margin": float(result.clean_margin[row_index].item()),
                    "adversarial_margin": float(
                        result.adversarial_margin[row_index].item()
                    ),
                    "mask_fraction": float(
                        result.attack_mask[row_index].float().mean().item()
                    ),
                    "max_delta_rgb": float(
                        result.delta_rgb[row_index].abs().max().item()
                    ),
                    "max_outside_mask_delta": float(
                        outside_delta[row_index].abs().max().item()
                    ),
                }
            )
    return {
        "split": split,
        "direction": direction,
        "eligible_count": int(len(indices)),
        "crossed_count": int(crossed_count),
        "crossing_rate": float(crossed_count / max(1, len(indices))),
        "empty_mask_count": int(empty_mask_count),
        "max_delta_rgb": float(max_delta),
        "max_outside_mask_delta": float(max_outside),
        "mean_active_delta_rgb": float(
            active_delta_sum / max(1.0, active_delta_count)
        ),
        "loader": loader_summary,
        "rows": output_rows,
    }


GradientSnapshot = Dict[str, Tensor]


def trainable_parameter_signature(
    model: nn.Module,
) -> Tuple[List[Tuple[str, nn.Parameter]], Dict[str, object]]:
    parameters = [
        (name, parameter)
        for name, parameter in model.named_parameters()
        if bool(parameter.requires_grad)
    ]
    if not parameters:
        raise ValueError("model has no trainable parameters")
    rows = [
        {
            "name": name,
            "shape": list(parameter.shape),
            "numel": int(parameter.numel()),
            "dtype": str(parameter.dtype),
        }
        for name, parameter in parameters
    ]
    canonical = json.dumps(rows, sort_keys=True, separators=(",", ":")).encode("utf-8")
    return parameters, {
        "count": int(len(parameters)),
        "numel": int(sum(parameter.numel() for _, parameter in parameters)),
        "sha256": hashlib.sha256(canonical).hexdigest(),
        "parameters": rows,
    }


def snapshot_model_gradients(
    parameters: Sequence[Tuple[str, nn.Parameter]],
) -> GradientSnapshot:
    snapshot: GradientSnapshot = {}
    for name, parameter in parameters:
        if parameter.grad is None:
            snapshot[name] = torch.zeros_like(parameter, device="cpu", dtype=torch.float32)
        else:
            snapshot[name] = parameter.grad.detach().float().cpu().clone()
    return snapshot


def combine_gradients(
    first: GradientSnapshot,
    second: GradientSnapshot,
    *,
    first_weight: float = 0.5,
    second_weight: float = 0.5,
) -> GradientSnapshot:
    if set(first) != set(second):
        raise ValueError("gradient snapshots have different parameter sets")
    return {
        name: float(first_weight) * first[name] + float(second_weight) * second[name]
        for name in first
    }


def gradient_cosine(first: GradientSnapshot, second: GradientSnapshot) -> float:
    if set(first) != set(second):
        raise ValueError("gradient snapshots have different parameter sets")
    dot = 0.0
    first_norm = 0.0
    second_norm = 0.0
    for name in first:
        first_flat = first[name].double().flatten()
        second_flat = second[name].double().flatten()
        dot += float(torch.dot(first_flat, second_flat).item())
        first_norm += float(torch.dot(first_flat, first_flat).item())
        second_norm += float(torch.dot(second_flat, second_flat).item())
    denominator = math.sqrt(first_norm) * math.sqrt(second_norm)
    if denominator <= 0.0:
        return float("nan")
    return float(dot / denominator)


def gradient_norm(snapshot: GradientSnapshot) -> float:
    squared = 0.0
    for gradient in snapshot.values():
        flat = gradient.double().flatten()
        squared += float(torch.dot(flat, flat).item())
    return float(math.sqrt(squared))


def accumulate_gradient(
    *,
    dataset: Dataset,
    indices: Sequence[int],
    model: nn.Module,
    parameters: Sequence[Tuple[str, nn.Parameter]],
    mean: Sequence[float],
    std: Sequence[float],
    bbox_token_prior_source: str,
    device: torch.device,
    objective: str,
    direction: Optional[str],
) -> GradientSnapshot:
    if not indices:
        raise ValueError("gradient objective has no samples")
    if str(objective) not in {"clean", "defense"}:
        raise ValueError("objective must be clean or defense")
    if str(objective) == "defense" and str(direction) not in {"protect", "suppress"}:
        raise ValueError("defense objective requires protect or suppress direction")
    loader = DataLoader(
        IndexedSubsetDataset(dataset, indices),
        batch_size=GRADIENT_BATCH_SIZE,
        shuffle=False,
        num_workers=0,
        pin_memory=device.type == "cuda",
        collate_fn=_collate_classification,
    )
    model.zero_grad(set_to_none=True)
    denominator = float(len(indices))
    for images, labels, metadata in loader:
        if not isinstance(metadata, Mapping):
            raise ValueError("classification metadata is required")
        images = images.to(device=device, dtype=torch.float32, non_blocking=True)
        labels = labels.to(device=device, dtype=torch.long, non_blocking=True)
        bbox, crop_bbox, image_valid_mask, _ = _metadata_tensors(metadata, device=device)
        selected_images = images
        if str(objective) == "defense":
            attack_mask = build_eroded_bbox_mask(
                crop_bbox,
                image_valid_mask,
                height=int(images.size(2)),
                width=int(images.size(3)),
            )

            def forward_logits(candidate_images: Tensor) -> Tensor:
                return _model_logits(
                    model=model,
                    images=candidate_images,
                    image_valid_mask=image_valid_mask,
                    bbox=bbox,
                    crop_bbox=crop_bbox,
                    bbox_token_prior_source=bbox_token_prior_source,
                    device=device,
                )

            attack = generate_friendly_adversarial_examples(
                images=images,
                labels=labels,
                attack_mask=attack_mask,
                forward_logits=forward_logits,
                mean=mean,
                std=std,
                direction=str(direction),
            )
            selected_images = attack.adversarial_images.detach()
        logits = _model_logits(
            model=model,
            images=selected_images,
            image_valid_mask=image_valid_mask,
            bbox=bbox,
            crop_bbox=crop_bbox,
            bbox_token_prior_source=bbox_token_prior_source,
            device=device,
        )
        loss = F.cross_entropy(logits, labels, reduction="sum") / denominator
        loss.backward()
    snapshot = snapshot_model_gradients(parameters)
    model.zero_grad(set_to_none=True)
    return snapshot


def _gradient_bundle(
    *,
    dataset: Dataset,
    protect_indices: Sequence[int],
    suppress_indices: Sequence[int],
    model: nn.Module,
    parameters: Sequence[Tuple[str, nn.Parameter]],
    mean: Sequence[float],
    std: Sequence[float],
    bbox_token_prior_source: str,
    device: torch.device,
) -> Dict[str, GradientSnapshot]:
    protect_defense = accumulate_gradient(
        dataset=dataset,
        indices=protect_indices,
        model=model,
        parameters=parameters,
        mean=mean,
        std=std,
        bbox_token_prior_source=bbox_token_prior_source,
        device=device,
        objective="defense",
        direction="protect",
    )
    suppress_defense = accumulate_gradient(
        dataset=dataset,
        indices=suppress_indices,
        model=model,
        parameters=parameters,
        mean=mean,
        std=std,
        bbox_token_prior_source=bbox_token_prior_source,
        device=device,
        objective="defense",
        direction="suppress",
    )
    protect_clean = accumulate_gradient(
        dataset=dataset,
        indices=protect_indices,
        model=model,
        parameters=parameters,
        mean=mean,
        std=std,
        bbox_token_prior_source=bbox_token_prior_source,
        device=device,
        objective="clean",
        direction=None,
    )
    suppress_clean = accumulate_gradient(
        dataset=dataset,
        indices=suppress_indices,
        model=model,
        parameters=parameters,
        mean=mean,
        std=std,
        bbox_token_prior_source=bbox_token_prior_source,
        device=device,
        objective="clean",
        direction=None,
    )
    return {
        "protect_defense": protect_defense,
        "suppress_defense": suppress_defense,
        "combined_defense": combine_gradients(protect_defense, suppress_defense),
        "combined_clean": combine_gradients(protect_clean, suppress_clean),
    }


def run_gradient_direction_audit(
    *,
    train_dataset: Dataset,
    val_dataset: Dataset,
    train_rows: Sequence[Mapping[str, object]],
    val_rows: Sequence[Mapping[str, object]],
    fold_assignments: Mapping[int, int],
    model: nn.Module,
    parameters: Sequence[Tuple[str, nn.Parameter]],
    mean: Sequence[float],
    std: Sequence[float],
    bbox_token_prior_source: str,
    device: torch.device,
) -> Dict[str, object]:
    fold_rows: List[Dict[str, object]] = []
    for fold_index in range(FOLDS):
        protect = select_cohort_indices(
            train_rows,
            cohort="protect_correct",
            fold_assignments=fold_assignments,
            exclude_fold=fold_index,
        )
        suppress = select_cohort_indices(
            train_rows,
            cohort="suppress_correct",
            fold_assignments=fold_assignments,
            exclude_fold=fold_index,
        )
        positive_error = select_cohort_indices(
            train_rows,
            cohort="positive_error",
            fold_assignments=fold_assignments,
            include_fold=fold_index,
        )
        negative_error = select_cohort_indices(
            train_rows,
            cohort="negative_error",
            fold_assignments=fold_assignments,
            include_fold=fold_index,
        )
        row: Dict[str, object] = {
            "fold": fold_index,
            "protect_count": len(protect),
            "suppress_count": len(suppress),
            "positive_error_count": len(positive_error),
            "negative_error_count": len(negative_error),
        }
        if not protect or not suppress or not positive_error or not negative_error:
            row.update(
                {
                    "complete": False,
                    "protect_correction_cosine": None,
                    "suppress_correction_cosine": None,
                    "combined_clean_cosine": None,
                }
            )
            fold_rows.append(row)
            continue
        bundle = _gradient_bundle(
            dataset=train_dataset,
            protect_indices=protect,
            suppress_indices=suppress,
            model=model,
            parameters=parameters,
            mean=mean,
            std=std,
            bbox_token_prior_source=bbox_token_prior_source,
            device=device,
        )
        positive_correction = accumulate_gradient(
            dataset=train_dataset,
            indices=positive_error,
            model=model,
            parameters=parameters,
            mean=mean,
            std=std,
            bbox_token_prior_source=bbox_token_prior_source,
            device=device,
            objective="clean",
            direction=None,
        )
        negative_correction = accumulate_gradient(
            dataset=train_dataset,
            indices=negative_error,
            model=model,
            parameters=parameters,
            mean=mean,
            std=std,
            bbox_token_prior_source=bbox_token_prior_source,
            device=device,
            objective="clean",
            direction=None,
        )
        row.update(
            {
                "complete": True,
                "protect_correction_cosine": gradient_cosine(
                    bundle["protect_defense"], positive_correction
                ),
                "suppress_correction_cosine": gradient_cosine(
                    bundle["suppress_defense"], negative_correction
                ),
                "combined_clean_cosine": gradient_cosine(
                    bundle["combined_defense"], bundle["combined_clean"]
                ),
                "protect_defense_norm": gradient_norm(bundle["protect_defense"]),
                "suppress_defense_norm": gradient_norm(bundle["suppress_defense"]),
            }
        )
        fold_rows.append(row)

    train_protect = select_cohort_indices(train_rows, cohort="protect_correct")
    train_suppress = select_cohort_indices(train_rows, cohort="suppress_correct")
    val_protect = select_cohort_indices(val_rows, cohort="protect_correct")
    val_suppress = select_cohort_indices(val_rows, cohort="suppress_correct")
    val_positive_error = select_cohort_indices(val_rows, cohort="positive_error")
    val_negative_error = select_cohort_indices(val_rows, cohort="negative_error")
    transfer_complete = all(
        (
            train_protect,
            train_suppress,
            val_protect,
            val_suppress,
            val_positive_error,
            val_negative_error,
        )
    )
    transfer: Dict[str, object] = {
        "complete": bool(transfer_complete),
        "train_protect_count": len(train_protect),
        "train_suppress_count": len(train_suppress),
        "val_protect_count": len(val_protect),
        "val_suppress_count": len(val_suppress),
        "val_positive_error_count": len(val_positive_error),
        "val_negative_error_count": len(val_negative_error),
    }
    if transfer_complete:
        train_bundle = _gradient_bundle(
            dataset=train_dataset,
            protect_indices=train_protect,
            suppress_indices=train_suppress,
            model=model,
            parameters=parameters,
            mean=mean,
            std=std,
            bbox_token_prior_source=bbox_token_prior_source,
            device=device,
        )
        val_bundle = _gradient_bundle(
            dataset=val_dataset,
            protect_indices=val_protect,
            suppress_indices=val_suppress,
            model=model,
            parameters=parameters,
            mean=mean,
            std=std,
            bbox_token_prior_source=bbox_token_prior_source,
            device=device,
        )
        val_positive_correction = accumulate_gradient(
            dataset=val_dataset,
            indices=val_positive_error,
            model=model,
            parameters=parameters,
            mean=mean,
            std=std,
            bbox_token_prior_source=bbox_token_prior_source,
            device=device,
            objective="clean",
            direction=None,
        )
        val_negative_correction = accumulate_gradient(
            dataset=val_dataset,
            indices=val_negative_error,
            model=model,
            parameters=parameters,
            mean=mean,
            std=std,
            bbox_token_prior_source=bbox_token_prior_source,
            device=device,
            objective="clean",
            direction=None,
        )
        transfer.update(
            {
                "protect_validation_correction_cosine": gradient_cosine(
                    train_bundle["protect_defense"], val_positive_correction
                ),
                "suppress_validation_correction_cosine": gradient_cosine(
                    train_bundle["suppress_defense"], val_negative_correction
                ),
                "train_combined_clean_cosine": gradient_cosine(
                    train_bundle["combined_defense"], train_bundle["combined_clean"]
                ),
                "val_combined_clean_cosine": gradient_cosine(
                    val_bundle["combined_defense"], val_bundle["combined_clean"]
                ),
            }
        )
    else:
        transfer.update(
            {
                "protect_validation_correction_cosine": None,
                "suppress_validation_correction_cosine": None,
                "train_combined_clean_cosine": None,
                "val_combined_clean_cosine": None,
            }
        )
    return {"folds": fold_rows, "transfer": transfer}


def _class1_metrics(metrics: Mapping[str, object]) -> Mapping[str, object]:
    per_class = metrics.get("per_class")
    if not isinstance(per_class, list) or len(per_class) <= FOCUS_CLASS:
        raise ValueError("metrics do not contain class-1 values")
    value = per_class[FOCUS_CLASS]
    if not isinstance(value, Mapping):
        raise ValueError("class-1 metrics are invalid")
    return value


def _public_attack(result: Mapping[str, object]) -> Dict[str, object]:
    return {key: value for key, value in result.items() if key != "rows"}


def _gate_check(name: str, value: object, requirement: str, passed: bool) -> Dict[str, object]:
    return {
        "name": name,
        "value": value,
        "requirement": requirement,
        "passed": bool(passed),
    }


def assess_smoke_permission(
    *,
    preflight: bool,
    checkpoint_sha256: str,
    train_support: int,
    val_support: int,
    source_overlap_count: int,
    val_metrics: Mapping[str, object],
    attacks: Mapping[str, Mapping[str, Mapping[str, object]]],
    gradients: Mapping[str, object],
) -> Dict[str, object]:
    class1 = _class1_metrics(val_metrics)
    checks: List[Dict[str, object]] = []
    checks.append(
        _gate_check(
            "full_protocol_not_preflight",
            not bool(preflight),
            "true",
            not bool(preflight),
        )
    )
    checks.append(
        _gate_check(
            "checkpoint_sha256",
            checkpoint_sha256,
            EXPECTED_CHECKPOINT_SHA256,
            str(checkpoint_sha256) == EXPECTED_CHECKPOINT_SHA256,
        )
    )
    checks.append(
        _gate_check(
            "full_support",
            {"train": int(train_support), "val": int(val_support)},
            f"{EXPECTED_TRAIN_COUNT}/{EXPECTED_VAL_COUNT}",
            int(train_support) == EXPECTED_TRAIN_COUNT
            and int(val_support) == EXPECTED_VAL_COUNT,
        )
    )
    checks.append(
        _gate_check(
            "source_overlap",
            int(source_overlap_count),
            "0",
            int(source_overlap_count) == 0,
        )
    )
    val_macro = float(val_metrics["macro_f1"])
    val_class1 = float(class1["f1"])
    checks.append(
        _gate_check(
            "fp32_keeper_reproduction",
            {"macro_f1": val_macro, "class1_f1": val_class1},
            (
                f"abs(macro-{EXPECTED_KEEPER_VAL_MACRO})<=1e-6 and "
                f"abs(class1-{EXPECTED_KEEPER_VAL_CLASS1})<=1e-6"
            ),
            abs(val_macro - EXPECTED_KEEPER_VAL_MACRO) <= 1e-6
            and abs(val_class1 - EXPECTED_KEEPER_VAL_CLASS1) <= 1e-6,
        )
    )
    for split in ("train", "val"):
        for direction in ("protect", "suppress"):
            result = attacks[split][direction]
            rate = float(result["crossing_rate"])
            checks.append(
                _gate_check(
                    f"{split}_{direction}_crossing_rate",
                    rate,
                    ">=0.10",
                    int(result["eligible_count"]) > 0 and rate >= 0.10,
                )
            )
    max_delta = max(
        float(attacks[split][direction]["max_delta_rgb"])
        for split in ("train", "val")
        for direction in ("protect", "suppress")
    )
    max_outside = max(
        float(attacks[split][direction]["max_outside_mask_delta"])
        for split in ("train", "val")
        for direction in ("protect", "suppress")
    )
    empty_masks = sum(
        int(attacks[split][direction]["empty_mask_count"])
        for split in ("train", "val")
        for direction in ("protect", "suppress")
    )
    checks.append(
        _gate_check(
            "perturbation_bound",
            max_delta,
            f"<={ATTACK_EPSILON + 1e-7:.10f}",
            max_delta <= ATTACK_EPSILON + 1e-7,
        )
    )
    checks.append(
        _gate_check(
            "outside_mask_delta",
            max_outside,
            "exactly 0",
            max_outside == 0.0,
        )
    )
    checks.append(
        _gate_check(
            "nonempty_attack_masks",
            empty_masks,
            "0 empty masks",
            empty_masks == 0,
        )
    )

    fold_rows = gradients.get("folds")
    if not isinstance(fold_rows, list):
        fold_rows = []
    complete_folds = [row for row in fold_rows if bool(row.get("complete", False))]
    protect_cosines = [
        float(row["protect_correction_cosine"])
        for row in complete_folds
        if row.get("protect_correction_cosine") is not None
    ]
    suppress_cosines = [
        float(row["suppress_correction_cosine"])
        for row in complete_folds
        if row.get("suppress_correction_cosine") is not None
    ]
    fold_complete = len(complete_folds) == FOLDS
    checks.append(
        _gate_check(
            "all_folds_have_both_correction_targets",
            len(complete_folds),
            f"{FOLDS}/{FOLDS}",
            fold_complete,
        )
    )
    for name, values in (("protect", protect_cosines), ("suppress", suppress_cosines)):
        median = float(np.median(values)) if values else float("nan")
        positive = int(sum(math.isfinite(value) and value > 0.0 for value in values))
        checks.append(
            _gate_check(
                f"train_{name}_median_fold_cosine",
                median,
                ">=0.05",
                len(values) == FOLDS and math.isfinite(median) and median >= 0.05,
            )
        )
        checks.append(
            _gate_check(
                f"train_{name}_positive_folds",
                positive,
                ">=4/5",
                len(values) == FOLDS and positive >= 4,
            )
        )

    transfer = gradients.get("transfer")
    if not isinstance(transfer, Mapping):
        transfer = {}
    protect_val = transfer.get("protect_validation_correction_cosine")
    suppress_val = transfer.get("suppress_validation_correction_cosine")
    train_clean = transfer.get("train_combined_clean_cosine")
    val_clean = transfer.get("val_combined_clean_cosine")
    transfer_checks = (
        ("protect_validation_correction_cosine", protect_val, 0.02),
        ("suppress_validation_correction_cosine", suppress_val, 0.02),
        ("train_combined_clean_cosine", train_clean, 0.10),
        ("val_combined_clean_cosine", val_clean, 0.0),
    )
    for name, value, threshold in transfer_checks:
        numeric = float(value) if value is not None else float("nan")
        checks.append(
            _gate_check(
                name,
                numeric,
                f">={threshold:.2f}",
                math.isfinite(numeric) and numeric >= threshold,
            )
        )
    passed = int(sum(bool(check["passed"]) for check in checks))
    return {
        "smoke_permission": passed == len(checks),
        "passed": passed,
        "total": int(len(checks)),
        "checks": checks,
    }


def _public_clean(result: Mapping[str, object]) -> Dict[str, object]:
    return {key: value for key, value in result.items() if key != "rows"}


def run_audit(args: argparse.Namespace) -> Dict[str, object]:
    if int(args.batch_size) < 1 or int(args.attack_batch_size) < 1:
        raise ValueError("batch sizes must be positive")
    if int(args.num_workers) < 0:
        raise ValueError("num_workers must be non-negative")
    capped = int(args.max_train_samples) > 0 or int(args.max_val_samples) > 0
    if capped and not bool(args.allow_preflight):
        raise ValueError("sample caps require --allow-preflight")

    output_dir = Path(args.output_dir).resolve()
    data_path = Path(args.data).resolve()
    if _is_relative_to(output_dir, data_path.parent):
        raise ValueError("output directory must stay outside the raw dataset")
    if output_dir.exists() and any(output_dir.iterdir()):
        raise FileExistsError(f"output directory is not empty: {output_dir}")
    ensure_dir(output_dir)
    set_seed(int(args.seed))
    device = _resolve_device(str(args.device))

    checkpoint_path = Path(args.checkpoint).resolve()
    checkpoint_sha = sha256_file(checkpoint_path)
    if checkpoint_sha != EXPECTED_CHECKPOINT_SHA256:
        raise ValueError(
            "readiness protocol is locked to the keeper checkpoint hash: "
            f"expected {EXPECTED_CHECKPOINT_SHA256}, got {checkpoint_sha}"
        )
    checkpoint = load_checkpoint(checkpoint_path, map_location="cpu")
    data_spec = load_data_spec(data_path, class_name_mode="raw", expected_num_classes=5)
    if str(data_spec.data_format).strip().lower() == "classification_folder":
        raise ValueError("friendly foreground readiness is locked to bbox-aware yolo_f")
    class_names = list(checkpoint.get("class_names", data_spec.class_names))
    if class_names != list(data_spec.class_names):
        raise ValueError("checkpoint and dataset class orders differ")
    model = build_model_from_checkpoint(dict(checkpoint), num_classes=len(class_names))
    model.to(device=device, dtype=torch.float32)
    model.eval()
    parameters, parameter_signature = trainable_parameter_signature(model)
    mean, std = checkpoint_input_normalization(checkpoint)
    image_size = int(checkpoint.get("model_config", {}).get("image_size", 256))
    transform = _build_eval_transform_from_checkpoint(dict(checkpoint), image_size=image_size)
    datasets: Dict[str, Dataset] = {}
    for split in ("train", "val"):
        datasets[split] = _build_classification_dataset(
            data_spec=data_spec,
            split=split,
            transform=transform,
            checkpoint=dict(checkpoint),
        )
    caps = {"train": int(args.max_train_samples), "val": int(args.max_val_samples)}
    indices = {
        split: list(range(min(len(datasets[split]), caps[split])))
        if caps[split] > 0
        else list(range(len(datasets[split])))
        for split in ("train", "val")
    }
    bbox_token_prior_source = str(
        checkpoint.get("model_config", {}).get("bbox_token_prior_source", "bbox")
        or "bbox"
    )

    clean_results = {
        split: collect_clean_split(
            split=split,
            dataset=datasets[split],
            indices=indices[split],
            model=model,
            class_names=class_names,
            bbox_token_prior_source=bbox_token_prior_source,
            device=device,
            batch_size=int(args.batch_size),
            num_workers=int(args.num_workers),
        )
        for split in ("train", "val")
    }
    train_rows = clean_results["train"]["rows"]
    val_rows = clean_results["val"]["rows"]
    if not isinstance(train_rows, list) or not isinstance(val_rows, list):
        raise RuntimeError("clean audit rows are invalid")
    fold_assignments = assign_source_grouped_folds(
        train_rows,
        folds=FOLDS,
        seed=int(args.seed),
    )
    for row in train_rows:
        row["readiness_fold"] = int(fold_assignments[int(row["sample_index"])])

    attacks: Dict[str, Dict[str, Dict[str, object]]] = {}
    attack_rows: List[Mapping[str, object]] = []
    for split in ("train", "val"):
        attacks[split] = {}
        for direction in ("protect", "suppress"):
            result = audit_attack_pool(
                split=split,
                direction=direction,
                dataset=datasets[split],
                rows=clean_results[split]["rows"],
                model=model,
                mean=mean,
                std=std,
                bbox_token_prior_source=bbox_token_prior_source,
                device=device,
                batch_size=int(args.attack_batch_size),
                num_workers=int(args.num_workers),
            )
            attacks[split][direction] = result
            attack_rows.extend(result["rows"])
            model.zero_grad(set_to_none=True)

    gradients = run_gradient_direction_audit(
        train_dataset=datasets["train"],
        val_dataset=datasets["val"],
        train_rows=train_rows,
        val_rows=val_rows,
        fold_assignments=fold_assignments,
        model=model,
        parameters=parameters,
        mean=mean,
        std=std,
        bbox_token_prior_source=bbox_token_prior_source,
        device=device,
    )
    train_sources = {str(row["source_stem"]) for row in train_rows}
    val_sources = {str(row["source_stem"]) for row in val_rows}
    source_overlap = sorted(train_sources & val_sources)
    gate = assess_smoke_permission(
        preflight=capped,
        checkpoint_sha256=checkpoint_sha,
        train_support=int(clean_results["train"]["support"]),
        val_support=int(clean_results["val"]["support"]),
        source_overlap_count=len(source_overlap),
        val_metrics=clean_results["val"]["metrics"],
        attacks=attacks,
        gradients=gradients,
    )

    _write_csv(output_dir / "train_clean_predictions.csv", train_rows)
    _write_csv(output_dir / "val_clean_predictions.csv", val_rows)
    _write_csv(output_dir / "friendly_attack_rows.csv", attack_rows)
    _write_csv(output_dir / "gradient_fold_cosines.csv", gradients["folds"])
    summary = {
        "mode": "friendly_foreground_adversarial_readiness",
        "protocol_locked": True,
        "preflight": bool(capped),
        "test_split_used": False,
        "raw_dataset_modified": False,
        "model_fit": False,
        "checkpoint_written": False,
        "checkpoint": str(checkpoint_path),
        "checkpoint_sha256": checkpoint_sha,
        "data": str(data_path),
        "device": str(device),
        "class_names": class_names,
        "image_size": image_size,
        "bbox_token_prior_source": bbox_token_prior_source,
        "protocol": {
            "friendly_adversarial_training": FAT_URL,
            "advprop": ADVPROP_URL,
            "boundary_guided_adversarial_training": LBGAT_URL,
            "focus_class": FOCUS_CLASS,
            "negative_classes": list(NEGATIVE_CLASSES),
            "epsilon_rgb": ATTACK_EPSILON,
            "step_size_rgb": ATTACK_STEP_SIZE,
            "steps": ATTACK_STEPS,
            "random_start": False,
            "early_stop": "first declared class-1 boundary crossing",
            "bbox_erode_ratio_each_side": BBOX_ERODE_RATIO,
            "gradient_sample_cap_per_cohort": GRADIENT_SAMPLE_CAP,
            "gradient_batch_size": GRADIENT_BATCH_SIZE,
            "folds": FOLDS,
            "train_keeper_in_sample_caveat": (
                "source folds measure update-direction transfer; the keeper representation "
                "was trained on the full train split and is not OOF"
            ),
            "threshold_or_attack_sweep": False,
        },
        "trainable_parameter_signature": parameter_signature,
        "source_overlap": {"count": len(source_overlap), "examples": source_overlap[:10]},
        "clean": {
            split: _public_clean(clean_results[split]) for split in ("train", "val")
        },
        "attacks": {
            split: {
                direction: _public_attack(attacks[split][direction])
                for direction in ("protect", "suppress")
            }
            for split in ("train", "val")
        },
        "gradients": gradients,
        "gate": gate,
        "artifact_manifest_path": "artifact_manifest.json",
    }
    json_dump(output_dir / "summary.json", summary)
    val_class1 = _class1_metrics(clean_results["val"]["metrics"])
    readme = [
        "# Friendly Foreground Adversarial Readiness",
        "",
        "- Full FP32 train/validation audit; no test, model fit, checkpoint, or raw-data edit.",
        f"- Keeper val macro/class1: `{clean_results['val']['metrics']['macro_f1']:.6f}/{float(val_class1['f1']):.6f}`.",
        f"- Train attack crossing protect/suppress: `{attacks['train']['protect']['crossing_rate']:.6f}/{attacks['train']['suppress']['crossing_rate']:.6f}`.",
        f"- Val attack crossing protect/suppress: `{attacks['val']['protect']['crossing_rate']:.6f}/{attacks['val']['suppress']['crossing_rate']:.6f}`.",
        f"- Smoke permission: `{gate['smoke_permission']}` ({gate['passed']}/{gate['total']} checks).",
        "",
        "Train fold results are update-direction diagnostics over an in-sample keeper representation, not keeper OOF performance.",
    ]
    (output_dir / "README.md").write_text("\n".join(readme) + "\n", encoding="utf-8")
    _write_artifact_manifest(
        output_dir,
        mode="friendly_foreground_adversarial_readiness_evidence_manifest",
    )
    return summary


def main() -> None:
    summary = run_audit(parse_args())
    print(json.dumps(summary["gate"], indent=2))


if __name__ == "__main__":
    main()
