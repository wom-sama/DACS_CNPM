from __future__ import annotations

from copy import deepcopy
import hashlib
import json
import math
from pathlib import Path
import time
from typing import Dict, List, Mapping, Optional, Sequence, Tuple

import cv2
import numpy as np
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import roc_auc_score
from sklearn.preprocessing import StandardScaler
import torch
from torch import Tensor, nn
from torch.utils.data import DataLoader, Dataset

from trkh.core.utils import build_safe_dataloader_kwargs
from trkh.tools.audit_cutpaste_surface_response_a0 import (
    _bbox_support,
    _to_rgb,
)
from trkh.tools.audit_more_model_rebalancing_readiness import CleanTrainRow
from trkh.tools.nsa_class_pair_poisson import (
    ClassQueryMismatchAdapter,
    PoissonView,
    deterministic_deranged_queries,
    downsample_support,
    downsample_target,
    exclude_entering_chromatic_occluders,
    geometry_sha256,
    make_poisson_view,
    map_summaries,
    masked_bce_per_sample,
    sample_poisson_geometry,
    sample_target_geometry,
    state_dict_sha256,
    validate_role_parameter_parity,
)


SEED = 20260724
FOLDS = (0, 1, 2, 3, 4)
ROLE_NAMES = (
    "poisson_query_candidate",
    "clean_query_control",
    "no_query_control",
    "permuted_query_control",
)
READOUT_ROLES = ("base_context", *ROLE_NAMES)
EPOCHS = 2
LEARNING_RATE = 3e-4
WEIGHT_DECAY = 1e-4
GRADIENT_CLIP = 5.0
FIT_TP_RETENTION = 0.98
DIAGNOSTIC_PER_CLASS = 15
MAX_BLEND_GEOMETRY_ATTEMPTS = 32
MINIMUM_SUPPORT_EXTENT = 16
GEOMETRY_INELIGIBLE_SAMPLE_INDICES = (7373,)


def _worker_init(_: int) -> None:
    cv2.setNumThreads(1)


def _digest(*parts: object) -> str:
    return hashlib.sha256("|".join(str(part) for part in parts).encode("utf-8")).hexdigest()


def _role_record(record: Mapping[str, object]) -> Dict[str, object]:
    return {
        key: value
        for key, value in record.items()
        if key not in {"selected_component_mask", "dilated_exclusion_mask", "components"}
    }


def _array_sha256(value: np.ndarray) -> str:
    array = np.ascontiguousarray(np.asarray(value))
    digest = hashlib.sha256()
    digest.update(str(array.dtype).encode("ascii"))
    digest.update(np.asarray(array.shape, dtype=np.int64).tobytes())
    digest.update(array.tobytes())
    return digest.hexdigest()


def build_balanced_fit_panel(
    rows: Sequence[CleanTrainRow],
    *,
    held_fold: int,
) -> Tuple[List[int], Dict[str, object]]:
    excluded = set(GEOMETRY_INELIGIBLE_SAMPLE_INDICES)
    fit = [
        row
        for row in rows
        if int(row.fold) != int(held_fold)
        and int(row.sample_index) not in excluded
    ]
    by_class = {
        target: [row for row in fit if int(row.target) == target]
        for target in range(5)
    }
    minimum = min(len(values) for values in by_class.values())
    selected: List[CleanTrainRow] = []
    for target, values in by_class.items():
        ranked = sorted(
            values,
            key=lambda row: (
                _digest(
                    SEED,
                    "fit-select",
                    held_fold,
                    target,
                    row.sample_index,
                    row.source_stem,
                ),
                row.sample_index,
            ),
        )
        selected.extend(ranked[:minimum])
    selected.sort(
        key=lambda row: (
            _digest(
                SEED,
                "fit-order",
                held_fold,
                row.sample_index,
                row.source_stem,
            ),
            row.sample_index,
        )
    )
    indices = [int(row.sample_index) for row in selected]
    if len(indices) != len(set(indices)):
        raise ValueError("balanced fit panel duplicates a sample index")
    if any(int(rows[index].fold) == int(held_fold) for index in indices):
        raise ValueError("held-fold row leaked into balanced fit panel")
    counts = {
        target: sum(int(rows[index].target) == target for index in indices)
        for target in range(5)
    }
    if set(counts.values()) != {minimum}:
        raise ValueError(f"balanced fit panel is not class balanced: {counts}")
    return indices, {
        "held_fold": int(held_fold),
        "fit_rows_available": len(fit),
        "selected_rows": len(indices),
        "per_class_count": minimum,
        "class_counts": counts,
        "duplicates": len(indices) - len(set(indices)),
        "held_rows_selected": 0,
        "geometry_ineligible_indices": list(
            GEOMETRY_INELIGIBLE_SAMPLE_INDICES
        ),
        "geometry_ineligible_selected": sum(
            index in excluded for index in indices
        ),
        "ordered_index_sha256": _digest(*indices),
    }


def _class_pools(
    rows: Sequence[CleanTrainRow],
    *,
    allowed_fold: Optional[int] = None,
    excluded_fold: Optional[int] = None,
) -> Dict[int, List[int]]:
    pools: Dict[int, List[int]] = {}
    for target in range(5):
        selected = [
            row
            for row in rows
            if int(row.target) == target
            and int(row.sample_index)
            not in GEOMETRY_INELIGIBLE_SAMPLE_INDICES
            and (allowed_fold is None or int(row.fold) == int(allowed_fold))
            and (excluded_fold is None or int(row.fold) != int(excluded_fold))
        ]
        selected.sort(
            key=lambda row: (
                _digest(
                    SEED,
                    "donor-pool",
                    allowed_fold,
                    excluded_fold,
                    target,
                    row.sample_index,
                    row.source_stem,
                ),
                row.sample_index,
            )
        )
        pools[target] = [int(row.sample_index) for row in selected]
        if not pools[target]:
            raise ValueError(f"donor pool for class {target} is empty")
    return pools


def _circular_candidates(
    pool: Sequence[int],
    rows: Sequence[CleanTrainRow],
    *,
    target_row: CleanTrainRow,
    epoch: int,
    role: str,
    limit: int = 64,
) -> List[int]:
    if not pool:
        raise ValueError("donor candidate pool is empty")
    offset = int(
        _digest(
            SEED,
            "donor-offset",
            target_row.sample_index,
            epoch,
            role,
        )[:16],
        16,
    ) % len(pool)
    output: List[int] = []
    for step in range(len(pool)):
        index = int(pool[(offset + step) % len(pool)])
        if rows[index].source_stem == target_row.source_stem:
            continue
        output.append(index)
        if len(output) == int(limit):
            break
    if not output:
        raise RuntimeError("no source-disjoint donor candidate exists")
    return output


def cross_class_schedule(target: int, sample_index: int, epoch: int) -> Tuple[int, int]:
    others = [value for value in range(5) if value != int(target)]
    offset = int(_digest(SEED, "cross-class", sample_index)[:16], 16) % 4
    ordered = others[offset:] + others[:offset]
    start = 2 * int(epoch)
    return int(ordered[start]), int(ordered[start + 1])


def _rgb_uint8(tensor: Tensor, semantics: Mapping[str, object]) -> np.ndarray:
    rgb = _to_rgb(tensor, semantics)
    return np.rint(
        rgb.permute(1, 2, 0).numpy().clip(0.0, 1.0) * 255.0
    ).astype(np.uint8)


def _normalize_rgb(rgb: np.ndarray, semantics: Mapping[str, object]) -> Tensor:
    value = torch.from_numpy(np.asarray(rgb, dtype=np.uint8).copy()).permute(2, 0, 1)
    value = value.float() / 255.0
    mean = torch.tensor(semantics["input_mean"], dtype=torch.float32).view(3, 1, 1)
    std = torch.tensor(semantics["input_std"], dtype=torch.float32).view(3, 1, 1)
    return (value - mean) / std


def load_processed_sample(
    dataset: Dataset,
    semantics: Mapping[str, object],
    sample_index: int,
) -> Dict[str, object]:
    tensor, label, metadata = dataset[int(sample_index)]
    if "crop_bbox" not in metadata or "image_mask" not in metadata:
        raise ValueError("NSA sample lacks crop_bbox/image_mask")
    rgb_tensor = _to_rgb(tensor, semantics)
    base_support_tensor = _bbox_support(
        rgb_tensor,
        metadata["crop_bbox"],
        metadata["image_mask"],
    )
    rgb = _rgb_uint8(tensor, semantics)
    base_support = base_support_tensor.numpy().astype(bool)
    support, exclusion = exclude_entering_chromatic_occluders(rgb, base_support)
    return {
        "tensor": tensor.float(),
        "label": int(label),
        "metadata": metadata,
        "rgb": rgb,
        "base_support": base_support,
        "support": support,
        "occluder_exclusion": exclusion,
    }


def _find_donor_view(
    dataset: Dataset,
    semantics: Mapping[str, object],
    rows: Sequence[CleanTrainRow],
    target: Mapping[str, object],
    target_row: CleanTrainRow,
    target_geometry,
    candidates: Sequence[int],
    *,
    epoch: int,
    source_role_id: int,
) -> Tuple[int, Dict[str, object], PoissonView]:
    errors: List[str] = []
    for candidate_index in candidates:
        donor = load_processed_sample(dataset, semantics, int(candidate_index))
        try:
            geometry = sample_poisson_geometry(
                np.asarray(target["support"], dtype=bool),
                np.asarray(donor["support"], dtype=bool),
                sample_index=int(target_row.sample_index),
                epoch=int(epoch),
                target_role_id=0,
                source_role_id=int(source_role_id) + int(candidate_index),
                matched_target=target_geometry,
            )
            view = make_poisson_view(
                np.asarray(target["rgb"], dtype=np.uint8),
                np.asarray(donor["rgb"], dtype=np.uint8),
                np.asarray(target["support"], dtype=bool),
                np.asarray(donor["support"], dtype=bool),
                geometry,
            )
            return int(candidate_index), donor, view
        except (cv2.error, RuntimeError, ValueError) as error:
            errors.append(f"{candidate_index}:{error}")
    raise RuntimeError(
        f"donor search exhausted target={target_row.sample_index}: {errors[:5]}"
    )


def _find_matched_same_cross_views(
    *,
    dataset: Dataset,
    semantics: Mapping[str, object],
    rows: Sequence[CleanTrainRow],
    target: Mapping[str, object],
    target_row: CleanTrainRow,
    same_candidates: Sequence[int],
    cross_candidates: Sequence[int],
    epoch: int,
    target_role_id: int,
    same_source_role_id: int,
    cross_source_role_id: int,
) -> Tuple[
    int,
    Dict[str, object],
    PoissonView,
    int,
    Dict[str, object],
    PoissonView,
    int,
]:
    errors = []
    for retry in range(MAX_BLEND_GEOMETRY_ATTEMPTS):
        try:
            geometry = sample_target_geometry(
                np.asarray(target["support"], dtype=bool),
                sample_index=int(target_row.sample_index),
                epoch=int(epoch),
                role_id=int(target_role_id) + 1000 * retry,
            )
            same_index, same_donor, same_view = _find_donor_view(
                dataset,
                semantics,
                rows,
                target,
                target_row,
                geometry,
                same_candidates,
                epoch=epoch,
                source_role_id=int(same_source_role_id) + 10000 * retry,
            )
            cross_index, cross_donor, cross_view = _find_donor_view(
                dataset,
                semantics,
                rows,
                target,
                target_row,
                geometry,
                cross_candidates,
                epoch=epoch,
                source_role_id=int(cross_source_role_id) + 10000 * retry,
            )
            return (
                same_index,
                same_donor,
                same_view,
                cross_index,
                cross_donor,
                cross_view,
                retry,
            )
        except (cv2.error, RuntimeError, ValueError) as error:
            errors.append(f"{retry}:{error}")
    raise RuntimeError(
        "matched same/cross geometry retries exhausted "
        f"target={target_row.sample_index}: {errors[:3]}"
    )


def _find_cross_view_with_geometry_retries(
    *,
    dataset: Dataset,
    semantics: Mapping[str, object],
    rows: Sequence[CleanTrainRow],
    target: Mapping[str, object],
    target_row: CleanTrainRow,
    candidates: Sequence[int],
    epoch: int,
    target_role_id: int,
    source_role_id: int,
) -> Tuple[int, Dict[str, object], PoissonView, int]:
    errors = []
    for retry in range(MAX_BLEND_GEOMETRY_ATTEMPTS):
        try:
            geometry = sample_target_geometry(
                np.asarray(target["support"], dtype=bool),
                sample_index=int(target_row.sample_index),
                epoch=int(epoch),
                role_id=int(target_role_id) + 1000 * retry,
            )
            donor_index, donor, view = _find_donor_view(
                dataset,
                semantics,
                rows,
                target,
                target_row,
                geometry,
                candidates,
                epoch=epoch,
                source_role_id=int(source_role_id) + 10000 * retry,
            )
            return donor_index, donor, view, retry
        except (cv2.error, RuntimeError, ValueError) as error:
            errors.append(f"{retry}:{error}")
    raise RuntimeError(
        "cross geometry retries exhausted "
        f"target={target_row.sample_index}: {errors[:3]}"
    )


def _view_record(
    *,
    role: str,
    target_row: CleanTrainRow,
    donor_row: CleanTrainRow,
    epoch: int,
    view: PoissonView,
    target: Mapping[str, object],
    donor: Mapping[str, object],
    geometry_retry: int,
) -> Dict[str, object]:
    return {
        "role": role,
        "epoch": int(epoch),
        "geometry_retry": int(geometry_retry),
        "target_sample_index": int(target_row.sample_index),
        "target_class": int(target_row.target),
        "target_fold": int(target_row.fold),
        "target_source_stem": target_row.source_stem,
        "donor_sample_index": int(donor_row.sample_index),
        "donor_class": int(donor_row.target),
        "donor_fold": int(donor_row.fold),
        "donor_source_stem": donor_row.source_stem,
        "source_disjoint": target_row.source_stem != donor_row.source_stem,
        "geometry_sha256": geometry_sha256(view.geometry),
        "composite_rgb_sha256": _array_sha256(view.composite_rgb),
        "changed_mask_sha256": _array_sha256(view.changed_mask),
        "intensity_target_sha256": _array_sha256(view.intensity_target),
        "changed_pixels": int(view.changed_mask.sum()),
        "intensity_positive_pixels": int((view.intensity_target > 0).sum()),
        "maximum_outside_target_delta": int(view.maximum_outside_target_delta),
        "target_support": _role_record(target["occluder_exclusion"]),
        "donor_support": _role_record(donor["occluder_exclusion"]),
    }


class PoissonTrainingDataset(Dataset):
    def __init__(
        self,
        *,
        transformed_dataset: Dataset,
        semantics: Mapping[str, object],
        rows: Sequence[CleanTrainRow],
        target_indices: Sequence[int],
        held_fold: int,
        epoch: int,
    ) -> None:
        self.dataset = transformed_dataset
        self.semantics = dict(semantics)
        self.rows = tuple(rows)
        self.target_indices = tuple(int(value) for value in target_indices)
        self.held_fold = int(held_fold)
        self.epoch = int(epoch)
        self.pools = _class_pools(rows, excluded_fold=held_fold)

    def __len__(self) -> int:
        return len(self.target_indices)

    def __getitem__(self, position: int) -> Dict[str, object]:
        sample_index = self.target_indices[int(position)]
        target_row = self.rows[sample_index]
        if int(target_row.fold) == self.held_fold:
            raise ValueError("held row reached Poisson training dataset")
        if sample_index in GEOMETRY_INELIGIBLE_SAMPLE_INDICES:
            raise ValueError("geometry-ineligible row reached Poisson training dataset")
        target = load_processed_sample(self.dataset, self.semantics, sample_index)
        if int(target["label"]) != int(target_row.target):
            raise ValueError("transformed target label differs from CIDT")
        cross_classes = cross_class_schedule(
            int(target_row.target), sample_index, self.epoch
        )
        same_candidates = _circular_candidates(
            self.pools[int(target_row.target)],
            self.rows,
            target_row=target_row,
            epoch=self.epoch,
            role="same",
        )
        cross0_candidates = _circular_candidates(
            self.pools[int(cross_classes[0])],
            self.rows,
            target_row=target_row,
            epoch=self.epoch,
            role=f"cross-0-{cross_classes[0]}",
        )
        (
            same_index,
            same_donor,
            same_view,
            cross0_index,
            cross0_donor,
            cross0_view,
            matched_retry,
        ) = _find_matched_same_cross_views(
            dataset=self.dataset,
            semantics=self.semantics,
            rows=self.rows,
            target=target,
            target_row=target_row,
            same_candidates=same_candidates,
            cross_candidates=cross0_candidates,
            epoch=self.epoch,
            target_role_id=10,
            same_source_role_id=100,
            cross_source_role_id=200,
        )
        cross1_candidates = _circular_candidates(
            self.pools[int(cross_classes[1])],
            self.rows,
            target_row=target_row,
            epoch=self.epoch,
            role=f"cross-1-{cross_classes[1]}",
        )
        (
            cross1_index,
            cross1_donor,
            cross1_view,
            cross1_retry,
        ) = _find_cross_view_with_geometry_retries(
            dataset=self.dataset,
            semantics=self.semantics,
            rows=self.rows,
            target=target,
            target_row=target_row,
            candidates=cross1_candidates,
            epoch=self.epoch,
            target_role_id=11,
            source_role_id=300,
        )
        cross_payload = [
            (cross0_index, cross0_donor, cross0_view, matched_retry),
            (cross1_index, cross1_donor, cross1_view, cross1_retry),
        ]

        support = downsample_support(
            torch.from_numpy(np.asarray(target["support"], dtype=bool))
        )[0]
        cross_targets = torch.stack(
            [
                downsample_target(
                    torch.from_numpy(payload[2].intensity_target)
                )[0]
                for payload in cross_payload
            ],
            dim=0,
        )
        same_record = _view_record(
            role="same",
            target_row=target_row,
            donor_row=self.rows[same_index],
            epoch=self.epoch,
            view=same_view,
            target=target,
            donor=same_donor,
            geometry_retry=matched_retry,
        )
        cross_records = [
            _view_record(
                role=f"cross_{index}",
                target_row=target_row,
                donor_row=self.rows[payload[0]],
                epoch=self.epoch,
                view=payload[2],
                target=target,
                donor=payload[1],
                geometry_retry=payload[3],
            )
            for index, payload in enumerate(cross_payload)
        ]
        return {
            "clean": target["tensor"],
            "same": _normalize_rgb(same_view.composite_rgb, self.semantics),
            "cross": torch.stack(
                [
                    _normalize_rgb(payload[2].composite_rgb, self.semantics)
                    for payload in cross_payload
                ],
                dim=0,
            ),
            "support": support,
            "cross_target": cross_targets,
            "label": torch.tensor(int(target_row.target), dtype=torch.long),
            "sample_index": torch.tensor(sample_index, dtype=torch.long),
            "same_record": json.dumps(same_record, sort_keys=True),
            "cross0_record": json.dumps(cross_records[0], sort_keys=True),
            "cross1_record": json.dumps(cross_records[1], sort_keys=True),
        }


class CleanScoringDataset(Dataset):
    def __init__(
        self,
        *,
        transformed_dataset: Dataset,
        semantics: Mapping[str, object],
        rows: Sequence[CleanTrainRow],
        indices: Sequence[int],
    ) -> None:
        self.dataset = transformed_dataset
        self.semantics = dict(semantics)
        self.rows = tuple(rows)
        self.indices = tuple(int(value) for value in indices)

    def __len__(self) -> int:
        return len(self.indices)

    def __getitem__(self, position: int) -> Dict[str, object]:
        sample_index = self.indices[int(position)]
        row = self.rows[sample_index]
        sample = load_processed_sample(self.dataset, self.semantics, sample_index)
        metadata = sample["metadata"]
        bbox = metadata.get("bbox")
        image_mask = metadata.get("image_mask")
        if not torch.is_tensor(bbox) or bbox.numel() < 4:
            raise ValueError("clean scoring sample lacks bbox")
        if not torch.is_tensor(image_mask):
            raise ValueError("clean scoring sample lacks image_mask")
        support = downsample_support(
            torch.from_numpy(np.asarray(sample["support"], dtype=bool))
        )[0]
        return {
            "image": sample["tensor"],
            "support": support,
            "sample_index": torch.tensor(sample_index, dtype=torch.long),
            "target": torch.tensor(int(row.target), dtype=torch.long),
            "fold": torch.tensor(int(row.fold), dtype=torch.long),
            "bbox": bbox.detach().float().reshape(-1)[:4],
            "valid_fraction": image_mask.detach().float().mean(),
            "support_fraction": torch.tensor(
                float(np.asarray(sample["support"]).mean()), dtype=torch.float32
            ),
        }


def initialize_roles(
    *,
    held_fold: int,
    device: torch.device,
) -> Tuple[Dict[str, ClassQueryMismatchAdapter], Dict[str, object]]:
    state_before = torch.random.get_rng_state().clone()
    torch.manual_seed(SEED + int(held_fold))
    base = ClassQueryMismatchAdapter()
    initial = deepcopy(base.state_dict())
    roles: Dict[str, ClassQueryMismatchAdapter] = {}
    for name in ROLE_NAMES:
        module = ClassQueryMismatchAdapter()
        module.load_state_dict(initial, strict=True)
        roles[name] = module.to(device=device, dtype=torch.float32)
    torch.random.set_rng_state(state_before)
    count, counts = validate_role_parameter_parity(roles)
    hashes = {name: state_dict_sha256(module) for name, module in roles.items()}
    if len(set(hashes.values())) != 1:
        raise RuntimeError("adapter role initial states differ")
    return roles, {
        "seed": SEED + int(held_fold),
        "parameter_count": count,
        "role_parameter_counts": counts,
        "initial_state_sha256": next(iter(hashes.values())),
    }


def _stem_forward(model: nn.Module, images: Tensor) -> Tensor:
    with torch.no_grad():
        output = model.stem(images)  # type: ignore[attr-defined]
    if not torch.is_tensor(output) or tuple(output.shape[1:]) != (256, 32, 32):
        raise ValueError(f"keeper stem shape differs from lock: {tuple(output.shape)}")
    if not bool(torch.isfinite(output).all()):
        raise ValueError("keeper stem emitted non-finite features")
    return output.detach().float()


def _clean_query_loss(
    module: ClassQueryMismatchAdapter,
    clean_stem: Tensor,
    support: Tensor,
    labels: Tensor,
    *,
    permuted: bool,
    sample_indices: Tensor,
    epoch: int,
) -> Tensor:
    batch = int(clean_stem.shape[0])
    queries = torch.arange(5, device=clean_stem.device).repeat(batch)
    expanded_stem = (
        clean_stem[:, None]
        .expand(-1, 5, -1, -1, -1)
        .reshape(batch * 5, 256, 32, 32)
    )
    expanded_support = (
        support[:, None].expand(-1, 5, -1, -1, -1).reshape(batch * 5, 1, 32, 32)
    )
    expanded_indices = sample_indices[:, None].expand(-1, 5).reshape(-1)
    original_queries = queries
    if permuted:
        queries = deterministic_deranged_queries(
            original_queries,
            sample_indices=expanded_indices,
            epoch=epoch,
            namespace=800,
        )
    targets = (
        original_queries.reshape(batch, 5)
        != labels.reshape(batch, 1)
    ).float().reshape(batch * 5, 1, 1, 1) * expanded_support.float()
    losses = masked_bce_per_sample(
        module(expanded_stem, queries),
        targets,
        expanded_support,
    ).reshape(batch, 5)
    true_mask = (
        torch.arange(5, device=labels.device).reshape(1, 5)
        == labels.reshape(-1, 1)
    )
    true_loss = losses[true_mask]
    false_loss = losses[~true_mask].reshape(batch, 4).mean(dim=1)
    return (0.5 * true_loss + 0.5 * false_loss).mean()


def _view_loss(
    module: ClassQueryMismatchAdapter,
    stem: Tensor,
    target: Tensor,
    support: Tensor,
    query: Tensor,
) -> Tensor:
    return masked_bce_per_sample(
        module(stem, query),
        target,
        support,
    ).mean()


def _batch_losses(
    *,
    roles: Mapping[str, ClassQueryMismatchAdapter],
    stem_views: Tensor,
    batch: Mapping[str, object],
    device: torch.device,
    epoch: int,
) -> Dict[str, Tensor]:
    batch_size = int(batch["label"].shape[0])  # type: ignore[index,union-attr]
    clean_stem = stem_views[:batch_size]
    same_stem = stem_views[batch_size : 2 * batch_size]
    cross0_stem = stem_views[2 * batch_size : 3 * batch_size]
    cross1_stem = stem_views[3 * batch_size : 4 * batch_size]
    labels = batch["label"].to(device=device, dtype=torch.long)  # type: ignore[union-attr]
    sample_indices = batch["sample_index"].to(device=device, dtype=torch.long)  # type: ignore[union-attr]
    support = batch["support"].to(device=device, dtype=torch.bool)  # type: ignore[union-attr]
    cross_target = batch["cross_target"].to(device=device, dtype=torch.float32)  # type: ignore[union-attr]
    zero_target = torch.zeros_like(support, dtype=torch.float32)

    candidate = roles["poisson_query_candidate"]
    candidate_clean = _clean_query_loss(
        candidate,
        clean_stem,
        support,
        labels,
        permuted=False,
        sample_indices=sample_indices,
        epoch=epoch,
    )
    candidate_same = _view_loss(
        candidate, same_stem, zero_target, support, labels
    )
    candidate_cross = 0.5 * (
        _view_loss(candidate, cross0_stem, cross_target[:, 0], support, labels)
        + _view_loss(candidate, cross1_stem, cross_target[:, 1], support, labels)
    )

    clean_control = roles["clean_query_control"]
    clean_only_loss = _clean_query_loss(
        clean_control,
        clean_stem,
        support,
        labels,
        permuted=False,
        sample_indices=sample_indices,
        epoch=epoch,
    )

    no_query = roles["no_query_control"]
    constant_query = torch.zeros_like(labels)
    no_query_same = _view_loss(
        no_query, same_stem, zero_target, support, constant_query
    )
    no_query_cross = 0.5 * (
        _view_loss(
            no_query,
            cross0_stem,
            cross_target[:, 0],
            support,
            constant_query,
        )
        + _view_loss(
            no_query,
            cross1_stem,
            cross_target[:, 1],
            support,
            constant_query,
        )
    )

    permuted = roles["permuted_query_control"]
    permuted_clean = _clean_query_loss(
        permuted,
        clean_stem,
        support,
        labels,
        permuted=True,
        sample_indices=sample_indices,
        epoch=epoch,
    )
    permuted_same_query = deterministic_deranged_queries(
        labels,
        sample_indices=sample_indices,
        epoch=epoch,
        namespace=801,
    )
    permuted_cross0_query = deterministic_deranged_queries(
        labels,
        sample_indices=sample_indices,
        epoch=epoch,
        namespace=802,
    )
    permuted_cross1_query = deterministic_deranged_queries(
        labels,
        sample_indices=sample_indices,
        epoch=epoch,
        namespace=803,
    )
    permuted_same = _view_loss(
        permuted,
        same_stem,
        zero_target,
        support,
        permuted_same_query,
    )
    permuted_cross = 0.5 * (
        _view_loss(
            permuted,
            cross0_stem,
            cross_target[:, 0],
            support,
            permuted_cross0_query,
        )
        + _view_loss(
            permuted,
            cross1_stem,
            cross_target[:, 1],
            support,
            permuted_cross1_query,
        )
    )
    return {
        "poisson_query_candidate": (
            candidate_clean + candidate_same + candidate_cross
        )
        / 3.0,
        "clean_query_control": clean_only_loss,
        "no_query_control": (no_query_same + no_query_cross) / 2.0,
        "permuted_query_control": (
            permuted_clean + permuted_same + permuted_cross
        )
        / 3.0,
    }


def train_fold(
    *,
    model: nn.Module,
    transformed_dataset: Dataset,
    semantics: Mapping[str, object],
    rows: Sequence[CleanTrainRow],
    held_fold: int,
    device: torch.device,
    batch_size: int,
    num_workers: int,
    output_dir: Path,
) -> Tuple[Dict[str, ClassQueryMismatchAdapter], Dict[str, object]]:
    panel, panel_summary = build_balanced_fit_panel(rows, held_fold=held_fold)
    roles, initialization = initialize_roles(held_fold=held_fold, device=device)
    initial_state = {
        name: {
            key: value.detach().cpu().clone()
            for key, value in module.state_dict().items()
        }
        for name, module in roles.items()
    }
    optimizers = {
        name: torch.optim.AdamW(
            roles[name].parameters(),
            lr=LEARNING_RATE,
            weight_decay=WEIGHT_DECAY,
        )
        for name in ROLE_NAMES
    }
    for module in roles.values():
        module.train()
    losses: List[Dict[str, object]] = []
    geometry_records: List[Dict[str, object]] = []
    gradient_sums = {
        name: 0.0 for name in ROLE_NAMES
    }
    query_gradient_sums = np.zeros(5, dtype=np.float64)
    loader_summaries: List[Dict[str, object]] = []
    started = time.perf_counter()
    steps = 0
    for epoch in range(EPOCHS):
        dataset = PoissonTrainingDataset(
            transformed_dataset=transformed_dataset,
            semantics=semantics,
            rows=rows,
            target_indices=panel,
            held_fold=held_fold,
            epoch=epoch,
        )
        kwargs, loader_summary = build_safe_dataloader_kwargs(
            requested_num_workers=int(num_workers),
            requested_pin_memory=True,
            context=f"nsa_poisson_train_fold_{held_fold}_epoch_{epoch}",
            prefetch_factor=2,
            persistent_workers=True,
        )
        loader_summaries.append(dict(loader_summary))
        loader_summaries[-1]["opencv_worker_threads"] = 1
        loader = DataLoader(
            dataset,
            batch_size=int(batch_size),
            shuffle=False,
            drop_last=False,
            generator=torch.Generator().manual_seed(SEED + held_fold * 10 + epoch),
            worker_init_fn=_worker_init,
            **kwargs,
        )
        for batch_index, batch in enumerate(loader):
            clean = batch["clean"].to(device=device, dtype=torch.float32, non_blocking=True)
            same = batch["same"].to(device=device, dtype=torch.float32, non_blocking=True)
            cross = batch["cross"].to(device=device, dtype=torch.float32, non_blocking=True)
            views = torch.cat((clean, same, cross[:, 0], cross[:, 1]), dim=0)
            stem_views = _stem_forward(model, views)
            batch_losses = _batch_losses(
                roles=roles,
                stem_views=stem_views,
                batch=batch,
                device=device,
                epoch=epoch,
            )
            step_record: Dict[str, object] = {
                "held_fold": held_fold,
                "epoch": epoch,
                "batch_index": batch_index,
                "batch_size": int(clean.shape[0]),
            }
            for name in ROLE_NAMES:
                optimizer = optimizers[name]
                optimizer.zero_grad(set_to_none=True)
                loss = batch_losses[name]
                if not bool(torch.isfinite(loss)):
                    raise FloatingPointError(f"non-finite {name} loss")
                loss.backward()
                if name == "poisson_query_candidate":
                    gradient = roles[name].query_embedding.weight.grad
                    if gradient is None:
                        raise RuntimeError("candidate query embedding has no gradient")
                    query_gradient_sums += (
                        gradient.detach().float().norm(dim=1).cpu().numpy()
                    )
                gradient_norm = torch.nn.utils.clip_grad_norm_(
                    roles[name].parameters(),
                    max_norm=GRADIENT_CLIP,
                )
                if not bool(torch.isfinite(gradient_norm)):
                    raise FloatingPointError(f"non-finite {name} gradient")
                optimizer.step()
                gradient_sums[name] += float(gradient_norm)
                step_record[f"{name}_loss"] = float(loss.detach().cpu())
                step_record[f"{name}_gradient_norm"] = float(gradient_norm)
            losses.append(step_record)
            for key in ("same_record", "cross0_record", "cross1_record"):
                geometry_records.extend(json.loads(value) for value in batch[key])
            steps += 1
            if steps % 25 == 0:
                print(
                    json.dumps(
                        {
                            "stage": "nsa_adapter_train",
                            "held_fold": held_fold,
                            "epoch": epoch,
                            "step": steps,
                            "elapsed_seconds": time.perf_counter() - started,
                            "losses": {
                                name: float(batch_losses[name].detach().cpu())
                                for name in ROLE_NAMES
                            },
                        }
                    ),
                    flush=True,
                )
    final_hashes = {name: state_dict_sha256(module) for name, module in roles.items()}
    update_norms = {}
    candidate_query_updates = None
    fold_dir = output_dir / f"fold_{held_fold}"
    fold_dir.mkdir(parents=True, exist_ok=True)
    for name, module in roles.items():
        torch.save(
            {
                "state_dict": module.state_dict(),
                "held_fold": held_fold,
                "role": name,
                "seed": SEED + held_fold,
            },
            fold_dir / f"{name}.pt",
        )
        module.eval()
        squared_update = torch.zeros((), dtype=torch.float64)
        for key, value in module.state_dict().items():
            delta = value.detach().cpu().double() - initial_state[name][key].double()
            squared_update += delta.square().sum()
        update_norms[name] = float(squared_update.sqrt())
        if name == "poisson_query_candidate":
            query_delta = (
                module.query_embedding.weight.detach().cpu().double()
                - initial_state[name]["query_embedding.weight"].double()
            )
            candidate_query_updates = query_delta.norm(dim=1).tolist()
    source_violations = sum(
        not bool(record["source_disjoint"]) for record in geometry_records
    )
    fold_violations = sum(
        int(record["target_fold"]) == held_fold
        or int(record["donor_fold"]) == held_fold
        for record in geometry_records
    )
    cross_coverage = {}
    for sample_index in panel:
        labels = [
            int(record["donor_class"])
            for record in geometry_records
            if int(record["target_sample_index"]) == sample_index
            and str(record["role"]).startswith("cross")
        ]
        cross_coverage[sample_index] = sorted(labels)
    cross_exact = all(
        labels == sorted(value for value in range(5) if value != rows[index].target)
        for index, labels in cross_coverage.items()
    )
    return roles, {
        "held_fold": held_fold,
        "panel": panel_summary,
        "initialization": initialization,
        "steps": steps,
        "expected_steps": EPOCHS * math.ceil(len(panel) / int(batch_size)),
        "optimizer_order": list(ROLE_NAMES),
        "losses": losses,
        "gradient_norm_sums": gradient_sums,
        "candidate_query_gradient_norm_sums": query_gradient_sums.tolist(),
        "all_candidate_query_gradients_nonzero": bool((query_gradient_sums > 0).all()),
        "candidate_query_update_l2_norms": candidate_query_updates,
        "all_candidate_query_updates_nonzero": bool(
            candidate_query_updates is not None
            and all(float(value) > 0.0 for value in candidate_query_updates)
        ),
        "final_state_sha256": final_hashes,
        "parameter_update_l2_norms": update_norms,
        "geometry_records": geometry_records,
        "geometry_record_count": len(geometry_records),
        "source_disjoint_violations": source_violations,
        "held_fold_geometry_violations": fold_violations,
        "cross_class_coverage_exact": cross_exact,
        "loader": loader_summaries,
        "elapsed_seconds": float(time.perf_counter() - started),
    }


def score_clean_cohort(
    *,
    model: nn.Module,
    roles: Mapping[str, ClassQueryMismatchAdapter],
    transformed_dataset: Dataset,
    semantics: Mapping[str, object],
    rows: Sequence[CleanTrainRow],
    cohort_indices: Sequence[int],
    device: torch.device,
    batch_size: int,
    num_workers: int,
) -> Dict[str, object]:
    dataset = CleanScoringDataset(
        transformed_dataset=transformed_dataset,
        semantics=semantics,
        rows=rows,
        indices=cohort_indices,
    )
    kwargs, loader_summary = build_safe_dataloader_kwargs(
        requested_num_workers=int(num_workers),
        requested_pin_memory=True,
        context="nsa_poisson_clean_score",
        prefetch_factor=2,
        persistent_workers=True,
    )
    loader = DataLoader(
        dataset,
        batch_size=int(batch_size),
        shuffle=False,
        drop_last=False,
        generator=torch.Generator().manual_seed(SEED),
        worker_init_fn=_worker_init,
        **kwargs,
    )
    loader_summary = {**loader_summary, "opencv_worker_threads": 1}
    role_summaries = {
        name: {
            "support_mean": [],
            "top_quintile_mean": [],
            "percentile_90": [],
            "compatibility": [],
        }
        for name in ROLE_NAMES
    }
    query_compatibility: List[np.ndarray] = []
    candidate_maps: List[np.ndarray] = []
    metadata = {
        "sample_index": [],
        "target": [],
        "fold": [],
        "bbox": [],
        "valid_fraction": [],
        "support_fraction": [],
    }
    for module in roles.values():
        module.eval()
    with torch.inference_mode():
        for batch in loader:
            images = batch["image"].to(
                device=device, dtype=torch.float32, non_blocking=True
            )
            support = batch["support"].to(
                device=device, dtype=torch.bool, non_blocking=True
            )
            stem = _stem_forward(model, images)
            query_one = torch.ones(int(images.shape[0]), device=device, dtype=torch.long)
            for name in ROLE_NAMES:
                role_query = (
                    torch.zeros_like(query_one)
                    if name == "no_query_control"
                    else query_one
                )
                probabilities = torch.sigmoid(roles[name](stem, role_query))
                summaries = map_summaries(probabilities, support)
                for key in role_summaries[name]:
                    role_summaries[name][key].append(
                        summaries[key].detach().cpu().numpy()
                    )
                if name == "poisson_query_candidate":
                    candidate_maps.append(
                        (probabilities * support.float())
                        .detach()
                        .cpu()
                        .numpy()
                        .astype(np.float32)
                    )
            batch_rows = int(images.shape[0])
            expanded_stem = (
                stem[:, None]
                .expand(-1, 5, -1, -1, -1)
                .reshape(batch_rows * 5, 256, 32, 32)
            )
            expanded_support = (
                support[:, None]
                .expand(-1, 5, -1, -1, -1)
                .reshape(batch_rows * 5, 1, 32, 32)
            )
            queries = torch.arange(5, device=device).repeat(batch_rows)
            probabilities = torch.sigmoid(
                roles["poisson_query_candidate"](expanded_stem, queries)
            )
            compat = map_summaries(
                probabilities, expanded_support
            )["compatibility"].reshape(batch_rows, 5)
            query_compatibility.append(compat.detach().cpu().numpy())
            for key in ("sample_index", "target", "fold", "bbox", "valid_fraction", "support_fraction"):
                metadata[key].append(batch[key].detach().cpu().numpy())
    output_roles = {
        name: {
            key: np.concatenate(values, axis=0)
            for key, values in summaries.items()
        }
        for name, summaries in role_summaries.items()
    }
    output_metadata = {
        key: np.concatenate(values, axis=0) for key, values in metadata.items()
    }
    if output_metadata["sample_index"].tolist() != [int(value) for value in cohort_indices]:
        raise ValueError("clean scoring changed cohort order")
    return {
        "roles": output_roles,
        "query_compatibility": np.concatenate(query_compatibility, axis=0),
        "candidate_maps": np.concatenate(candidate_maps, axis=0),
        "metadata": output_metadata,
        "loader": dict(loader_summary),
    }


def base_context_features(
    cohort: Sequence[CleanTrainRow],
    metadata: Mapping[str, np.ndarray],
) -> np.ndarray:
    probabilities = np.asarray(
        [row.keeper_probabilities for row in cohort], dtype=np.float64
    )
    log_probabilities = np.log(np.clip(probabilities, 1e-8, 1.0))
    margin = (
        log_probabilities[:, 1]
        - log_probabilities[:, [0, 2, 4]].max(axis=1)
    )[:, None]
    bbox = np.asarray(metadata["bbox"], dtype=np.float64)
    if bbox.shape != (len(cohort), 4):
        raise ValueError("bbox context shape differs from cohort")
    width = np.clip(bbox[:, 2], 1e-8, None)
    height = np.clip(bbox[:, 3], 1e-8, None)
    geometry = np.stack(
        (
            np.log(width * height),
            np.log(width / height),
            np.asarray(metadata["valid_fraction"], dtype=np.float64),
            np.asarray(metadata["support_fraction"], dtype=np.float64),
        ),
        axis=1,
    )
    return np.concatenate((log_probabilities, margin, bbox, geometry), axis=1)


def _positive_threshold(scores: np.ndarray, labels: np.ndarray) -> float:
    positives = np.sort(
        np.asarray(scores, dtype=np.float64)[np.asarray(labels, dtype=np.int64) == 1]
    )
    if positives.size == 0:
        raise ValueError("readout fit has no positive rows")
    allowed_breaks = int(math.floor((1.0 - FIT_TP_RETENTION) * positives.size))
    threshold = float(positives[min(allowed_breaks, positives.size - 1)])
    if float((positives >= threshold).mean()) + 1e-12 < FIT_TP_RETENTION:
        raise RuntimeError("readout threshold violates fit TP retention")
    return threshold


def _positive_probability(model: LogisticRegression, features: np.ndarray) -> np.ndarray:
    classes = np.asarray(model.classes_, dtype=np.int64)
    location = np.flatnonzero(classes == 1)
    if location.size != 1:
        raise ValueError("readout lacks exactly one positive class")
    return model.predict_proba(features)[:, int(location[0])]


def fit_source_held_readouts(
    *,
    cohort: Sequence[CleanTrainRow],
    fold_scores: Mapping[int, Mapping[str, object]],
) -> Dict[str, object]:
    labels = np.asarray([int(row.target == 1) for row in cohort], dtype=np.int64)
    folds = np.asarray([int(row.fold) for row in cohort], dtype=np.int64)
    sources = np.asarray([row.source_stem for row in cohort], dtype=object)
    role_scores = {
        role: np.full(len(cohort), np.nan, dtype=np.float64)
        for role in READOUT_ROLES
    }
    role_actions = {
        role: np.full(len(cohort), False, dtype=bool)
        for role in READOUT_ROLES
    }
    states: List[Dict[str, object]] = []
    for held_fold in FOLDS:
        fold_output = fold_scores[int(held_fold)]
        metadata = fold_output["metadata"]
        base = base_context_features(cohort, metadata)
        fit_mask = folds != held_fold
        held_mask = folds == held_fold
        if set(sources[fit_mask]).intersection(set(sources[held_mask])):
            raise ValueError("readout fit/held source overlap")
        role_features = {"base_context": base}
        for role in ROLE_NAMES:
            summaries = fold_output["roles"][role]
            map_features = np.stack(
                (
                    summaries["support_mean"],
                    summaries["top_quintile_mean"],
                    summaries["percentile_90"],
                ),
                axis=1,
            )
            role_features[role] = np.concatenate((base, map_features), axis=1)
        for role in READOUT_ROLES:
            scaler = StandardScaler()
            fit_features = scaler.fit_transform(role_features[role][fit_mask])
            held_features = scaler.transform(role_features[role][held_mask])
            model = LogisticRegression(
                C=0.1,
                class_weight="balanced",
                solver="lbfgs",
                max_iter=4000,
                tol=1e-9,
                random_state=SEED,
            )
            model.fit(fit_features, labels[fit_mask])
            fit_probabilities = _positive_probability(model, fit_features)
            held_probabilities = _positive_probability(model, held_features)
            threshold = _positive_threshold(fit_probabilities, labels[fit_mask])
            role_scores[role][held_mask] = held_probabilities
            role_actions[role][held_mask] = held_probabilities >= threshold
            states.append(
                {
                    "held_fold": held_fold,
                    "role": role,
                    "threshold": threshold,
                    "feature_dim": int(role_features[role].shape[1]),
                    "scaler_mean": scaler.mean_.tolist(),
                    "scaler_scale": scaler.scale_.tolist(),
                    "coefficient": model.coef_.reshape(-1).tolist(),
                    "intercept": model.intercept_.reshape(-1).tolist(),
                    "classes": model.classes_.tolist(),
                    "iterations": model.n_iter_.tolist(),
                    "fit_rows": int(fit_mask.sum()),
                    "held_rows": int(held_mask.sum()),
                    "fit_tp_retention": float(
                        (
                            fit_probabilities[labels[fit_mask] == 1]
                            >= threshold
                        ).mean()
                    ),
                }
            )
    if any(not np.isfinite(values).all() for values in role_scores.values()):
        raise ValueError("OOF readout scores are incomplete")
    return {
        "labels": labels,
        "folds": folds,
        "scores": role_scores,
        "actions": role_actions,
        "states": states,
    }


def apply_saved_readouts(
    *,
    cohort: Sequence[CleanTrainRow],
    fold_scores: Mapping[int, Mapping[str, object]],
    states: Sequence[Mapping[str, object]],
) -> Dict[str, object]:
    labels = np.asarray([int(row.target == 1) for row in cohort], dtype=np.int64)
    folds = np.asarray([int(row.fold) for row in cohort], dtype=np.int64)
    role_scores = {
        role: np.full(len(cohort), np.nan, dtype=np.float64)
        for role in READOUT_ROLES
    }
    role_actions = {
        role: np.full(len(cohort), False, dtype=bool)
        for role in READOUT_ROLES
    }
    keyed_states = {
        (int(state["held_fold"]), str(state["role"])): state for state in states
    }
    expected_keys = {
        (held_fold, role) for held_fold in FOLDS for role in READOUT_ROLES
    }
    if set(keyed_states) != expected_keys:
        raise ValueError("saved readout states differ from fold/role lock")
    for held_fold in FOLDS:
        fold_output = fold_scores[int(held_fold)]
        base = base_context_features(cohort, fold_output["metadata"])
        held_mask = folds == held_fold
        role_features = {"base_context": base}
        for role in ROLE_NAMES:
            summaries = fold_output["roles"][role]
            map_features = np.stack(
                (
                    summaries["support_mean"],
                    summaries["top_quintile_mean"],
                    summaries["percentile_90"],
                ),
                axis=1,
            )
            role_features[role] = np.concatenate((base, map_features), axis=1)
        for role in READOUT_ROLES:
            state = keyed_states[(held_fold, role)]
            features = np.asarray(role_features[role][held_mask], dtype=np.float64)
            mean = np.asarray(state["scaler_mean"], dtype=np.float64)
            scale = np.asarray(state["scaler_scale"], dtype=np.float64)
            coefficient = np.asarray(state["coefficient"], dtype=np.float64)
            intercept = float(np.asarray(state["intercept"], dtype=np.float64)[0])
            if (
                features.shape[1] != mean.size
                or mean.shape != scale.shape
                or mean.shape != coefficient.shape
                or bool((scale <= 0).any())
            ):
                raise ValueError(
                    f"saved readout shape differs for fold={held_fold} role={role}"
                )
            standardized = (features - mean) / scale
            logits = standardized @ coefficient + intercept
            probabilities = np.empty_like(logits, dtype=np.float64)
            nonnegative = logits >= 0
            probabilities[nonnegative] = 1.0 / (
                1.0 + np.exp(-logits[nonnegative])
            )
            exponent = np.exp(logits[~nonnegative])
            probabilities[~nonnegative] = exponent / (1.0 + exponent)
            threshold = float(state["threshold"])
            role_scores[role][held_mask] = probabilities
            role_actions[role][held_mask] = probabilities >= threshold
    if any(not np.isfinite(values).all() for values in role_scores.values()):
        raise ValueError("saved readout replay scores are incomplete")
    return {
        "labels": labels,
        "folds": folds,
        "scores": role_scores,
        "actions": role_actions,
        "states": [dict(state) for state in states],
    }


def role_metrics(
    *,
    cohort: Sequence[CleanTrainRow],
    labels: np.ndarray,
    folds: np.ndarray,
    scores: np.ndarray,
    actions: np.ndarray,
) -> Dict[str, object]:
    positive = labels == 1
    negative = ~positive
    alternate = np.asarray(
        [
            int(
                max(
                    (index for index in range(5) if index != 1),
                    key=lambda index: row.keeper_probabilities[index],
                )
            )
            for row in cohort
        ],
        dtype=np.int64,
    )
    targets = np.asarray([row.target for row in cohort], dtype=np.int64)
    prediction = np.where(actions, 1, alternate)
    fold_metrics = []
    for fold in FOLDS:
        mask = folds == fold
        fold_metrics.append(
            {
                "fold": fold,
                "support": int(mask.sum()),
                "auroc": float(roc_auc_score(labels[mask], scores[mask])),
                "tp_retention": float(actions[mask & positive].mean()),
                "fp_rejection": float((~actions[mask & negative]).mean()),
            }
        )
    target_rejection = {
        str(target): float(
            (~actions[negative & (targets == target)]).mean()
            if bool((negative & (targets == target)).any())
            else float("nan")
        )
        for target in (0, 2, 4)
    }
    return {
        "support": len(cohort),
        "auroc": float(roc_auc_score(labels, scores)),
        "tp_retention": float(actions[positive].mean()),
        "fp_rejection": float((~actions[negative]).mean()),
        "tp_broken": int((~actions[positive]).sum()),
        "fp_rejected": int((~actions[negative]).sum()),
        "fp_corrected_to_target": int((prediction[negative] == targets[negative]).sum()),
        "corrections": int((prediction[negative] == targets[negative]).sum()),
        "harms": int((prediction[positive] != targets[positive]).sum()),
        "target_rejection": target_rejection,
        "folds": fold_metrics,
    }


def geometry_records_sha256(records: Sequence[Mapping[str, object]]) -> str:
    digest = hashlib.sha256()
    for record in records:
        digest.update(
            json.dumps(
                dict(record),
                sort_keys=True,
                separators=(",", ":"),
            ).encode("utf-8")
        )
        digest.update(b"\n")
    return digest.hexdigest()


def replay_training_geometry(
    *,
    transformed_dataset: Dataset,
    semantics: Mapping[str, object],
    rows: Sequence[CleanTrainRow],
    held_fold: int,
) -> Dict[str, object]:
    panel, panel_summary = build_balanced_fit_panel(rows, held_fold=held_fold)
    records: List[Dict[str, object]] = []
    for epoch in range(EPOCHS):
        dataset = PoissonTrainingDataset(
            transformed_dataset=transformed_dataset,
            semantics=semantics,
            rows=rows,
            target_indices=panel,
            held_fold=held_fold,
            epoch=epoch,
        )
        for position in range(len(dataset)):
            item = dataset[position]
            for key in ("same_record", "cross0_record", "cross1_record"):
                records.append(json.loads(str(item[key])))
    return {
        "held_fold": int(held_fold),
        "panel": panel_summary,
        "record_count": len(records),
        "records_sha256": geometry_records_sha256(records),
    }


def fixed_visual_indices(cohort: Sequence[CleanTrainRow]) -> List[int]:
    selected: List[int] = []
    for fold in FOLDS:
        for target in (1, 0, 2, 4):
            candidates = [
                row
                for row in cohort
                if int(row.fold) == fold and int(row.target) == target
            ]
            if not candidates:
                continue
            candidates.sort(
                key=lambda row: (
                    _digest(
                        SEED,
                        "fixed-visual",
                        fold,
                        target,
                        row.sample_index,
                        row.source_stem,
                    ),
                    row.sample_index,
                )
            )
            selected.append(int(candidates[0].sample_index))
    if len(selected) != 19 or len(set(selected)) != len(selected):
        raise ValueError("fixed visual panel differs from 19-row lock")
    return selected


def collect_fixed_visual_rows(
    *,
    model: nn.Module,
    roles: Mapping[str, ClassQueryMismatchAdapter],
    transformed_dataset: Dataset,
    semantics: Mapping[str, object],
    rows: Sequence[CleanTrainRow],
    target_indices: Sequence[int],
    held_fold: int,
    device: torch.device,
) -> List[Dict[str, object]]:
    selected = [
        int(index)
        for index in target_indices
        if int(rows[int(index)].fold) == int(held_fold)
    ]
    dataset = HeldDiagnosticDataset(
        transformed_dataset=transformed_dataset,
        semantics=semantics,
        rows=rows,
        target_indices=selected,
        held_fold=held_fold,
    )
    output: List[Dict[str, object]] = []
    for position, sample_index in enumerate(selected):
        item = dataset[position]
        clean = load_processed_sample(
            transformed_dataset, semantics, int(sample_index)
        )
        clean_tensor = clean["tensor"].unsqueeze(0).to(
            device=device, dtype=torch.float32
        )
        support = item["support"].unsqueeze(0).to(
            device=device, dtype=torch.bool
        )
        stem = _stem_forward(model, clean_tensor)
        query_one = torch.ones(1, device=device, dtype=torch.long)
        role_maps = {}
        query_maps = {}
        with torch.inference_mode():
            for role in ROLE_NAMES:
                role_query = (
                    torch.zeros_like(query_one)
                    if role == "no_query_control"
                    else query_one
                )
                role_maps[role] = (
                    torch.sigmoid(roles[role](stem, role_query))
                    * support.float()
                )[0, 0].detach().cpu().numpy().astype(np.float32)
            expanded = stem.expand(5, -1, -1, -1)
            queries = torch.arange(5, device=device, dtype=torch.long)
            maps = torch.sigmoid(
                roles["poisson_query_candidate"](expanded, queries)
            ) * support.expand(5, -1, -1, -1).float()
            for query in range(5):
                query_maps[str(query)] = (
                    maps[query, 0].detach().cpu().numpy().astype(np.float32)
                )
        output.append(
            {
                "sample_index": int(sample_index),
                "target": int(rows[int(sample_index)].target),
                "fold": int(held_fold),
                "clean_rgb": np.asarray(clean["rgb"], dtype=np.uint8),
                "support": np.asarray(clean["support"], dtype=bool),
                "same_rgb": _rgb_uint8(item["same"], semantics),
                "cross_rgb": _rgb_uint8(item["cross"], semantics),
                "cross_target": item["cross_target"][0].numpy().astype(np.float32),
                "role_maps": role_maps,
                "query_maps": query_maps,
                "same_record": json.loads(str(item["same_record"])),
                "cross_record": json.loads(str(item["cross_record"])),
            }
        )
    return output


def query_metrics(
    *,
    cohort: Sequence[CleanTrainRow],
    fold_scores: Mapping[int, Mapping[str, object]],
) -> Dict[str, object]:
    true_better = []
    correct = []
    compatibility_rows = []
    map_rows = []
    for position, row in enumerate(cohort):
        output = fold_scores[int(row.fold)]
        compatibility = np.asarray(output["query_compatibility"][position], dtype=np.float64)
        true_value = float(compatibility[int(row.target)])
        false_mean = float(np.delete(compatibility, int(row.target)).mean())
        true_better.append(true_value > false_mean)
        correct.append(int(np.argmax(compatibility)) == int(row.target))
        compatibility_rows.append(compatibility)
        map_rows.append(np.asarray(output["candidate_maps"][position], dtype=np.float64).reshape(-1))
    maps = np.stack(map_rows, axis=0)
    centered = maps - maps.mean(axis=0, keepdims=True)
    singular = np.linalg.svd(centered, full_matrices=False, compute_uv=False)
    weights = np.square(singular)
    effective_rank = float(
        weights.sum() ** 2
        / np.clip(np.square(weights).sum(), 1e-12, None)
    )
    return {
        "true_query_better_fraction": float(np.mean(true_better)),
        "five_query_accuracy": float(np.mean(correct)),
        "effective_spatial_rank": effective_rank,
        "compatibility_sha256": hashlib.sha256(
            np.asarray(compatibility_rows, dtype=np.float32).tobytes()
        ).hexdigest(),
    }


class HeldDiagnosticDataset(PoissonTrainingDataset):
    def __init__(
        self,
        *,
        transformed_dataset: Dataset,
        semantics: Mapping[str, object],
        rows: Sequence[CleanTrainRow],
        target_indices: Sequence[int],
        held_fold: int,
    ) -> None:
        self.dataset = transformed_dataset
        self.semantics = dict(semantics)
        self.rows = tuple(rows)
        self.target_indices = tuple(int(value) for value in target_indices)
        self.held_fold = int(held_fold)
        self.epoch = 0
        self.pools = _class_pools(rows, allowed_fold=held_fold)

    def __getitem__(self, position: int) -> Dict[str, object]:
        sample_index = self.target_indices[int(position)]
        target_row = self.rows[sample_index]
        if int(target_row.fold) != self.held_fold:
            raise ValueError("diagnostic target is outside held fold")
        if sample_index in GEOMETRY_INELIGIBLE_SAMPLE_INDICES:
            raise ValueError("geometry-ineligible row reached held diagnostics")
        target = load_processed_sample(self.dataset, self.semantics, sample_index)
        same_candidates = _circular_candidates(
            self.pools[int(target_row.target)],
            self.rows,
            target_row=target_row,
            epoch=0,
            role="diagnostic-same",
        )
        cross_class = cross_class_schedule(
            int(target_row.target), sample_index, 0
        )[0]
        cross_candidates = _circular_candidates(
            self.pools[cross_class],
            self.rows,
            target_row=target_row,
            epoch=0,
            role=f"diagnostic-cross-{cross_class}",
        )
        (
            same_index,
            same_donor,
            same_view,
            cross_index,
            cross_donor,
            cross_view,
            geometry_retry,
        ) = _find_matched_same_cross_views(
            dataset=self.dataset,
            semantics=self.semantics,
            rows=self.rows,
            target=target,
            target_row=target_row,
            same_candidates=same_candidates,
            cross_candidates=cross_candidates,
            epoch=0,
            target_role_id=20,
            same_source_role_id=400,
            cross_source_role_id=500,
        )
        support = downsample_support(
            torch.from_numpy(np.asarray(target["support"], dtype=bool))
        )[0]
        target_map = downsample_target(
            torch.from_numpy(cross_view.intensity_target)
        )[0]
        return {
            "same": _normalize_rgb(same_view.composite_rgb, self.semantics),
            "cross": _normalize_rgb(cross_view.composite_rgb, self.semantics),
            "support": support,
            "cross_target": target_map,
            "query": torch.tensor(int(target_row.target), dtype=torch.long),
            "sample_index": torch.tensor(sample_index, dtype=torch.long),
            "same_record": json.dumps(
                _view_record(
                    role="diagnostic_same",
                    target_row=target_row,
                    donor_row=self.rows[same_index],
                    epoch=0,
                    view=same_view,
                    target=target,
                    donor=same_donor,
                    geometry_retry=geometry_retry,
                ),
                sort_keys=True,
            ),
            "cross_record": json.dumps(
                _view_record(
                    role="diagnostic_cross",
                    target_row=target_row,
                    donor_row=self.rows[cross_index],
                    epoch=0,
                    view=cross_view,
                    target=target,
                    donor=cross_donor,
                    geometry_retry=geometry_retry,
                ),
                sort_keys=True,
            ),
        }


def diagnostic_indices(
    rows: Sequence[CleanTrainRow],
    held_fold: int,
) -> List[int]:
    selected = []
    for target in range(5):
        values = [
            row
            for row in rows
            if row.fold == held_fold
            and row.target == target
            and row.sample_index not in GEOMETRY_INELIGIBLE_SAMPLE_INDICES
        ]
        values.sort(
            key=lambda row: (
                _digest(
                    SEED,
                    "diagnostic",
                    held_fold,
                    target,
                    row.sample_index,
                    row.source_stem,
                ),
                row.sample_index,
            )
        )
        selected.extend(int(row.sample_index) for row in values[:DIAGNOSTIC_PER_CLASS])
    if len(selected) != 5 * DIAGNOSTIC_PER_CLASS or len(set(selected)) != len(selected):
        raise ValueError("held diagnostic selection differs from lock")
    return selected


def score_held_diagnostics(
    *,
    model: nn.Module,
    candidate: ClassQueryMismatchAdapter,
    transformed_dataset: Dataset,
    semantics: Mapping[str, object],
    rows: Sequence[CleanTrainRow],
    held_fold: int,
    device: torch.device,
    batch_size: int,
    num_workers: int,
) -> Dict[str, object]:
    indices = diagnostic_indices(rows, held_fold)
    dataset = HeldDiagnosticDataset(
        transformed_dataset=transformed_dataset,
        semantics=semantics,
        rows=rows,
        target_indices=indices,
        held_fold=held_fold,
    )
    kwargs, loader_summary = build_safe_dataloader_kwargs(
        requested_num_workers=int(num_workers),
        requested_pin_memory=True,
        context=f"nsa_poisson_held_diagnostic_{held_fold}",
        prefetch_factor=2,
        persistent_workers=True,
    )
    loader = DataLoader(
        dataset,
        batch_size=int(batch_size),
        shuffle=False,
        drop_last=False,
        generator=torch.Generator().manual_seed(SEED + held_fold),
        worker_init_fn=_worker_init,
        **kwargs,
    )
    loader_summary = {**loader_summary, "opencv_worker_threads": 1}
    cross_inside_mass = 0.0
    cross_total_mass = 0.0
    same_positive_pixels = 0
    same_support_pixels = 0
    records = []
    candidate.eval()
    with torch.inference_mode():
        for batch in loader:
            same = batch["same"].to(device=device, dtype=torch.float32, non_blocking=True)
            cross = batch["cross"].to(device=device, dtype=torch.float32, non_blocking=True)
            support = batch["support"].to(device=device, dtype=torch.bool, non_blocking=True)
            cross_target = batch["cross_target"].to(
                device=device, dtype=torch.float32, non_blocking=True
            )
            query = batch["query"].to(device=device, dtype=torch.long, non_blocking=True)
            stem = _stem_forward(model, torch.cat((same, cross), dim=0))
            batch_size_actual = int(same.shape[0])
            same_probability = torch.sigmoid(candidate(stem[:batch_size_actual], query))
            cross_probability = torch.sigmoid(candidate(stem[batch_size_actual:], query))
            positive_mask = (cross_target > 0) & support
            cross_inside_mass += float((cross_probability * positive_mask.float()).sum())
            cross_total_mass += float((cross_probability * support.float()).sum())
            same_positive_pixels += int(((same_probability >= 0.5) & support).sum())
            same_support_pixels += int(support.sum())
            records.extend(json.loads(value) for value in batch["same_record"])
            records.extend(json.loads(value) for value in batch["cross_record"])
    return {
        "held_fold": held_fold,
        "indices": indices,
        "rows": len(indices),
        "cross_map_mass_inside_intensity": float(
            cross_inside_mass / max(cross_total_mass, 1e-12)
        ),
        "same_support_pixel_mismatch_rate": float(
            same_positive_pixels / max(same_support_pixels, 1)
        ),
        "cross_inside_mass": cross_inside_mass,
        "cross_total_mass": cross_total_mass,
        "same_positive_pixels": same_positive_pixels,
        "same_support_pixels": same_support_pixels,
        "records": records,
        "loader": dict(loader_summary),
    }


def assess_mechanism_gate(
    *,
    metrics: Mapping[str, Mapping[str, object]],
    query: Mapping[str, object],
    diagnostics: Mapping[str, object],
    candidate_query_gradients_nonzero: bool,
    candidate_query_updates_nonzero: bool,
) -> Dict[str, object]:
    candidate = metrics["poisson_query_candidate"]
    base = metrics["base_context"]
    clean = metrics["clean_query_control"]
    no_query = metrics["no_query_control"]
    permuted = metrics["permuted_query_control"]
    diagnostic_mass = float(diagnostics["cross_map_mass_inside_intensity"])
    diagnostic_same = float(diagnostics["same_support_pixel_mismatch_rate"])
    candidate_folds = {int(value["fold"]): value for value in candidate["folds"]}
    base_folds = {int(value["fold"]): value for value in base["folds"]}
    clean_folds = {int(value["fold"]): value for value in clean["folds"]}
    checks = {
        "candidate_auroc_at_least_0p855": float(candidate["auroc"]) >= 0.855,
        "auroc_gain_over_base_at_least_0p020": float(candidate["auroc"])
        - float(base["auroc"])
        >= 0.020,
        "auroc_gain_over_clean_at_least_0p010": float(candidate["auroc"])
        - float(clean["auroc"])
        >= 0.010,
        "auroc_gain_over_no_query_at_least_0p020": float(candidate["auroc"])
        - float(no_query["auroc"])
        >= 0.020,
        "auroc_gain_over_permuted_at_least_0p020": float(candidate["auroc"])
        - float(permuted["auroc"])
        >= 0.020,
        "candidate_tp_retention_at_least_0p98": float(candidate["tp_retention"])
        >= 0.98,
        "candidate_fp_rejection_at_least_0p12": float(candidate["fp_rejection"])
        >= 0.12,
        "five_more_fp_than_clean": int(candidate["fp_rejected"])
        >= int(clean["fp_rejected"]) + 5,
        "corrections_at_least_three_times_harms": int(candidate["corrections"])
        >= 3 * int(candidate["harms"]),
        "four_of_five_folds_beat_base_and_clean": sum(
            float(candidate_folds[fold]["auroc"]) > float(base_folds[fold]["auroc"])
            and float(candidate_folds[fold]["auroc"]) > float(clean_folds[fold]["auroc"])
            for fold in FOLDS
        )
        >= 4,
        "every_fold_tp_retention_at_least_0p95": all(
            float(candidate_folds[fold]["tp_retention"]) >= 0.95 for fold in FOLDS
        ),
        "target_0_rejection_at_least_0p08": float(
            candidate["target_rejection"]["0"]
        )
        >= 0.08,
        "target_2_rejection_at_least_0p08": float(
            candidate["target_rejection"]["2"]
        )
        >= 0.08,
        "at_least_one_target_4_rejected": float(
            candidate["target_rejection"]["4"]
        )
        > 0.0,
        "true_query_better_fraction_at_least_0p70": float(
            query["true_query_better_fraction"]
        )
        >= 0.70,
        "five_query_accuracy_at_least_0p45": float(query["five_query_accuracy"])
        >= 0.45,
        "all_candidate_query_gradients_nonzero": bool(
            candidate_query_gradients_nonzero
        ),
        "all_candidate_query_updates_nonzero": bool(
            candidate_query_updates_nonzero
        ),
        "effective_spatial_rank_nonzero": float(query["effective_spatial_rank"]) > 1.0,
        "cross_mass_inside_intensity_at_least_0p75": diagnostic_mass >= 0.75,
        "same_pixel_mismatch_at_most_0p15": diagnostic_same <= 0.15,
    }
    return {
        "checks": checks,
        "passed": sum(checks.values()),
        "total": len(checks),
        "mechanism_pass": all(checks.values()),
    }
