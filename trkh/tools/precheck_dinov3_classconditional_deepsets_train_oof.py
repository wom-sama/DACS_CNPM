from __future__ import annotations

import argparse
import hashlib
import json
import math
import random
import time
from pathlib import Path
from typing import Dict, List, Mapping, Optional, Sequence, Tuple

import numpy as np
import torch
import torch.nn.functional as F
from torch import Tensor, nn
from torch.utils.data import DataLoader, Dataset
from tqdm import tqdm

from trkh.models.model import build_model_from_checkpoint
from trkh.tools.audit_dinov3_pair_patch_stat_readiness import (
    FOCUS_CLASS,
    RIVAL_CLASSES,
    _assert_output_outside_train,
    _binary_metrics,
    _model_contract,
    _sha256,
    _write_csv,
    normalized_source_group,
)
from trkh.tools.precheck_dinov3_pair_patchstats_train_oof import (
    MODEL_NAME,
    _validate_checkpoint_contract,
    assign_global_source_folds,
)
from trkh.tools.probe_embedding_prototypes import (
    _build_dataset,
    _collate_classification,
    _resolve_device,
)


PROTOCOL_ID = "TRKH_PRETRAINED_CLASSF_B6_DEEPSETS_MIL_READINESS_20260731"
EXPECTED_DATA_SHA256 = (
    "fa9581d1a595134bd366105c3999170d3553712099e7222cf67f2339f84da156"
)
EXPECTED_CHECKPOINT_SHA256 = (
    "4d3205e7029fa25c313ac622f3ea1abbe0bbf4103d804ee9ac87f527759352a6"
)
EXPECTED_TRAIN_SAMPLES = 8278
MIN_SOURCE_GROUPS = 7000
PATCH_TOKENS = 256
TOKEN_WIDTH = 384
GLOBAL_LOGIT_WIDTH = 5
PAIR_RIVALS = tuple(int(value) for value in RIVAL_CLASSES)
PAIR_COUNT = len(PAIR_RIVALS)
PAIR_EMBED_WIDTH = 24
READOUT_HIDDEN = 16
FOLDS = 5
SEED = 20260731
HEAD_EPOCHS = 20
HEAD_BATCH_SIZE = 128
HEAD_LR = 1e-3
HEAD_WEIGHT_DECAY = 1e-4
HEAD_GRAD_CLIP = 1.0
THRESHOLD = 0.5
MAX_HEAD_PARAMETERS = 12_000
EXPECTED_HEAD_PARAMETERS = 10_577
MAX_HEAD_MACS = 2_600_000
CACHE_DTYPE = np.float16
CACHE_FILENAME = "raw_patch_tokens_f16.npy"
GLOBAL_LOGITS_FILENAME = "global_logits_f32.npy"
LABELS_FILENAME = "labels_i64.npy"
CACHE_MANIFEST_FILENAME = "cache_manifest.json"


def _parse_args(argv: Optional[Sequence[str]] = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Locked B6 train-only source-grouped readiness precheck for a tiny "
            "class-conditional DeepSets readout over frozen DINOv3 patches. "
            "Validation and test datasets are never constructed."
        )
    )
    parser.add_argument("--data", type=Path, required=True)
    parser.add_argument("--checkpoint", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--class-name-mode", type=str, default="raw")
    parser.add_argument("--batch-size", type=int, default=32)
    parser.add_argument("--workers", type=int, default=0)
    parser.add_argument("--device", type=str, default="")
    parser.add_argument("--torch-threads", type=int, default=4)
    parser.add_argument("--preflight-only", action="store_true", default=False)
    return parser.parse_args(argv)


class ClassConditionalDeepSetsReadout(nn.Module):
    """Tiny pair-conditioned permutation-invariant patch distribution probe."""

    def __init__(
        self,
        *,
        token_width: int = TOKEN_WIDTH,
        pair_count: int = PAIR_COUNT,
        pair_embed_width: int = PAIR_EMBED_WIDTH,
        global_logit_width: int = GLOBAL_LOGIT_WIDTH,
        readout_hidden: int = READOUT_HIDDEN,
    ) -> None:
        super().__init__()
        self.token_width = int(token_width)
        self.pair_count = int(pair_count)
        self.pair_embed_width = int(pair_embed_width)
        self.global_logit_width = int(global_logit_width)
        if min(
            self.token_width,
            self.pair_count,
            self.pair_embed_width,
            self.global_logit_width,
            int(readout_hidden),
        ) <= 0:
            raise ValueError("DeepSets dimensions must be positive")
        self.token_norm = nn.LayerNorm(
            self.token_width,
            elementwise_affine=False,
        )
        self.token_projection = nn.Linear(
            self.token_width,
            self.pair_embed_width,
            bias=True,
        )
        self.pair_embedding = nn.Embedding(
            self.pair_count,
            self.pair_embed_width,
        )
        descriptor_width = (
            3 * self.pair_embed_width + self.global_logit_width
        )
        self.output = nn.Sequential(
            nn.Linear(descriptor_width, int(readout_hidden), bias=True),
            nn.GELU(),
            nn.Linear(int(readout_hidden), 1, bias=True),
        )

    def _conditioned_descriptor(
        self,
        projected_tokens: Tensor,
        global_logits: Tensor,
        pair_embedding: Tensor,
    ) -> Tensor:
        conditioned = F.gelu(
            projected_tokens + pair_embedding.unsqueeze(-2)
        )
        mean = conditioned.mean(dim=-2)
        variance = (conditioned - mean.unsqueeze(-2)).square().mean(dim=-2)
        return torch.cat(
            (mean, variance, global_logits, pair_embedding),
            dim=-1,
        )

    def forward(
        self,
        patch_tokens: Tensor,
        global_logits: Tensor,
        pair_ids: Tensor,
    ) -> Tensor:
        if patch_tokens.ndim != 3:
            raise ValueError(
                f"patch_tokens must be [B,N,D], got {tuple(patch_tokens.shape)}"
            )
        if int(patch_tokens.size(-1)) != self.token_width:
            raise ValueError(
                f"Expected token width {self.token_width}, got "
                f"{int(patch_tokens.size(-1))}"
            )
        if global_logits.ndim != 2 or int(global_logits.size(-1)) != int(
            self.global_logit_width
        ):
            raise ValueError(
                "global_logits must be [B,global_logit_width], got "
                f"{tuple(global_logits.shape)}"
            )
        pair_ids = pair_ids.reshape(-1).to(dtype=torch.long)
        if not (
            int(patch_tokens.size(0))
            == int(global_logits.size(0))
            == int(pair_ids.numel())
        ):
            raise ValueError("DeepSets batch dimensions are inconsistent")
        if bool((pair_ids < 0).any()) or bool((pair_ids >= self.pair_count).any()):
            raise ValueError("pair_ids contain an out-of-range value")
        projected = self.token_projection(self.token_norm(patch_tokens))
        pair_embedding = self.pair_embedding(pair_ids)
        descriptor = self._conditioned_descriptor(
            projected,
            global_logits,
            pair_embedding,
        )
        return self.output(descriptor).squeeze(1)

    def forward_all_pairs(
        self,
        patch_tokens: Tensor,
        global_logits: Tensor,
    ) -> Tensor:
        """Evaluate all pair conditions after projecting each token once."""
        if patch_tokens.ndim != 3:
            raise ValueError(
                f"patch_tokens must be [B,N,D], got {tuple(patch_tokens.shape)}"
            )
        if int(patch_tokens.size(-1)) != self.token_width:
            raise ValueError(
                f"Expected token width {self.token_width}, got "
                f"{int(patch_tokens.size(-1))}"
            )
        if global_logits.ndim != 2 or tuple(global_logits.shape) != (
            int(patch_tokens.size(0)),
            self.global_logit_width,
        ):
            raise ValueError(
                "global_logits must align with the patch batch, got "
                f"{tuple(global_logits.shape)}"
            )
        projected = self.token_projection(self.token_norm(patch_tokens))
        pair_ids = torch.arange(
            self.pair_count,
            dtype=torch.long,
            device=patch_tokens.device,
        )
        pair_embedding = self.pair_embedding(pair_ids)
        descriptor = self._conditioned_descriptor(
            projected.unsqueeze(1),
            global_logits.unsqueeze(1).expand(-1, self.pair_count, -1),
            pair_embedding.unsqueeze(0).expand(
                int(patch_tokens.size(0)), -1, -1
            ),
        )
        return self.output(descriptor).squeeze(-1)


def count_trainable_parameters(module: nn.Module) -> int:
    return int(sum(parameter.numel() for parameter in module.parameters()))


def estimate_head_macs(
    *,
    patch_tokens: int = PATCH_TOKENS,
    token_width: int = TOKEN_WIDTH,
    pair_count: int = PAIR_COUNT,
    pair_embed_width: int = PAIR_EMBED_WIDTH,
    global_logit_width: int = GLOBAL_LOGIT_WIDTH,
    readout_hidden: int = READOUT_HIDDEN,
) -> int:
    projection = int(patch_tokens) * int(token_width) * int(pair_embed_width)
    descriptor_width = 3 * int(pair_embed_width) + int(global_logit_width)
    readout = int(pair_count) * (
        descriptor_width * int(readout_hidden) + int(readout_hidden)
    )
    return int(projection + readout)


def architecture_self_check(
    *,
    seed: int = SEED,
    device: torch.device = torch.device("cpu"),
) -> Dict[str, object]:
    torch.manual_seed(int(seed))
    candidate = ClassConditionalDeepSetsReadout().to(
        device=device, dtype=torch.float32
    )
    torch.manual_seed(int(seed))
    control = ClassConditionalDeepSetsReadout().to(
        device=device, dtype=torch.float32
    )
    candidate.eval()
    control.eval()
    generator = torch.Generator(device="cpu").manual_seed(int(seed) + 1)
    patches = torch.randn(
        3,
        PATCH_TOKENS,
        TOKEN_WIDTH,
        generator=generator,
        dtype=torch.float32,
    ).to(device)
    global_logits = torch.randn(
        3,
        GLOBAL_LOGIT_WIDTH,
        generator=generator,
        dtype=torch.float32,
    ).to(device)
    pair_ids = torch.arange(3, dtype=torch.long, device=device)
    permutation = torch.randperm(PATCH_TOKENS, generator=generator).to(
        device=device
    )
    constant = patches.mean(dim=1, keepdim=True)
    with torch.inference_mode():
        baseline = candidate(patches, global_logits, pair_ids)
        permuted = candidate(patches[:, permutation], global_logits, pair_ids)
        repeated = candidate(
            constant.expand(-1, PATCH_TOKENS, -1),
            global_logits,
            pair_ids,
        )
        control_repeated = control(
            constant.expand(-1, PATCH_TOKENS, -1),
            global_logits,
            pair_ids,
        )
        all_pairs = candidate.forward_all_pairs(patches, global_logits)
        one_pair_at_a_time = torch.stack(
            [
                candidate(
                    patches,
                    global_logits,
                    torch.full_like(pair_ids, pair_index),
                )
                for pair_index in range(PAIR_COUNT)
            ],
            dim=1,
        )
    state_errors = [
        float(
            (
                candidate.state_dict()[name]
                - control.state_dict()[name]
            )
            .abs()
            .max()
            .cpu()
            .item()
        )
        for name in candidate.state_dict()
    ]
    return {
        "token_permutation_max_abs_error": float(
            (baseline - permuted).abs().max().cpu().item()
        ),
        "constant_bag_control_max_abs_error": float(
            (repeated - control_repeated).abs().max().cpu().item()
        ),
        "all_pairs_path_max_abs_error": float(
            (all_pairs - one_pair_at_a_time).abs().max().cpu().item()
        ),
        "initial_state_max_abs_error": max(state_errors, default=math.inf),
        "candidate_parameters": count_trainable_parameters(candidate),
        "control_parameters": count_trainable_parameters(control),
        "estimated_head_macs": estimate_head_macs(),
    }


def expand_pair_rows(
    labels: np.ndarray,
    sample_indices: Optional[np.ndarray] = None,
) -> Dict[str, np.ndarray]:
    labels = np.asarray(labels, dtype=np.int64).reshape(-1)
    if sample_indices is None:
        indices = np.arange(labels.size, dtype=np.int64)
    else:
        indices = np.asarray(sample_indices, dtype=np.int64).reshape(-1)
    sample_rows: List[int] = []
    pair_rows: List[int] = []
    target_rows: List[int] = []
    rival_to_pair = {int(rival): index for index, rival in enumerate(PAIR_RIVALS)}
    for sample_index in indices.tolist():
        if sample_index < 0 or sample_index >= labels.size:
            raise IndexError(f"sample index is out of range: {sample_index}")
        target_class = int(labels[sample_index])
        if target_class == int(FOCUS_CLASS):
            for pair_index in range(PAIR_COUNT):
                sample_rows.append(sample_index)
                pair_rows.append(pair_index)
                target_rows.append(1)
        elif target_class in rival_to_pair:
            sample_rows.append(sample_index)
            pair_rows.append(int(rival_to_pair[target_class]))
            target_rows.append(0)
    payload = {
        "sample_indices": np.asarray(sample_rows, dtype=np.int64),
        "pair_ids": np.asarray(pair_rows, dtype=np.int64),
        "targets": np.asarray(target_rows, dtype=np.int64),
    }
    if int(payload["targets"].size) == 0:
        raise ValueError("No B6 pair rows were produced")
    return payload


def six_stratum_weights(pair_ids: np.ndarray, targets: np.ndarray) -> np.ndarray:
    pairs = np.asarray(pair_ids, dtype=np.int64).reshape(-1)
    values = np.asarray(targets, dtype=np.int64).reshape(-1)
    if pairs.size != values.size or pairs.size == 0:
        raise ValueError("pair_ids/targets must be non-empty and aligned")
    weights = np.zeros(values.size, dtype=np.float32)
    for pair_index in range(PAIR_COUNT):
        for target in (0, 1):
            mask = np.logical_and(pairs == pair_index, values == target)
            count = int(mask.sum())
            if count <= 0:
                raise ValueError(
                    f"Missing locked stratum pair={pair_index}, target={target}"
                )
            weights[mask] = float(values.size) / float(2 * PAIR_COUNT * count)
    stratum_totals = [
        float(
            weights[
                np.logical_and(pairs == pair_index, values == target)
            ].sum(dtype=np.float64)
        )
        for pair_index in range(PAIR_COUNT)
        for target in (0, 1)
    ]
    expected_total = float(values.size) / float(2 * PAIR_COUNT)
    # Each example weight is consumed as float32 by the training loss.  At the
    # locked support (thousands of rows per stratum), quantizing one weight and
    # then summing it can legitimately accumulate several 1e-5 of absolute
    # error even though the relative imbalance is below one float32 epsilon.
    # Scale the guard to the expected total instead of using a size-dependent
    # absolute threshold that rejects the canonical dataset.
    rounding_tolerance = max(
        1e-5,
        2.0
        * float(np.finfo(np.float32).eps)
        * max(1.0, abs(expected_total)),
    )
    maximum_deviation = max(
        abs(total - expected_total) for total in stratum_totals
    )
    if maximum_deviation > rounding_tolerance:
        raise RuntimeError("Six-stratum weights are not balanced")
    return weights


def permute_fit_targets_within_pairs(
    pair_ids: np.ndarray,
    targets: np.ndarray,
    *,
    seed: int,
) -> np.ndarray:
    pairs = np.asarray(pair_ids, dtype=np.int64).reshape(-1)
    values = np.asarray(targets, dtype=np.int64).reshape(-1)
    if pairs.size != values.size:
        raise ValueError("pair_ids/targets must be aligned")
    result = values.copy()
    rng = np.random.default_rng(int(seed))
    for pair_index in range(PAIR_COUNT):
        local = np.flatnonzero(pairs == pair_index)
        if local.size == 0:
            raise ValueError(f"Pair {pair_index} has no fit rows")
        result[local] = rng.permutation(values[local])
        if int(result[local].sum()) != int(values[local].sum()):
            raise RuntimeError("Pair-label permutation changed class support")
    return result


class _PairTokenDataset(Dataset):
    def __init__(
        self,
        *,
        token_cache: np.ndarray,
        global_logits: np.ndarray,
        rows: Mapping[str, np.ndarray],
        targets: Optional[np.ndarray] = None,
        pooled_control: bool,
    ) -> None:
        self.token_cache = token_cache
        self.global_logits = np.asarray(global_logits)
        self.sample_indices = np.asarray(rows["sample_indices"], dtype=np.int64)
        self.pair_ids = np.asarray(rows["pair_ids"], dtype=np.int64)
        base_targets = np.asarray(rows["targets"], dtype=np.int64)
        self.targets = (
            base_targets
            if targets is None
            else np.asarray(targets, dtype=np.int64).reshape(-1)
        )
        if not (
            self.sample_indices.size
            == self.pair_ids.size
            == self.targets.size
        ):
            raise ValueError("Expanded B6 rows are not aligned")
        if int(self.token_cache.shape[0]) != int(self.global_logits.shape[0]):
            raise ValueError("Token/global-logit cache rows differ")
        self.weights = six_stratum_weights(self.pair_ids, self.targets)
        self.pooled_control = bool(pooled_control)

    def __len__(self) -> int:
        return int(self.targets.size)

    def __getitem__(self, index: int):
        sample_index = int(self.sample_indices[index])
        tokens = np.asarray(
            self.token_cache[sample_index],
            dtype=np.float32,
        )
        if self.pooled_control:
            pooled = tokens.mean(axis=0, dtype=np.float32, keepdims=True)
            tokens = np.repeat(pooled, PATCH_TOKENS, axis=0)
        tokens = np.array(tokens, dtype=np.float32, copy=True)
        logits = np.array(
            self.global_logits[sample_index],
            dtype=np.float32,
            copy=True,
        )
        return (
            torch.from_numpy(tokens),
            torch.from_numpy(logits),
            torch.as_tensor(int(self.pair_ids[index]), dtype=torch.long),
            torch.as_tensor(float(self.targets[index]), dtype=torch.float32),
            torch.as_tensor(float(self.weights[index]), dtype=torch.float32),
        )


def _seed_everything(seed: int) -> None:
    random.seed(int(seed))
    np.random.seed(int(seed) % (2**32 - 1))
    torch.manual_seed(int(seed))
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(int(seed))


def _train_readout(
    *,
    dataset: Dataset,
    device: torch.device,
    seed: int,
    epochs: int = HEAD_EPOCHS,
    batch_size: int = HEAD_BATCH_SIZE,
) -> Tuple[ClassConditionalDeepSetsReadout, Dict[str, object]]:
    _seed_everything(int(seed))
    model = ClassConditionalDeepSetsReadout().to(
        device=device,
        dtype=torch.float32,
    )
    generator = torch.Generator(device="cpu").manual_seed(int(seed))
    loader = DataLoader(
        dataset,
        batch_size=max(1, int(batch_size)),
        shuffle=True,
        num_workers=0,
        pin_memory=False,
        generator=generator,
    )
    optimizer = torch.optim.AdamW(
        model.parameters(),
        lr=HEAD_LR,
        weight_decay=HEAD_WEIGHT_DECAY,
    )
    final_loss = math.nan
    successful_updates = 0
    model.train()
    for _epoch in range(int(epochs)):
        loss_numerator = 0.0
        loss_denominator = 0.0
        for tokens, logits, pair_ids, targets, weights in loader:
            tokens = tokens.to(device=device, dtype=torch.float32)
            logits = logits.to(device=device, dtype=torch.float32)
            pair_ids = pair_ids.to(device=device, dtype=torch.long)
            targets = targets.to(device=device, dtype=torch.float32)
            weights = weights.to(device=device, dtype=torch.float32)
            optimizer.zero_grad(set_to_none=True)
            predictions = model(tokens, logits, pair_ids)
            elementwise = F.binary_cross_entropy_with_logits(
                predictions,
                targets,
                reduction="none",
            )
            loss = (elementwise * weights).sum() / weights.sum().clamp_min(1e-12)
            if not bool(torch.isfinite(loss)):
                raise FloatingPointError("B6 readout produced a non-finite loss")
            loss.backward()
            gradient_norm = torch.nn.utils.clip_grad_norm_(
                model.parameters(),
                HEAD_GRAD_CLIP,
            )
            if not bool(torch.isfinite(gradient_norm)):
                raise FloatingPointError("B6 readout produced non-finite gradients")
            optimizer.step()
            successful_updates += 1
            loss_numerator += float((elementwise * weights).sum().item())
            loss_denominator += float(weights.sum().item())
        final_loss = loss_numerator / max(1e-12, loss_denominator)
    model.eval()
    if successful_updates <= 0 or not math.isfinite(final_loss):
        raise RuntimeError("B6 readout completed without a finite optimizer update")
    return model, {
        "final_weighted_bce": float(final_loss),
        "successful_optimizer_updates": int(successful_updates),
        "nonfinite_steps": 0,
    }


def _predict_readout(
    *,
    model: ClassConditionalDeepSetsReadout,
    dataset: Dataset,
    device: torch.device,
    batch_size: int = HEAD_BATCH_SIZE,
) -> np.ndarray:
    loader = DataLoader(
        dataset,
        batch_size=max(1, int(batch_size)),
        shuffle=False,
        num_workers=0,
        pin_memory=False,
    )
    batches: List[np.ndarray] = []
    model.eval()
    with torch.inference_mode():
        for tokens, logits, pair_ids, _targets, _weights in loader:
            tokens = tokens.to(device=device, dtype=torch.float32)
            logits = logits.to(device=device, dtype=torch.float32)
            pair_ids = pair_ids.to(device=device, dtype=torch.long)
            probabilities = torch.sigmoid(model(tokens, logits, pair_ids))
            if not bool(torch.isfinite(probabilities).all()):
                raise FloatingPointError("B6 readout produced non-finite probabilities")
            batches.append(probabilities.cpu().numpy().astype(np.float64))
    if not batches:
        raise RuntimeError("B6 prediction loader produced no batches")
    result = np.concatenate(batches, axis=0)
    if int(result.size) != len(dataset):
        raise RuntimeError("B6 prediction count does not match the dataset")
    return result


def _path_rows_sha256(paths: Sequence[str]) -> str:
    digest = hashlib.sha256()
    for path in paths:
        digest.update(str(Path(path).resolve()).casefold().encode("utf-8"))
        digest.update(b"\n")
    return digest.hexdigest()


def _labels_sha256(labels: np.ndarray) -> str:
    values = np.asarray(labels, dtype="<i8").reshape(-1)
    return hashlib.sha256(values.tobytes(order="C")).hexdigest()


def _atomic_json(path: Path, payload: Mapping[str, object]) -> None:
    target = Path(path)
    temporary = target.with_suffix(target.suffix + ".partial")
    temporary.write_text(json.dumps(dict(payload), indent=2), encoding="utf-8")
    temporary.replace(target)


def _cache_paths(output_dir: Path) -> Dict[str, Path]:
    root = Path(output_dir)
    return {
        "tokens": root / CACHE_FILENAME,
        "logits": root / GLOBAL_LOGITS_FILENAME,
        "labels": root / LABELS_FILENAME,
        "manifest": root / CACHE_MANIFEST_FILENAME,
    }


def _array_all_finite(array: np.ndarray, *, row_chunk: int = 64) -> bool:
    values = np.asarray(array)
    if values.ndim == 0:
        return bool(np.isfinite(values).all())
    for start in range(0, int(values.shape[0]), max(1, int(row_chunk))):
        stop = min(int(values.shape[0]), start + max(1, int(row_chunk)))
        if not bool(np.isfinite(values[start:stop]).all()):
            return False
    return True


def _validate_cache_manifest(
    *,
    manifest: Mapping[str, object],
    cache_paths: Mapping[str, Path],
    expected_paths_sha256: str,
) -> None:
    expected = {
        "protocol_id": PROTOCOL_ID,
        "data_sha256": EXPECTED_DATA_SHA256,
        "checkpoint_sha256": EXPECTED_CHECKPOINT_SHA256,
        "paths_sha256": str(expected_paths_sha256),
        "token_shape": [EXPECTED_TRAIN_SAMPLES, PATCH_TOKENS, TOKEN_WIDTH],
        "token_dtype": "float16",
        "global_logits_shape": [EXPECTED_TRAIN_SAMPLES, GLOBAL_LOGIT_WIDTH],
        "global_logits_dtype": "float32",
        "labels_shape": [EXPECTED_TRAIN_SAMPLES],
        "labels_dtype": "int64",
        "train_split_used": True,
        "validation_split_used": False,
        "test_split_used": False,
    }
    mismatches = {
        key: {"expected": value, "observed": manifest.get(key)}
        for key, value in expected.items()
        if manifest.get(key) != value
    }
    if mismatches:
        raise ValueError(f"B6 cache manifest mismatch: {mismatches}")
    for key in ("tokens", "logits", "labels"):
        path = Path(cache_paths[key])
        if not path.is_file():
            raise FileNotFoundError(f"B6 cache file is missing: {path}")
        expected_hash = str(manifest.get(f"{key}_sha256", ""))
        if not expected_hash or _sha256(path) != expected_hash:
            raise ValueError(f"B6 cache hash mismatch: {path}")


def _load_existing_cache(
    *,
    output_dir: Path,
    paths: Sequence[str],
) -> Optional[Dict[str, object]]:
    cache_paths = _cache_paths(output_dir)
    present = {key: path.is_file() for key, path in cache_paths.items()}
    if not any(present.values()):
        return None
    if not all(present.values()):
        raise RuntimeError(
            "B6 output contains a partial cache; use a new output directory: "
            f"{present}"
        )
    manifest = json.loads(cache_paths["manifest"].read_text(encoding="utf-8"))
    if not isinstance(manifest, Mapping):
        raise TypeError("B6 cache manifest must be a JSON object")
    _validate_cache_manifest(
        manifest=manifest,
        cache_paths=cache_paths,
        expected_paths_sha256=_path_rows_sha256(paths),
    )
    tokens = np.load(cache_paths["tokens"], mmap_mode="r", allow_pickle=False)
    logits = np.load(cache_paths["logits"], mmap_mode="r", allow_pickle=False)
    labels = np.load(cache_paths["labels"], mmap_mode="r", allow_pickle=False)
    if tuple(tokens.shape) != (EXPECTED_TRAIN_SAMPLES, PATCH_TOKENS, TOKEN_WIDTH):
        raise ValueError(f"Unexpected B6 token cache shape: {tokens.shape}")
    if tuple(logits.shape) != (EXPECTED_TRAIN_SAMPLES, GLOBAL_LOGIT_WIDTH):
        raise ValueError(f"Unexpected B6 global-logit shape: {logits.shape}")
    if tuple(labels.shape) != (EXPECTED_TRAIN_SAMPLES,):
        raise ValueError(f"Unexpected B6 label-cache shape: {labels.shape}")
    if tokens.dtype != np.dtype(np.float16):
        raise ValueError(f"Unexpected B6 token-cache dtype: {tokens.dtype}")
    if logits.dtype != np.dtype(np.float32):
        raise ValueError(f"Unexpected B6 global-logit dtype: {logits.dtype}")
    if labels.dtype != np.dtype(np.int64):
        raise ValueError(f"Unexpected B6 label-cache dtype: {labels.dtype}")
    if not _array_all_finite(tokens):
        raise ValueError("B6 token cache contains non-finite values")
    if not _array_all_finite(logits):
        raise ValueError("B6 global-logit cache contains non-finite values")
    labels_array = np.asarray(labels, dtype=np.int64)
    if _labels_sha256(labels_array) != str(
        manifest.get("labels_content_sha256", "")
    ):
        raise ValueError("B6 cached label content hash mismatch")
    return {
        "tokens": tokens,
        "global_logits": logits,
        "labels": labels_array,
        "manifest": dict(manifest),
    }


def _extract_train_cache(
    *,
    model: nn.Module,
    dataset: Dataset,
    paths: Sequence[str],
    output_dir: Path,
    device: torch.device,
    batch_size: int,
    workers: int,
) -> Dict[str, object]:
    cache_paths = _cache_paths(output_dir)
    partial_tokens = cache_paths["tokens"].with_suffix(".npy.partial")
    partial_logits = cache_paths["logits"].with_suffix(".npy.partial")
    partial_labels = cache_paths["labels"].with_suffix(".npy.partial")
    for path in (partial_tokens, partial_logits, partial_labels):
        if path.exists():
            raise RuntimeError(
                f"A partial B6 cache exists; use a new output directory: {path}"
            )
    token_cache = np.lib.format.open_memmap(
        partial_tokens,
        mode="w+",
        dtype=CACHE_DTYPE,
        shape=(len(dataset), PATCH_TOKENS, TOKEN_WIDTH),
    )
    logit_cache = np.lib.format.open_memmap(
        partial_logits,
        mode="w+",
        dtype=np.float32,
        shape=(len(dataset), GLOBAL_LOGIT_WIDTH),
    )
    label_cache = np.lib.format.open_memmap(
        partial_labels,
        mode="w+",
        dtype=np.int64,
        shape=(len(dataset),),
    )
    loader = DataLoader(
        dataset,
        batch_size=max(1, int(batch_size)),
        shuffle=False,
        num_workers=max(0, int(workers)),
        pin_memory=False,
        collate_fn=_collate_classification,
    )
    prefix_tokens = int(getattr(model, "num_prefix_tokens", 0) or 0)
    maximum_pool_parity_error = 0.0
    maximum_quantization_error = 0.0
    cursor = 0
    model.eval()
    with torch.inference_mode():
        iterator = tqdm(loader, desc="b6-dinov3-patch-cache-fp32", dynamic_ncols=True)
        for images, labels, _metadata in iterator:
            images = images.to(device=device, dtype=torch.float32, non_blocking=True)
            raw_tokens = model.forward_features(images)
            if not torch.is_tensor(raw_tokens) or raw_tokens.ndim != 3:
                raise TypeError("DINOv3 forward_features must return [B,N,D]")
            patches = raw_tokens.float()[:, prefix_tokens:]
            if tuple(patches.shape[1:]) != (PATCH_TOKENS, TOKEN_WIDTH):
                raise ValueError(
                    "Locked B6 patch shape mismatch: "
                    f"{tuple(patches.shape[1:])} != {(PATCH_TOKENS, TOKEN_WIDTH)}"
                )
            pooled = model.forward_head(raw_tokens, pre_logits=True).float()
            parity_error = float(
                (pooled - patches.mean(dim=1)).abs().max().cpu().item()
            )
            maximum_pool_parity_error = max(
                maximum_pool_parity_error,
                parity_error,
            )
            if parity_error > 1e-6:
                raise RuntimeError(
                    "B6 raw patch mean does not match deployed average pool: "
                    f"max_abs={parity_error}"
                )
            global_logits = model.head(pooled).float()
            if tuple(global_logits.shape[1:]) != (GLOBAL_LOGIT_WIDTH,):
                raise ValueError(
                    f"Locked B6 global-logit shape mismatch: {global_logits.shape}"
                )
            patch_numpy = patches.cpu().numpy().astype(np.float32, copy=False)
            global_logit_numpy = global_logits.cpu().numpy().astype(
                np.float32, copy=False
            )
            if not bool(np.isfinite(patch_numpy).all()):
                raise FloatingPointError(
                    "B6 DINOv3 patch features contain non-finite values"
                )
            if not bool(np.isfinite(global_logit_numpy).all()):
                raise FloatingPointError(
                    "B6 DINOv3 global logits contain non-finite values"
                )
            quantized = patch_numpy.astype(CACHE_DTYPE)
            quantization_error = float(
                np.max(np.abs(patch_numpy - quantized.astype(np.float32)))
            )
            maximum_quantization_error = max(
                maximum_quantization_error,
                quantization_error,
            )
            count = int(labels.numel())
            end = cursor + count
            token_cache[cursor:end] = quantized
            logit_cache[cursor:end] = global_logit_numpy
            label_cache[cursor:end] = labels.cpu().numpy().astype(np.int64)
            cursor = end
    if cursor != len(dataset) or cursor != len(paths):
        raise RuntimeError(
            f"B6 cache row mismatch: cached={cursor}, dataset={len(dataset)}, "
            f"paths={len(paths)}"
        )
    token_cache.flush()
    logit_cache.flush()
    label_cache.flush()
    del token_cache, logit_cache, label_cache
    partial_tokens.replace(cache_paths["tokens"])
    partial_logits.replace(cache_paths["logits"])
    partial_labels.replace(cache_paths["labels"])
    labels_array = np.load(cache_paths["labels"], mmap_mode="r", allow_pickle=False)
    manifest: Dict[str, object] = {
        "schema_version": 1,
        "protocol_id": PROTOCOL_ID,
        "data_sha256": EXPECTED_DATA_SHA256,
        "checkpoint_sha256": EXPECTED_CHECKPOINT_SHA256,
        "paths_sha256": _path_rows_sha256(paths),
        "labels_content_sha256": _labels_sha256(labels_array),
        "token_shape": [len(dataset), PATCH_TOKENS, TOKEN_WIDTH],
        "token_dtype": "float16",
        "global_logits_shape": [len(dataset), GLOBAL_LOGIT_WIDTH],
        "global_logits_dtype": "float32",
        "labels_shape": [len(dataset)],
        "labels_dtype": "int64",
        "maximum_pool_parity_error": float(maximum_pool_parity_error),
        "maximum_float16_quantization_error": float(maximum_quantization_error),
        "tokens_sha256": _sha256(cache_paths["tokens"]),
        "logits_sha256": _sha256(cache_paths["logits"]),
        "labels_sha256": _sha256(cache_paths["labels"]),
        "train_split_used": True,
        "validation_split_used": False,
        "test_split_used": False,
        "validation_dataset_constructed": False,
        "test_dataset_constructed": False,
    }
    _atomic_json(cache_paths["manifest"], manifest)
    return {
        "tokens": np.load(
            cache_paths["tokens"], mmap_mode="r", allow_pickle=False
        ),
        "global_logits": np.load(
            cache_paths["logits"], mmap_mode="r", allow_pickle=False
        ),
        "labels": np.asarray(labels_array, dtype=np.int64),
        "manifest": manifest,
    }


def _load_or_extract_cache(
    *,
    model: nn.Module,
    dataset: Dataset,
    paths: Sequence[str],
    output_dir: Path,
    device: torch.device,
    batch_size: int,
    workers: int,
) -> Dict[str, object]:
    existing = _load_existing_cache(
        output_dir=output_dir,
        paths=paths,
    )
    if existing is not None:
        return existing
    return _extract_train_cache(
        model=model,
        dataset=dataset,
        paths=paths,
        output_dir=output_dir,
        device=device,
        batch_size=batch_size,
        workers=workers,
    )


def _prefixed_metrics(
    row: Dict[str, object],
    prefix: str,
    metrics: Mapping[str, float],
) -> None:
    for key, value in metrics.items():
        row[f"{prefix}_{key}"] = float(value)


def run_deepsets_oof(
    *,
    token_cache: np.ndarray,
    global_logits: np.ndarray,
    labels: np.ndarray,
    source_groups: np.ndarray,
    fold_assignments: np.ndarray,
    paths: Sequence[str],
    output_dir: Path,
    device: torch.device,
) -> Tuple[List[Dict[str, object]], List[Dict[str, object]], List[Dict[str, object]]]:
    labels = np.asarray(labels, dtype=np.int64).reshape(-1)
    groups = np.asarray(source_groups, dtype=object).reshape(-1)
    assignments = np.asarray(fold_assignments, dtype=np.int64).reshape(-1)
    if not (
        int(token_cache.shape[0])
        == int(global_logits.shape[0])
        == labels.size
        == groups.size
        == assignments.size
        == len(paths)
    ):
        raise ValueError("B6 cached arrays/folds/paths are not aligned")
    fold_metric_rows: List[Dict[str, object]] = []
    prediction_rows: List[Dict[str, object]] = []
    model_dir = Path(output_dir) / "fold_readouts"
    model_dir.mkdir(parents=True, exist_ok=True)
    for fold_index in range(FOLDS):
        fit_indices = np.flatnonzero(assignments != int(fold_index))
        hold_indices = np.flatnonzero(assignments == int(fold_index))
        fit_sources = set(groups[fit_indices].tolist())
        hold_sources = set(groups[hold_indices].tolist())
        overlap = fit_sources & hold_sources
        if overlap:
            raise RuntimeError(
                f"B6 source leakage in fold {fold_index}: {len(overlap)} groups"
            )
        fit_rows = expand_pair_rows(labels, fit_indices)
        hold_rows = expand_pair_rows(labels, hold_indices)
        fold_seed = SEED + int(fold_index)
        candidate_fit = _PairTokenDataset(
            token_cache=token_cache,
            global_logits=global_logits,
            rows=fit_rows,
            pooled_control=False,
        )
        control_fit = _PairTokenDataset(
            token_cache=token_cache,
            global_logits=global_logits,
            rows=fit_rows,
            pooled_control=True,
        )
        permuted_targets = permute_fit_targets_within_pairs(
            fit_rows["pair_ids"],
            fit_rows["targets"],
            seed=fold_seed,
        )
        placebo_fit = _PairTokenDataset(
            token_cache=token_cache,
            global_logits=global_logits,
            rows=fit_rows,
            targets=permuted_targets,
            pooled_control=False,
        )
        candidate, candidate_train = _train_readout(
            dataset=candidate_fit,
            device=device,
            seed=fold_seed,
        )
        control, control_train = _train_readout(
            dataset=control_fit,
            device=device,
            seed=fold_seed,
        )
        placebo, placebo_train = _train_readout(
            dataset=placebo_fit,
            device=device,
            seed=fold_seed,
        )
        if count_trainable_parameters(candidate) != count_trainable_parameters(control):
            raise RuntimeError("B6 candidate/control parameter counts differ")
        candidate_hold = _PairTokenDataset(
            token_cache=token_cache,
            global_logits=global_logits,
            rows=hold_rows,
            pooled_control=False,
        )
        control_hold = _PairTokenDataset(
            token_cache=token_cache,
            global_logits=global_logits,
            rows=hold_rows,
            pooled_control=True,
        )
        candidate_probabilities = _predict_readout(
            model=candidate,
            dataset=candidate_hold,
            device=device,
        )
        control_probabilities = _predict_readout(
            model=control,
            dataset=control_hold,
            device=device,
        )
        placebo_probabilities = _predict_readout(
            model=placebo,
            dataset=candidate_hold,
            device=device,
        )
        checkpoint_payload = {
            "protocol_id": PROTOCOL_ID,
            "fold": int(fold_index),
            "seed": int(fold_seed),
            "candidate_state_dict": candidate.state_dict(),
            "control_state_dict": control.state_dict(),
            "placebo_state_dict": placebo.state_dict(),
            "candidate_train": candidate_train,
            "control_train": control_train,
            "placebo_train": placebo_train,
        }
        torch.save(checkpoint_payload, model_dir / f"fold_{fold_index:02d}.pt")
        hold_pair_ids = np.asarray(hold_rows["pair_ids"], dtype=np.int64)
        hold_targets = np.asarray(hold_rows["targets"], dtype=np.int64)
        hold_sample_indices = np.asarray(hold_rows["sample_indices"], dtype=np.int64)
        for pair_index, rival in enumerate(PAIR_RIVALS):
            local = np.flatnonzero(hold_pair_ids == int(pair_index))
            pair_targets = hold_targets[local]
            control_metrics = _binary_metrics(
                pair_targets,
                control_probabilities[local],
            )
            candidate_metrics = _binary_metrics(
                pair_targets,
                candidate_probabilities[local],
            )
            placebo_metrics = _binary_metrics(
                pair_targets,
                placebo_probabilities[local],
            )
            fold_row: Dict[str, object] = {
                "pair": f"{int(rival)}-1",
                "rival_class": int(rival),
                "fold": int(fold_index),
                "fit_samples": int(fit_indices.size),
                "holdout_samples": int(hold_indices.size),
                "pair_holdout_samples": int(local.size),
                "source_overlap": int(len(overlap)),
                "candidate_final_weighted_bce": float(
                    candidate_train["final_weighted_bce"]
                ),
                "control_final_weighted_bce": float(
                    control_train["final_weighted_bce"]
                ),
                "placebo_final_weighted_bce": float(
                    placebo_train["final_weighted_bce"]
                ),
            }
            _prefixed_metrics(fold_row, "control", control_metrics)
            _prefixed_metrics(fold_row, "candidate", candidate_metrics)
            _prefixed_metrics(fold_row, "placebo", placebo_metrics)
            fold_row["delta_auroc"] = float(
                candidate_metrics["auroc"] - control_metrics["auroc"]
            )
            fold_row["delta_f1_class1"] = float(
                candidate_metrics["f1_class1"] - control_metrics["f1_class1"]
            )
            fold_metric_rows.append(fold_row)
            for offset in local.tolist():
                sample_index = int(hold_sample_indices[offset])
                prediction_rows.append(
                    {
                        "sample_index": sample_index,
                        "image_path": str(paths[sample_index]),
                        "source_group": str(groups[sample_index]),
                        "fold": int(fold_index),
                        "target_index": int(labels[sample_index]),
                        "pair": f"{int(rival)}-1",
                        "binary_target_class1": int(hold_targets[offset]),
                        "control_probability_class1": float(
                            control_probabilities[offset]
                        ),
                        "candidate_probability_class1": float(
                            candidate_probabilities[offset]
                        ),
                        "placebo_probability_class1": float(
                            placebo_probabilities[offset]
                        ),
                    }
                )
    pair_rows: List[Dict[str, object]] = []
    for rival in PAIR_RIVALS:
        selected = [row for row in prediction_rows if row["pair"] == f"{rival}-1"]
        targets = np.asarray(
            [row["binary_target_class1"] for row in selected],
            dtype=np.int64,
        )
        control_probabilities = np.asarray(
            [row["control_probability_class1"] for row in selected],
            dtype=np.float64,
        )
        candidate_probabilities = np.asarray(
            [row["candidate_probability_class1"] for row in selected],
            dtype=np.float64,
        )
        placebo_probabilities = np.asarray(
            [row["placebo_probability_class1"] for row in selected],
            dtype=np.float64,
        )
        control_metrics = _binary_metrics(targets, control_probabilities)
        candidate_metrics = _binary_metrics(targets, candidate_probabilities)
        placebo_metrics = _binary_metrics(targets, placebo_probabilities)
        pair_row: Dict[str, object] = {
            "pair": f"{int(rival)}-1",
            "rival_class": int(rival),
            "samples": int(targets.size),
            "class1_samples": int(targets.sum()),
            "rival_samples": int((targets == 0).sum()),
        }
        _prefixed_metrics(pair_row, "control", control_metrics)
        _prefixed_metrics(pair_row, "candidate", candidate_metrics)
        _prefixed_metrics(pair_row, "placebo", placebo_metrics)
        for key in (
            "balanced_accuracy",
            "auroc",
            "precision_class1",
            "recall_class1",
            "specificity_rival",
            "f1_class1",
        ):
            pair_row[f"delta_{key}"] = float(
                candidate_metrics[key] - control_metrics[key]
            )
        pair_row["rival_fp_reduction"] = float(
            (control_metrics["fp"] - candidate_metrics["fp"])
            / max(1.0, control_metrics["fp"])
        )
        pair_rows.append(pair_row)
    return pair_rows, fold_metric_rows, prediction_rows


def assess_deepsets_readiness(
    *,
    pair_rows: Sequence[Mapping[str, object]],
    fold_metric_rows: Sequence[Mapping[str, object]],
    global_fold_rows: Sequence[Mapping[str, object]],
    train_samples: int,
    source_groups: int,
    architecture: Mapping[str, object],
    maximum_pool_parity_error: float,
    cache_finite: bool,
) -> Dict[str, object]:
    pairs = [dict(row) for row in pair_rows]
    folds = [dict(row) for row in fold_metric_rows]
    global_folds = [dict(row) for row in global_fold_rows]
    thresholds: Dict[str, object] = {
        "required_train_samples": EXPECTED_TRAIN_SAMPLES,
        "min_source_groups": MIN_SOURCE_GROUPS,
        "required_global_folds": FOLDS,
        "required_pair_folds": FOLDS * PAIR_COUNT,
        "max_source_overlap": 0,
        "max_pool_parity_error": 1e-6,
        "max_invariance_error": 1e-6,
        "max_constant_bag_error": 1e-6,
        "max_all_pairs_path_error": 1e-6,
        "max_initial_state_error": 0.0,
        "expected_head_parameters": EXPECTED_HEAD_PARAMETERS,
        "max_head_parameters": MAX_HEAD_PARAMETERS,
        "max_head_macs": MAX_HEAD_MACS,
        "max_placebo_mean_auroc": 0.55,
        "min_mean_auroc_gain": 0.010,
        "min_positive_pair_fold_gains": 10,
        "min_pairs_with_auroc_gain": 2,
        "min_pair_auroc_gain": 0.010,
        "max_pair_auroc_loss": 0.005,
        "min_pairs_with_f1_gain": 2,
        "min_pair_f1_gain": 0.010,
        "max_pair_recall_loss": 0.010,
        "min_class1_tp_retention": 0.980,
        "min_rival_fp_reduction": 0.100,
        "critical_rival_class": 2,
        "min_critical_auroc_gain": 0.010,
        "min_critical_fp_reduction": 0.100,
    }
    if len(pairs) != PAIR_COUNT:
        raise ValueError(f"B6 requires {PAIR_COUNT} pair rows, got {len(pairs)}")
    critical = next(
        (row for row in pairs if int(row["rival_class"]) == 2),
        None,
    )
    if critical is None:
        raise ValueError("B6 readiness is missing the critical 2-1 pair")
    expected_pair_fold_keys = {
        (int(rival), int(fold_index))
        for rival in PAIR_RIVALS
        for fold_index in range(FOLDS)
    }
    observed_pair_fold_keys = {
        (int(row["rival_class"]), int(row["fold"])) for row in folds
    }
    observed_global_folds = {int(row["fold"]) for row in global_folds}
    required_fold_fields = {
        "control_auroc",
        "candidate_auroc",
        "placebo_auroc",
        "delta_auroc",
        "control_f1_class1",
        "candidate_f1_class1",
    }
    metric_fields_complete = all(
        required_fold_fields.issubset(row) for row in folds
    )
    numeric_values = [
        float(value)
        for row in [*pairs, *folds]
        for value in row.values()
        if isinstance(value, (int, float, np.integer, np.floating))
    ]
    metrics_finite = bool(numeric_values) and bool(
        np.isfinite(np.asarray(numeric_values, dtype=np.float64)).all()
    )
    mean_auroc_gain = float(np.mean([float(row["delta_auroc"]) for row in pairs]))
    positive_pair_fold_gains = int(
        sum(float(row["delta_auroc"]) > 0.0 for row in folds)
    )
    pairs_with_auroc_gain = int(
        sum(
            float(row["delta_auroc"])
            >= float(thresholds["min_pair_auroc_gain"])
            for row in pairs
        )
    )
    pairs_with_f1_gain = int(
        sum(
            float(row["delta_f1_class1"])
            >= float(thresholds["min_pair_f1_gain"])
            for row in pairs
        )
    )
    minimum_pair_auroc_delta = float(
        min(float(row["delta_auroc"]) for row in pairs)
    )
    minimum_pair_recall_delta = float(
        min(float(row["delta_recall_class1"]) for row in pairs)
    )
    control_tp = float(sum(float(row["control_tp"]) for row in pairs))
    candidate_tp = float(sum(float(row["candidate_tp"]) for row in pairs))
    control_fp = float(sum(float(row["control_fp"]) for row in pairs))
    candidate_fp = float(sum(float(row["candidate_fp"]) for row in pairs))
    tp_retention = candidate_tp / max(1.0, control_tp)
    fp_reduction = (control_fp - candidate_fp) / max(1.0, control_fp)
    placebo_mean_auroc = float(
        np.mean([float(row["placebo_auroc"]) for row in folds])
    )
    maximum_overlap = int(
        max(
            [int(row["source_overlap"]) for row in global_folds]
            + [int(row["source_overlap"]) for row in folds]
        )
    )
    candidate_parameters = int(architecture["candidate_parameters"])
    control_parameters = int(architecture["control_parameters"])
    estimated_macs = int(architecture["estimated_head_macs"])
    permutation_error = float(
        architecture["token_permutation_max_abs_error"]
    )
    constant_error = float(
        architecture["constant_bag_control_max_abs_error"]
    )
    all_pairs_error = float(architecture["all_pairs_path_max_abs_error"])
    initial_state_error = float(architecture["initial_state_max_abs_error"])
    critical_fp_reduction = float(critical["rival_fp_reduction"])
    observed: Dict[str, object] = {
        "mean_auroc_gain": mean_auroc_gain,
        "positive_pair_fold_gains": positive_pair_fold_gains,
        "pairs_with_auroc_gain": pairs_with_auroc_gain,
        "pairs_with_f1_gain": pairs_with_f1_gain,
        "minimum_pair_auroc_delta": minimum_pair_auroc_delta,
        "minimum_pair_recall_delta": minimum_pair_recall_delta,
        "class1_tp_retention": float(tp_retention),
        "rival_fp_reduction": float(fp_reduction),
        "control_tp": int(control_tp),
        "candidate_tp": int(candidate_tp),
        "control_fp": int(control_fp),
        "candidate_fp": int(candidate_fp),
        "placebo_mean_auroc": placebo_mean_auroc,
        "maximum_source_overlap": maximum_overlap,
        "maximum_pool_parity_error": float(maximum_pool_parity_error),
        "candidate_parameters": candidate_parameters,
        "control_parameters": control_parameters,
        "estimated_head_macs": estimated_macs,
        "token_permutation_max_abs_error": permutation_error,
        "constant_bag_control_max_abs_error": constant_error,
        "all_pairs_path_max_abs_error": all_pairs_error,
        "initial_state_max_abs_error": initial_state_error,
        "metrics_finite": metrics_finite,
        "metric_fields_complete": metric_fields_complete,
        "critical_2_1_auroc_gain": float(critical["delta_auroc"]),
        "critical_2_1_fp_reduction": critical_fp_reduction,
        "critical_2_1_recall_delta": float(critical["delta_recall_class1"]),
    }
    checks = {
        "canonical_train_support": int(train_samples) == EXPECTED_TRAIN_SAMPLES,
        "source_group_support": int(source_groups) >= MIN_SOURCE_GROUPS,
        "complete_global_folds": len(global_folds) == FOLDS
        and observed_global_folds == set(range(FOLDS)),
        "complete_pair_fold_coverage": len(folds) == FOLDS * PAIR_COUNT
        and observed_pair_fold_keys == expected_pair_fold_keys
        and metric_fields_complete,
        "source_group_folds_disjoint": maximum_overlap
        <= int(thresholds["max_source_overlap"]),
        "cache_finite": bool(cache_finite),
        "metrics_finite": metrics_finite,
        "deployed_pool_parity": float(maximum_pool_parity_error)
        <= float(thresholds["max_pool_parity_error"]),
        "token_permutation_invariance": permutation_error
        <= float(thresholds["max_invariance_error"]),
        "constant_bag_control_parity": constant_error
        <= float(thresholds["max_constant_bag_error"]),
        "all_pairs_projection_path_parity": all_pairs_error
        <= float(thresholds["max_all_pairs_path_error"]),
        "initial_state_matched": initial_state_error
        <= float(thresholds["max_initial_state_error"]),
        "parameter_count_matched": candidate_parameters == control_parameters
        == int(thresholds["expected_head_parameters"]),
        "parameter_budget": candidate_parameters
        <= int(thresholds["max_head_parameters"]),
        "mac_budget": estimated_macs <= int(thresholds["max_head_macs"]),
        "placebo_falsification": placebo_mean_auroc
        <= float(thresholds["max_placebo_mean_auroc"]),
        "mean_auroc_gain": mean_auroc_gain
        >= float(thresholds["min_mean_auroc_gain"]),
        "fold_direction_stability": positive_pair_fold_gains
        >= int(thresholds["min_positive_pair_fold_gains"]),
        "pair_auroc_support": pairs_with_auroc_gain
        >= int(thresholds["min_pairs_with_auroc_gain"]),
        "no_pair_auroc_collapse": minimum_pair_auroc_delta
        >= -float(thresholds["max_pair_auroc_loss"]),
        "pair_f1_support": pairs_with_f1_gain
        >= int(thresholds["min_pairs_with_f1_gain"]),
        "class1_recall_protected": minimum_pair_recall_delta
        >= -float(thresholds["max_pair_recall_loss"]),
        "class1_tp_retained": tp_retention
        >= float(thresholds["min_class1_tp_retention"]),
        "rival_false_positives_reduced": fp_reduction
        >= float(thresholds["min_rival_fp_reduction"]),
        "critical_2_1_auroc_gain": float(critical["delta_auroc"])
        >= float(thresholds["min_critical_auroc_gain"]),
        "critical_2_1_fp_reduction": critical_fp_reduction
        >= float(thresholds["min_critical_fp_reduction"]),
        "critical_2_1_recall_protected": float(
            critical["delta_recall_class1"]
        )
        >= -float(thresholds["max_pair_recall_loss"]),
    }
    failed = [name for name, passed in checks.items() if not bool(passed)]
    ready = not failed
    return {
        "deepsets_patch_distribution_ready": bool(ready),
        "implementation_permission": bool(ready),
        "validation_permission": False,
        "smoke_permission": False,
        "full_train_permission": False,
        "test_permission": False,
        "checks": checks,
        "failed_checks": failed,
        "observed": observed,
        "thresholds": thresholds,
    }


def _locked_protocol_payload() -> Dict[str, object]:
    return {
        "protocol_id": PROTOCOL_ID,
        "focus_class": int(FOCUS_CLASS),
        "rival_classes": list(PAIR_RIVALS),
        "folds": FOLDS,
        "seed": SEED,
        "candidate": "raw_patch_bag_classconditional_deepsets",
        "control": "repeated_deployed_pooled_token_equal_capacity_deepsets",
        "token_projection": [TOKEN_WIDTH, PAIR_EMBED_WIDTH],
        "aggregates": ["mean", "variance"],
        "global_logit_width": GLOBAL_LOGIT_WIDTH,
        "readout_hidden": READOUT_HIDDEN,
        "trainable_parameters": EXPECTED_HEAD_PARAMETERS,
        "optimized_all_pairs_head_macs": estimate_head_macs(),
        "projection_reused_across_pairs": True,
        "head_epochs": HEAD_EPOCHS,
        "head_batch_size": HEAD_BATCH_SIZE,
        "optimizer": "AdamW",
        "learning_rate": HEAD_LR,
        "weight_decay": HEAD_WEIGHT_DECAY,
        "gradient_clip": HEAD_GRAD_CLIP,
        "threshold": THRESHOLD,
        "head_fp32": True,
        "early_stopping": False,
        "candidate_selection_uses_validation": False,
        "architecture_or_threshold_sweep": False,
    }


def run_precheck(args: argparse.Namespace) -> Dict[str, object]:
    if int(args.torch_threads) > 0:
        torch.set_num_threads(int(args.torch_threads))
    if int(args.batch_size) <= 0 or int(args.workers) < 0:
        raise ValueError("batch-size must be positive and workers non-negative")
    data_path = Path(args.data).resolve()
    checkpoint_path = Path(args.checkpoint).resolve()
    output_dir = Path(args.output_dir).resolve()
    for required in (data_path, checkpoint_path):
        if not required.is_file():
            raise FileNotFoundError(required)
    data_sha256 = _sha256(data_path)
    checkpoint_sha256 = _sha256(checkpoint_path)
    if data_sha256 != EXPECTED_DATA_SHA256:
        raise ValueError(
            "Locked B6 data YAML hash mismatch: "
            f"{data_sha256} != {EXPECTED_DATA_SHA256}"
        )
    if checkpoint_sha256 != EXPECTED_CHECKPOINT_SHA256:
        raise ValueError(
            "Locked B6 checkpoint hash mismatch: "
            f"{checkpoint_sha256} != {EXPECTED_CHECKPOINT_SHA256}"
        )
    checkpoint = torch.load(checkpoint_path, map_location="cpu", weights_only=False)
    if not isinstance(checkpoint, Mapping):
        raise TypeError(f"Invalid B6 checkpoint payload: {checkpoint_path}")
    model = build_model_from_checkpoint(dict(checkpoint)).eval()
    model_contract = _model_contract(model)
    checkpoint_contract = _validate_checkpoint_contract(checkpoint, model_contract)
    dataset, class_names = _build_dataset(
        data_yaml=data_path,
        split="train",
        checkpoint=checkpoint,
        class_name_mode=str(args.class_name_mode),
        max_samples=0,
    )
    if len(dataset) != EXPECTED_TRAIN_SAMPLES:
        raise ValueError(
            f"Locked B6 train support mismatch: {len(dataset)} != "
            f"{EXPECTED_TRAIN_SAMPLES}"
        )
    if len(class_names) != GLOBAL_LOGIT_WIDTH:
        raise ValueError(
            f"Locked B6 requires five classes, got {len(class_names)}"
        )
    sample_paths_fn = getattr(dataset, "sample_paths", None)
    if not callable(sample_paths_fn):
        raise TypeError("B6 train dataset must expose sample_paths()")
    paths = [str(path) for path in sample_paths_fn()]
    if len(paths) != len(dataset):
        raise RuntimeError("B6 train paths and dataset length differ")
    source_groups = np.asarray(
        [normalized_source_group(path) for path in paths],
        dtype=object,
    )
    if any(not str(value).strip() for value in source_groups.tolist()):
        raise ValueError("B6 source grouping produced an empty identifier")
    source_group_count = int(np.unique(source_groups).size)
    if source_group_count < MIN_SOURCE_GROUPS:
        raise ValueError(
            f"Locked B6 source support is too small: {source_group_count}"
        )
    _assert_output_outside_train(output_dir, dataset)
    output_dir.mkdir(parents=True, exist_ok=True)
    architecture = architecture_self_check()
    preflight: Dict[str, object] = {
        "schema_version": 1,
        "mode": "dinov3_classconditional_deepsets_train_only_oof",
        "protocol_id": PROTOCOL_ID,
        "data": str(data_path),
        "data_sha256": data_sha256,
        "checkpoint": str(checkpoint_path),
        "checkpoint_sha256": checkpoint_sha256,
        "output_dir": str(output_dir),
        "class_names": [str(name) for name in class_names],
        "train_samples": int(len(dataset)),
        "source_groups": source_group_count,
        "model_name": MODEL_NAME,
        "model_contract": model_contract,
        "checkpoint_contract": checkpoint_contract,
        "architecture_contract": architecture,
        "locked_protocol": _locked_protocol_payload(),
        "estimated_cache_bytes": int(
            EXPECTED_TRAIN_SAMPLES
            * (
                PATCH_TOKENS * TOKEN_WIDTH * np.dtype(CACHE_DTYPE).itemsize
                + GLOBAL_LOGIT_WIDTH * np.dtype(np.float32).itemsize
                + np.dtype(np.int64).itemsize
            )
        ),
        "representation_scope": (
            "matched_incremental_readiness_on_full_train_fitted_b2_encoder;"
            "not_unbiased_new_source_generalization"
        ),
        "train_split_used": True,
        "validation_split_used": False,
        "test_split_used": False,
        "validation_dataset_constructed": False,
        "test_dataset_constructed": False,
        "raw_dataset_modified": False,
    }
    _atomic_json(output_dir / "preflight.json", preflight)
    if bool(args.preflight_only):
        return preflight
    device = _resolve_device(str(args.device or ""))
    model.to(device=device, dtype=torch.float32)
    start = time.perf_counter()
    cache = _load_or_extract_cache(
        model=model,
        dataset=dataset,
        paths=paths,
        output_dir=output_dir,
        device=device,
        batch_size=int(args.batch_size),
        workers=int(args.workers),
    )
    del model
    if device.type == "cuda":
        torch.cuda.empty_cache()
    labels = np.asarray(cache["labels"], dtype=np.int64)
    if _labels_sha256(labels) != str(
        cache["manifest"]["labels_content_sha256"]
    ):
        raise ValueError("B6 cached label content hash mismatch")
    if set(np.unique(labels).tolist()) != set(range(GLOBAL_LOGIT_WIDTH)):
        raise ValueError(
            "B6 cached labels must contain exactly canonical classes 0..4"
        )
    assignments, global_fold_rows = assign_global_source_folds(
        labels,
        source_groups,
        folds=FOLDS,
        seed=SEED,
    )
    pair_rows, fold_metric_rows, prediction_rows = run_deepsets_oof(
        token_cache=np.asarray(cache["tokens"]),
        global_logits=np.asarray(cache["global_logits"]),
        labels=labels,
        source_groups=source_groups,
        fold_assignments=assignments,
        paths=paths,
        output_dir=output_dir,
        device=device,
    )
    assignment_rows = [
        {
            "sample_index": int(index),
            "image_path": str(paths[index]),
            "source_group": str(source_groups[index]),
            "target_index": int(labels[index]),
            "fold": int(assignments[index]),
        }
        for index in range(labels.size)
    ]
    _write_csv(output_dir / "global_fold_assignments.csv", assignment_rows)
    _write_csv(output_dir / "pair_fold_metrics.csv", fold_metric_rows)
    _write_csv(output_dir / "pair_metrics.csv", pair_rows)
    _write_csv(output_dir / "train_oof_pair_predictions.csv", prediction_rows)
    cache_finite = bool(
        _array_all_finite(np.asarray(cache["tokens"]))
        and _array_all_finite(np.asarray(cache["global_logits"]))
    )
    readiness = assess_deepsets_readiness(
        pair_rows=pair_rows,
        fold_metric_rows=fold_metric_rows,
        global_fold_rows=global_fold_rows,
        train_samples=int(labels.size),
        source_groups=source_group_count,
        architecture=architecture,
        maximum_pool_parity_error=float(
            cache["manifest"]["maximum_pool_parity_error"]
        ),
        cache_finite=cache_finite,
    )
    summary: Dict[str, object] = {
        **preflight,
        "elapsed_seconds": float(time.perf_counter() - start),
        "device": str(device),
        "cache_manifest": dict(cache["manifest"]),
        "global_folds": global_fold_rows,
        "pair_results": pair_rows,
        "readiness": readiness,
        "artifacts": {
            "preflight": str(output_dir / "preflight.json"),
            "cache_manifest": str(output_dir / CACHE_MANIFEST_FILENAME),
            "global_fold_assignments": str(
                output_dir / "global_fold_assignments.csv"
            ),
            "pair_fold_metrics": str(output_dir / "pair_fold_metrics.csv"),
            "pair_metrics": str(output_dir / "pair_metrics.csv"),
            "oof_predictions": str(output_dir / "train_oof_pair_predictions.csv"),
            "fold_readouts": str(output_dir / "fold_readouts"),
        },
    }
    _atomic_json(output_dir / "summary.json", summary)
    return summary


def main(argv: Optional[Sequence[str]] = None) -> int:
    args = _parse_args(argv)
    summary = run_precheck(args)
    readiness = summary.get("readiness")
    if isinstance(readiness, Mapping):
        output = {
            "deepsets_patch_distribution_ready": bool(
                readiness.get("deepsets_patch_distribution_ready", False)
            ),
            "failed_checks": list(readiness.get("failed_checks", [])),
            "observed": dict(readiness.get("observed", {})),
            "validation_split_used": False,
            "test_split_used": False,
            "output_dir": str(Path(args.output_dir).resolve()),
        }
    else:
        output = {
            "preflight_only": bool(args.preflight_only),
            "train_samples": int(summary.get("train_samples", 0)),
            "validation_split_used": False,
            "test_split_used": False,
            "output_dir": str(Path(args.output_dir).resolve()),
        }
    print(json.dumps(output, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
