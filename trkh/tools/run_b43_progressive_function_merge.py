"""B43 metric-free preflight for progressive function-space consolidation.

This preflight consumes only fixed TRAIN-side images and B41 endpoint states.
It trains the elementwise task-vector coefficients of the final DINO block to
verify activation matching, materialization, RNG isolation, and VRAM safety.
It intentionally computes no classifier logits, predictions, or metrics.
"""

from __future__ import annotations

import argparse
import copy
import gc
import hashlib
import json
import os
from pathlib import Path
from typing import Any, Mapping, Sequence

import numpy as np
import torch
from safetensors.torch import load_file, save_file
from torch import Tensor, nn
from torch.nn import functional as F

from trkh.data.dataset import ClassificationFolderDataset
from trkh.models.dinov3_multidepth_convpass_b21 import DIRECT_MODE
from trkh.tools import run_b36_head_first_dino_fold as b36
from trkh.tools import run_b38_relit_bicar_fold2 as b38
from trkh.tools import run_b39_stratified_pair_bicar_fold3 as b39
from trkh.tools import run_b41_backbone_routed_pair_mean_fold3 as b41
from trkh.tools import run_dinov3_convpass_b21_train_fold as b21
from trkh.tools.average_checkpoints import average_state_dicts
from trkh.training.progressive_task_vector import ElementwiseTaskVectorBlock
from trkh.training.relighting import relight_luminance


PROTOCOL_ID = "TRKH_PRETRAINED_CLASSF_B43_PROGRESSIVE_FUNCTION_MERGE_20260809"
SOURCE_FOLD = 3
SEED = 20260809
BLOCK_INDEX = 11
BASES_PER_CLASS = 2
CAPTURE_BATCH_SIZE = 5
COEFFICIENT_INIT = 0.5
PREFLIGHT_STEPS = 25
LEARNING_RATE = 0.01
RELIGHT_BRIGHTNESS = 0.25
RELIGHT_CONTRAST = 0.10
MAX_FINAL_TO_INITIAL_LOSS = 0.90
MAX_REPLAY_ABS = 1e-5
MIN_COEFFICIENT_CHANGE = 1e-6
MAX_PEAK_CUDA_BYTES = 4 * 1024**3

FORMAL_FOLD = 4
FORMAL_SEED = 42
FORMAL_EPOCHS = 9
FORMAL_SCHEDULER_HORIZON = 30
FORMAL_START_EPOCH = 3
FORMAL_BASES_PER_CLASS = 16
FORMAL_LAYER_EPOCHS = 10
FORMAL_CAPTURE_BATCH_SIZE = 16
FORMAL_PER_SOURCE_BATCH_SIZE = 8
MAX_FORMAL_LAYER_FINAL_TO_INITIAL = 1.0
MAX_FORMAL_CUDA_BYTES = 6 * 1024**3
FORMAL_PREFLIGHT_SHA256 = "2d9f183465ad5e7ab80b80aadf095ed54cdbac3dfc147f569c27391096eabd86"
FORMAL_PREFLIGHT_HEAD = "db41aefd974d75af1d54244eae8fdef6442f05d0"

INITIAL_PRIMARY_STATE_SHA256 = "5f8f1073f5e1394b9e8cf8aff6e03e985ca22225e2678a59dd550cafdaea34f8"
INITIAL_FULL_STATE_SHA256 = "787bc0d96bb502159e52f43e9e87111a76419f18156fb2c60491bd591d3023fb"
CONTROL_FILE_SHA256 = "233ee2e3ce2d309ed47ae8cb2d6bd15c1ff40a1373964e4f031239a4448769d1"
CONTROL_STATE_SHA256 = "cde14be9c29d73625b5b9ac35902bc3951be0ab005caef91ef71bbd054cee00c"
ROBUST_FILE_SHA256 = "24c36e4208994f17f9ba147e16d7d2b1cd8aa9f8a1ee458cdef0d3dbc945e761"
ROBUST_STATE_SHA256 = "4ae9aff2cec524370d29a07d267ddc28cc60089b719bce74a81697e11a3a48f9"
MIDPOINT_FILE_SHA256 = "cd9354c55589956e2c2f05719c6dfe056f2d8ded6e907d3a03d0acb1c901ebea"
MIDPOINT_STATE_SHA256 = "1200b0a712dcc838c67800de50f7757e444fe15e7ada392a7a560e0cb389094c"


class B43ContractError(RuntimeError):
    pass


def _args(argv: Sequence[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--assignment-csv", type=Path, required=True)
    parser.add_argument("--dino-weight", type=Path, required=True)
    parser.add_argument("--control-checkpoint", type=Path)
    parser.add_argument("--robust-checkpoint", type=Path)
    parser.add_argument("--midpoint-checkpoint", type=Path)
    parser.add_argument("--preflight-artifact", type=Path)
    parser.add_argument("--preflight-only", action="store_true")
    return parser.parse_args(argv)


def _source_hashes(repo: Path) -> dict[str, str]:
    paths = {
        "runner": Path("trkh/tools/run_b43_progressive_function_merge.py"),
        "primitive": Path("trkh/training/progressive_task_vector.py"),
        "model": Path("trkh/models/dinov3_multidepth_convpass_b21.py"),
        "model_builder": Path("trkh/tools/run_dinov3_convpass_b21_train_fold.py"),
        "endpoint_objective": Path("trkh/tools/run_b41_backbone_routed_pair_mean_fold3.py"),
        "endpoint_formal": Path("trkh/tools/run_b39_stratified_pair_bicar_fold3.py"),
        "evaluator": Path("trkh/tools/run_b36_head_first_dino_fold.py"),
        "relighting": Path("trkh/training/relighting.py"),
        "averaging": Path("trkh/tools/average_checkpoints.py"),
        "decision": Path("docs/TRKH_PRETRAINED_CLASSF_REFOCUS_20260802.md"),
        "primitive_test": Path("tests/test_progressive_task_vector.py"),
        "runner_test": Path("tests/test_run_b43_progressive_function_merge.py"),
    }
    return {name: b21.sha256_file(repo / path) for name, path in paths.items()}


def _selection_key(relative_path: str) -> str:
    return hashlib.sha256(
        f"{PROTOCOL_ID}|calibration|{relative_path}".encode("utf-8")
    ).hexdigest()


def _select_calibration_positions(
    assignment: Mapping[str, Any],
    *,
    excluded_fold: int = SOURCE_FOLD,
    bases_per_class: int = BASES_PER_CLASS,
) -> list[int]:
    labels = np.asarray(assignment["labels"], dtype=np.int64)
    folds = np.asarray(assignment["folds"], dtype=np.int64)
    paths = [str(value) for value in assignment["relative_paths"]]
    selected: list[int] = []
    for label in range(len(b21.EXPECTED_CLASS_NAMES)):
        eligible = [
            index
            for index in range(len(paths))
            if int(labels[index]) == label and int(folds[index]) != int(excluded_fold)
        ]
        eligible.sort(key=lambda index: (_selection_key(paths[index]), paths[index]))
        if len(eligible) < int(bases_per_class):
            raise B43ContractError(
                f"Class {label} has only {len(eligible)} eligible calibration rows."
            )
        selected.extend(eligible[: int(bases_per_class)])
    return selected


def _build_calibration_views(
    train_root: Path,
    assignment: Mapping[str, Any],
    *,
    excluded_fold: int = SOURCE_FOLD,
    bases_per_class: int = BASES_PER_CLASS,
) -> tuple[Tensor, Tensor, list[dict[str, Any]]]:
    _train_transform, eval_transform = b21.build_transforms()
    dataset = ClassificationFolderDataset(
        train_root,
        b21.EXPECTED_CLASS_NAMES,
        transform=eval_transform,
        split="train_b43_calibration",
    )
    mapped_indices = b21.map_dataset_indices(
        dataset,
        data_root=train_root.parent,
        assignment_paths=assignment["relative_paths"],
        assignment_labels=np.asarray(assignment["labels"], dtype=np.int64),
    )
    positions = _select_calibration_positions(
        assignment,
        excluded_fold=excluded_fold,
        bases_per_class=bases_per_class,
    )
    clean_rows: list[Tensor] = []
    polarities: list[float] = []
    records: list[dict[str, Any]] = []
    per_class_rank = {label: 0 for label in range(len(b21.EXPECTED_CLASS_NAMES))}
    for position in positions:
        item = dataset[mapped_indices[position]]
        if not isinstance(item, (tuple, list)) or len(item) < 2:
            raise B43ContractError("Unexpected calibration dataset schema.")
        image = item[0]
        observed_label = int(item[1])
        expected_label = int(assignment["labels"][position])
        if not torch.is_tensor(image) or observed_label != expected_label:
            raise B43ContractError("Calibration tensor/label contract changed.")
        rank = per_class_rank[expected_label]
        per_class_rank[expected_label] += 1
        polarity = -1.0 if rank % 2 == 0 else 1.0
        clean_rows.append(image)
        polarities.append(polarity)
        records.append(
            {
                "relative_path": str(assignment["relative_paths"][position]),
                "class_index": expected_label,
                "fold": int(assignment["folds"][position]),
                "robust_view": "dim" if polarity < 0 else "bright",
            }
        )
    clean = torch.stack(clean_rows, dim=0).contiguous()
    polarity_tensor = torch.tensor(polarities, dtype=clean.dtype)
    robust = relight_luminance(
        clean,
        polarity_tensor,
        brightness=RELIGHT_BRIGHTNESS,
        contrast=RELIGHT_CONTRAST,
    ).contiguous()
    return clean, robust, records


def _block_state(full_state: Mapping[str, Tensor], block_index: int) -> dict[str, Tensor]:
    prefix = f"backbone.blocks.{int(block_index)}."
    result = {
        name[len(prefix) :]: value.detach().cpu().clone()
        for name, value in full_state.items()
        if name.startswith(prefix)
    }
    if not result:
        raise B43ContractError(f"No state found for block {block_index}.")
    return result


def _capture_block_io(
    model: nn.Module,
    images: Tensor,
    *,
    block_index: int,
    batch_size: int,
    device: torch.device,
) -> tuple[Tensor, Tensor, Tensor | None]:
    blocks = getattr(model, "blocks", None)
    if not isinstance(blocks, nn.ModuleList) or int(block_index) >= len(blocks):
        raise B43ContractError("Model block contract changed.")
    captured_inputs: list[Tensor] = []
    captured_outputs: list[Tensor] = []
    captured_rope: Tensor | None = None

    def hook(
        _module: nn.Module,
        args: tuple[Any, ...],
        kwargs: dict[str, Any],
        output: Tensor,
    ) -> None:
        nonlocal captured_rope
        if not args or not torch.is_tensor(args[0]) or not torch.is_tensor(output):
            raise B43ContractError("Unexpected EVA block call schema.")
        captured_inputs.append(args[0].detach().cpu())
        captured_outputs.append(output.detach().cpu())
        rope = kwargs.get("rope")
        if rope is None and len(args) > 1:
            rope = args[1]
        if rope is not None:
            if not torch.is_tensor(rope):
                raise B43ContractError("EVA RoPE is not a tensor.")
            rope_cpu = rope.detach().cpu()
            if captured_rope is None:
                captured_rope = rope_cpu
            elif not torch.equal(captured_rope, rope_cpu):
                raise B43ContractError("EVA RoPE changed between fixed-resolution batches.")

    handle = blocks[int(block_index)].register_forward_hook(hook, with_kwargs=True)
    try:
        model.eval()
        with torch.inference_mode():
            for start in range(0, int(images.size(0)), int(batch_size)):
                model(images[start : start + int(batch_size)].to(device))
    finally:
        handle.remove()
    if not captured_inputs or len(captured_inputs) != len(captured_outputs):
        raise B43ContractError("EVA block hook captured no complete batches.")
    return (
        torch.cat(captured_inputs, dim=0),
        torch.cat(captured_outputs, dim=0),
        captured_rope,
    )


def _capture_all_block_outputs(
    model: nn.Module,
    images: Tensor,
    *,
    batch_size: int,
    device: torch.device,
) -> list[Tensor]:
    blocks = getattr(model, "blocks", None)
    if not isinstance(blocks, nn.ModuleList) or len(blocks) != 12:
        raise B43ContractError("Formal B43 requires the locked 12-block DINO graph.")
    captured: list[list[Tensor]] = [[] for _ in blocks]
    handles = []
    for index, block in enumerate(blocks):
        def hook(
            _module: nn.Module,
            _args: tuple[Any, ...],
            _kwargs: dict[str, Any],
            output: Tensor,
            *,
            block_index: int = index,
        ) -> None:
            if not torch.is_tensor(output):
                raise B43ContractError("Unexpected EVA block output schema.")
            captured[block_index].append(output.detach().cpu())

        handles.append(block.register_forward_hook(hook, with_kwargs=True))
    try:
        model.eval()
        with torch.inference_mode():
            for start in range(0, int(images.size(0)), int(batch_size)):
                model(images[start : start + int(batch_size)].to(device))
    finally:
        for handle in handles:
            handle.remove()
    if any(not rows for rows in captured):
        raise B43ContractError("One or more DINO blocks produced no teacher activation.")
    outputs = [torch.cat(rows, dim=0).contiguous() for rows in captured]
    if any(int(value.size(0)) != int(images.size(0)) for value in outputs):
        raise B43ContractError("Teacher activation row count changed.")
    return outputs


def _load_verified_state(path: Path, *, file_sha: str, state_sha: str) -> dict[str, Tensor]:
    resolved = path.expanduser().resolve(strict=True)
    if b21.sha256_file(resolved) != file_sha:
        raise B43ContractError(f"Checkpoint file hash changed: {resolved}")
    state = load_file(str(resolved), device="cpu")
    if b21.state_sha256(state) != state_sha:
        raise B43ContractError(f"Checkpoint state hash changed: {resolved}")
    return state


def _rng_snapshot() -> dict[str, Any]:
    return {
        "cpu": torch.get_rng_state().clone(),
        "cuda": [state.clone() for state in torch.cuda.get_rng_state_all()],
    }


def _same_rng(left: Mapping[str, Any], right: Mapping[str, Any]) -> bool:
    return torch.equal(left["cpu"], right["cpu"]) and len(left["cuda"]) == len(
        right["cuda"]
    ) and all(torch.equal(a, b) for a, b in zip(left["cuda"], right["cuda"]))


def _max_state_abs(left: Mapping[str, Tensor], right: Mapping[str, Tensor]) -> float:
    if set(left) != set(right):
        raise B43ContractError("Compared state keys differ.")
    maximum = 0.0
    for name in left:
        if left[name].is_floating_point():
            left_value = left[name].detach().float().cpu()
            right_value = right[name].detach().float().cpu()
            maximum = max(
                maximum,
                float((left_value - right_value).abs().max()),
            )
        elif not torch.equal(left[name].detach().cpu(), right[name].detach().cpu()):
            raise B43ContractError(f"Non-floating state differs for {name!r}.")
    return maximum


def _preflight(args: argparse.Namespace, *, repo: Path, git: Mapping[str, Any]) -> dict[str, Any]:
    if not torch.cuda.is_available():
        raise B43ContractError("B43 preflight requires CUDA.")
    device = torch.device("cuda")
    b21._seed_all(SEED)
    torch.cuda.synchronize()
    rng_before = _rng_snapshot()
    torch.cuda.reset_peak_memory_stats()

    assignment = b21.read_locked_assignment(args.assignment_csv)
    train_root = b21.read_locked_train_root(b21.EXPECTED_DATA_YAML)
    clean, robust, records = _build_calibration_views(train_root, assignment)
    clean_count = int(clean.size(0))
    robust_count = int(robust.size(0))

    model = b21.build_arm_model(args.dino_weight, DIRECT_MODE)
    model.set_mode(DIRECT_MODE)
    if (
        b21.state_sha256(model.state_dict(), exclude_adapters=True)
        != INITIAL_PRIMARY_STATE_SHA256
        or b21.state_sha256(model.state_dict()) != INITIAL_FULL_STATE_SHA256
    ):
        raise B43ContractError("Exact B41 common initialization changed.")
    base_block = copy.deepcopy(model.blocks[BLOCK_INDEX]).eval().requires_grad_(False)

    control_state = _load_verified_state(
        args.control_checkpoint,
        file_sha=CONTROL_FILE_SHA256,
        state_sha=CONTROL_STATE_SHA256,
    )
    control_block_state = _block_state(control_state, BLOCK_INDEX)
    model.load_state_dict(control_state, strict=True)
    model.to(device).eval()
    _control_input, control_target, control_rope = _capture_block_io(
        model,
        clean,
        block_index=BLOCK_INDEX,
        batch_size=CAPTURE_BATCH_SIZE,
        device=device,
    )
    del control_state, _control_input

    robust_state = _load_verified_state(
        args.robust_checkpoint,
        file_sha=ROBUST_FILE_SHA256,
        state_sha=ROBUST_STATE_SHA256,
    )
    robust_block_state = _block_state(robust_state, BLOCK_INDEX)
    model.load_state_dict(robust_state, strict=True)
    _robust_input, robust_target, robust_rope = _capture_block_io(
        model,
        robust,
        block_index=BLOCK_INDEX,
        batch_size=CAPTURE_BATCH_SIZE,
        device=device,
    )
    del robust_state, _robust_input

    midpoint_state = _load_verified_state(
        args.midpoint_checkpoint,
        file_sha=MIDPOINT_FILE_SHA256,
        state_sha=MIDPOINT_STATE_SHA256,
    )
    midpoint_block_state = _block_state(midpoint_state, BLOCK_INDEX)
    model.load_state_dict(midpoint_state, strict=True)
    student_input, midpoint_output, midpoint_rope = _capture_block_io(
        model,
        torch.cat((clean, robust), dim=0),
        block_index=BLOCK_INDEX,
        batch_size=CAPTURE_BATCH_SIZE,
        device=device,
    )
    del midpoint_state, model, clean, robust
    gc.collect()
    torch.cuda.empty_cache()

    ropes = [rope for rope in (control_rope, robust_rope, midpoint_rope) if rope is not None]
    if len(ropes) not in (0, 3) or any(not torch.equal(ropes[0], rope) for rope in ropes[1:]):
        raise B43ContractError("Teacher/student RoPE contract differs.")
    rope_device = None if midpoint_rope is None else midpoint_rope.to(device)
    target = torch.cat((control_target, robust_target), dim=0).to(device)
    student_input = student_input.to(device)
    midpoint_output = midpoint_output.to(device)

    merger = ElementwiseTaskVectorBlock(
        base_block,
        control_block_state,
        robust_block_state,
        coefficient_init=COEFFICIENT_INIT,
    ).to(device)
    merger.base_block.eval()
    midpoint_state_error = _max_state_abs(
        merger.materialize_state_dict(),
        midpoint_block_state,
    )
    midpoint_replay_block = copy.deepcopy(base_block).to(device).eval().requires_grad_(False)
    midpoint_replay_block.load_state_dict(midpoint_block_state, strict=True)
    with torch.no_grad():
        initial_output = merger(student_input, rope=rope_device)
        checkpoint_output = midpoint_replay_block(student_input, rope=rope_device)
        midpoint_replay_error = float((initial_output - checkpoint_output).abs().max())
        midpoint_context_error = float((initial_output - midpoint_output).abs().max())
        initial_loss = float(F.mse_loss(initial_output.float(), target.float()))
    del midpoint_replay_block, checkpoint_output
    if not math_is_finite_positive(initial_loss):
        raise B43ContractError(f"Initial activation loss is invalid: {initial_loss}")

    optimizer = torch.optim.Adam(merger.coefficient_parameters(), lr=LEARNING_RATE)
    losses: list[float] = []
    for _step in range(PREFLIGHT_STEPS):
        optimizer.zero_grad(set_to_none=True)
        predicted = merger(student_input, rope=rope_device)
        loss = F.mse_loss(predicted.float(), target.float())
        if not bool(torch.isfinite(loss).item()):
            raise B43ContractError("Non-finite B43 activation loss.")
        loss.backward()
        optimizer.step()
        losses.append(float(loss.detach().cpu()))
    with torch.no_grad():
        final_output = merger(student_input, rope=rope_device)
        final_loss = float(F.mse_loss(final_output.float(), target.float()))
    materialized_state = merger.materialize_state_dict()
    replay_block = copy.deepcopy(base_block).to(device).eval().requires_grad_(False)
    replay_block.load_state_dict(materialized_state, strict=True)
    with torch.no_grad():
        replay_output = replay_block(student_input, rope=rope_device)
    materialized_replay_error = float((final_output - replay_output).abs().max())
    coefficient_summary = merger.coefficient_summary()
    coefficient_change = max(
        abs(float(coefficient_summary[endpoint][bound]) - COEFFICIENT_INIT)
        for endpoint in ("endpoint_a", "endpoint_b")
        for bound in ("min", "max")
    )

    torch.cuda.synchronize()
    peak_cuda_bytes = int(torch.cuda.max_memory_allocated())
    rng_after = _rng_snapshot()
    rng_preserved = _same_rng(rng_before, rng_after)
    class_counts = {
        str(label): sum(int(row["class_index"]) == label for row in records)
        for label in range(len(b21.EXPECTED_CLASS_NAMES))
    }
    view_counts = {
        "clean": clean_count,
        "robust": robust_count,
        "dim": sum(row["robust_view"] == "dim" for row in records),
        "bright": sum(row["robust_view"] == "bright" for row in records),
    }
    checks = {
        "source_rows_balanced": clean_count == robust_count == 10,
        "classes_balanced": set(class_counts.values()) == {BASES_PER_CLASS},
        "robust_polarities_balanced": view_counts["dim"] == view_counts["bright"] == 5,
        "held_source_fold_excluded": all(int(row["fold"]) != SOURCE_FOLD for row in records),
        "initial_midpoint_state_replay": midpoint_state_error <= MAX_REPLAY_ABS,
        "initial_midpoint_function_replay": midpoint_replay_error <= MAX_REPLAY_ABS,
        "activation_loss_reduction": final_loss <= initial_loss * MAX_FINAL_TO_INITIAL_LOSS,
        "coefficients_changed_finitely": coefficient_change >= MIN_COEFFICIENT_CHANGE
        and all(
            np.isfinite(float(coefficient_summary[endpoint][field]))
            for endpoint in ("endpoint_a", "endpoint_b")
            for field in ("mean", "min", "max")
        ),
        "materialized_function_replay": materialized_replay_error <= MAX_REPLAY_ABS,
        "rng_preserved": rng_preserved,
        "cuda_memory_safe": peak_cuda_bytes <= MAX_PEAK_CUDA_BYTES,
    }
    return {
        "schema_version": 1,
        "protocol_id": PROTOCOL_ID,
        "git": dict(git),
        "source_hashes": _source_hashes(repo),
        "passed": bool(all(checks.values())),
        "checks": checks,
        "train": False,
        "validation": False,
        "test": False,
        "classifier_logits_or_metrics_computed": False,
        "scope": {
            "purpose": "mechanics_only_final_block_activation_replay",
            "source_endpoint_fold": SOURCE_FOLD,
            "block_index": BLOCK_INDEX,
            "calibration_records": records,
            "class_counts": class_counts,
            "view_counts": view_counts,
            "calibration_tensor_sha256": b21.state_sha256(
                {"student_input": student_input.detach().cpu(), "target": target.detach().cpu()}
            ),
        },
        "optimization": {
            "coefficient_init": COEFFICIENT_INIT,
            "optimizer": "Adam",
            "learning_rate": LEARNING_RATE,
            "steps": PREFLIGHT_STEPS,
            "loss": "mean_squared_block_activation",
            "initial_loss": initial_loss,
            "step_losses": losses,
            "final_loss": final_loss,
            "final_to_initial": final_loss / initial_loss,
            "coefficient_summary": coefficient_summary,
            "maximum_coefficient_change_from_init": coefficient_change,
        },
        "replay": {
            "midpoint_state_max_abs": midpoint_state_error,
            "midpoint_function_max_abs": midpoint_replay_error,
            "midpoint_hooked_context_max_abs_diagnostic": midpoint_context_error,
            "materialized_function_max_abs": materialized_replay_error,
        },
        "resources": {
            "peak_cuda_bytes": peak_cuda_bytes,
            "peak_cuda_gib": peak_cuda_bytes / float(1024**3),
            "maximum_cuda_bytes": MAX_PEAK_CUDA_BYTES,
        },
        "thresholds": {
            "maximum_final_to_initial_loss": MAX_FINAL_TO_INITIAL_LOSS,
            "maximum_replay_abs": MAX_REPLAY_ABS,
            "minimum_coefficient_change": MIN_COEFFICIENT_CHANGE,
        },
        "checkpoint_created": False,
        "next_permission": (
            "implement_locked_progressive_all_block_fold4_screen"
            if all(checks.values())
            else "close_or_repair_only_a_demonstrated_b43_implementation_defect"
        ),
    }


def math_is_finite_positive(value: float) -> bool:
    return bool(np.isfinite(float(value)) and float(value) > 0.0)


def _read_formal_preflight(path: Path) -> dict[str, str]:
    resolved = path.expanduser().resolve(strict=True)
    payload = json.loads(resolved.read_text(encoding="utf-8"))
    checks = {
        "file_sha256": b21.sha256_file(resolved) == FORMAL_PREFLIGHT_SHA256,
        "protocol": payload.get("protocol_id") == PROTOCOL_ID,
        "passed": payload.get("passed") is True,
        "source_commit": payload.get("git", {}).get("head") == FORMAL_PREFLIGHT_HEAD,
        "scope": payload.get("train") is False
        and payload.get("validation") is False
        and payload.get("test") is False,
        "no_metrics": payload.get("classifier_logits_or_metrics_computed") is False,
        "no_checkpoint": payload.get("checkpoint_created") is False,
    }
    if not all(checks.values()):
        raise B43ContractError(f"B43 formal preflight is not accepted: {checks}")
    return {"path": str(resolved), "sha256": FORMAL_PREFLIGHT_SHA256}


def _balanced_source_indices(
    source_rows: int,
    start: int,
    stop: int,
    *,
    device: torch.device,
) -> Tensor:
    if not 0 <= int(start) < int(stop) <= int(source_rows):
        raise ValueError("Invalid source-balanced batch range.")
    clean = torch.arange(int(start), int(stop), device=device, dtype=torch.long)
    robust = clean + int(source_rows)
    return torch.cat((clean, robust), dim=0)


@torch.no_grad()
def _activation_loss(
    merger: ElementwiseTaskVectorBlock,
    student_input: Tensor,
    target: Tensor,
    *,
    rope: Tensor | None,
    source_rows: int,
) -> float:
    total = 0.0
    count = 0
    for start in range(0, int(source_rows), FORMAL_PER_SOURCE_BATCH_SIZE):
        stop = min(int(source_rows), start + FORMAL_PER_SOURCE_BATCH_SIZE)
        indices = _balanced_source_indices(
            source_rows,
            start,
            stop,
            device=student_input.device,
        )
        predicted = merger(student_input.index_select(0, indices), rope=rope)
        rows = int(indices.numel())
        total += float(
            F.mse_loss(
                predicted.float(),
                target.index_select(0, indices).float(),
            ).cpu()
        ) * rows
        count += rows
    if count != int(student_input.size(0)):
        raise B43ContractError("Formal activation-loss batching lost rows.")
    return total / float(count)


def _load_checkpoint_record(record: Mapping[str, Any]) -> dict[str, Tensor]:
    path = Path(str(record["path"])).expanduser().resolve(strict=True)
    if b21.sha256_file(path) != str(record["sha256"]):
        raise B43ContractError(f"Fresh endpoint checkpoint file changed: {path}")
    state = load_file(str(path), device="cpu")
    if b21.state_sha256(state) != str(record["state_sha256"]):
        raise B43ContractError(f"Fresh endpoint checkpoint state changed: {path}")
    return state


def _replace_block_state(
    full_state: dict[str, Tensor],
    block_index: int,
    block_state: Mapping[str, Tensor],
) -> None:
    prefix = f"backbone.blocks.{int(block_index)}."
    expected = {name[len(prefix) :] for name in full_state if name.startswith(prefix)}
    if set(block_state) != expected:
        raise B43ContractError(f"Materialized block {block_index} state keys changed.")
    for name, value in block_state.items():
        full_state[prefix + name] = value.detach().cpu().contiguous().clone()


def _save_safetensors_atomic(path: Path, state: Mapping[str, Tensor]) -> dict[str, Any]:
    temporary = path.with_suffix(path.suffix + ".tmp")
    save_file(
        {name: value.detach().cpu().contiguous() for name, value in state.items()},
        str(temporary),
    )
    os.replace(temporary, path)
    return {
        "mode": DIRECT_MODE,
        "path": str(path.resolve()),
        "sha256": b21.sha256_file(path),
        "state_sha256": b21.state_sha256(state),
    }


def _progressively_consolidate(
    *,
    dino_weight: Path,
    control_state: Mapping[str, Tensor],
    robust_state: Mapping[str, Tensor],
    clean: Tensor,
    robust: Tensor,
) -> tuple[dict[str, Tensor], dict[str, Any]]:
    device = torch.device("cuda")
    base_model = b21.build_arm_model(dino_weight, DIRECT_MODE)
    base_model.set_mode(DIRECT_MODE)
    if (
        b21.state_sha256(base_model.state_dict(), exclude_adapters=True)
        != INITIAL_PRIMARY_STATE_SHA256
        or b21.state_sha256(base_model.state_dict()) != INITIAL_FULL_STATE_SHA256
    ):
        raise B43ContractError("Formal B43 common initialization changed.")

    candidate_state, average_stats = average_state_dicts(
        [control_state, robust_state],
        [COEFFICIENT_INIT, COEFFICIENT_INIT],
    )
    if int(average_stats["copied_different_non_float"]) != 0:
        raise B43ContractError("Fresh endpoints disagree in non-floating state.")

    student = b21.build_arm_model(dino_weight, DIRECT_MODE)
    student.set_mode(DIRECT_MODE)
    student.load_state_dict(control_state, strict=True)
    student.to(device).eval().requires_grad_(False)
    control_targets = _capture_all_block_outputs(
        student,
        clean,
        batch_size=FORMAL_CAPTURE_BATCH_SIZE,
        device=device,
    )
    student.load_state_dict(robust_state, strict=True)
    robust_targets = _capture_all_block_outputs(
        student,
        robust,
        batch_size=FORMAL_CAPTURE_BATCH_SIZE,
        device=device,
    )
    student.load_state_dict(candidate_state, strict=True)

    all_views = torch.cat((clean, robust), dim=0).contiguous()
    source_rows = int(clean.size(0))
    if source_rows != int(robust.size(0)) or source_rows != 80:
        raise B43ContractError("Formal B43 requires 80 clean and 80 robust views.")
    teacher_target_sha256 = b21.state_sha256(
        {
            **{f"control.{index}": value for index, value in enumerate(control_targets)},
            **{f"robust.{index}": value for index, value in enumerate(robust_targets)},
        }
    )
    layer_records: list[dict[str, Any]] = []
    torch.cuda.reset_peak_memory_stats()

    for block_index in range(len(base_model.blocks)):
        student_input, _old_output, rope = _capture_block_io(
            student,
            all_views,
            block_index=block_index,
            batch_size=FORMAL_CAPTURE_BATCH_SIZE,
            device=device,
        )
        del _old_output
        target = torch.cat(
            (control_targets[block_index], robust_targets[block_index]),
            dim=0,
        ).contiguous()
        student_input = student_input.to(device)
        target = target.to(device)
        rope_device = None if rope is None else rope.to(device)
        merger = ElementwiseTaskVectorBlock(
            base_model.blocks[block_index],
            _block_state(control_state, block_index),
            _block_state(robust_state, block_index),
            coefficient_init=COEFFICIENT_INIT,
        ).to(device)
        merger.base_block.eval()

        initial_state_error = _max_state_abs(
            merger.materialize_state_dict(),
            _block_state(candidate_state, block_index),
        )
        initial_loss = _activation_loss(
            merger,
            student_input,
            target,
            rope=rope_device,
            source_rows=source_rows,
        )
        optimizer = torch.optim.Adam(
            merger.coefficient_parameters(),
            lr=LEARNING_RATE,
        )
        epoch_losses: list[float] = []
        for _epoch in range(FORMAL_LAYER_EPOCHS):
            total = 0.0
            count = 0
            for start in range(0, source_rows, FORMAL_PER_SOURCE_BATCH_SIZE):
                stop = min(source_rows, start + FORMAL_PER_SOURCE_BATCH_SIZE)
                indices = _balanced_source_indices(
                    source_rows,
                    start,
                    stop,
                    device=device,
                )
                optimizer.zero_grad(set_to_none=True)
                predicted = merger(
                    student_input.index_select(0, indices),
                    rope=rope_device,
                )
                loss = F.mse_loss(
                    predicted.float(),
                    target.index_select(0, indices).float(),
                )
                if not bool(torch.isfinite(loss).item()):
                    raise B43ContractError(
                        f"Non-finite B43 loss at block {block_index}."
                    )
                loss.backward()
                optimizer.step()
                rows = int(indices.numel())
                total += float(loss.detach().cpu()) * rows
                count += rows
            epoch_losses.append(total / float(count))

        final_loss = _activation_loss(
            merger,
            student_input,
            target,
            rope=rope_device,
            source_rows=source_rows,
        )
        materialized = merger.materialize_state_dict()
        replay_block = copy.deepcopy(base_model.blocks[block_index]).to(device).eval()
        replay_block.requires_grad_(False)
        replay_block.load_state_dict(materialized, strict=True)
        replay_indices = _balanced_source_indices(
            source_rows,
            0,
            FORMAL_PER_SOURCE_BATCH_SIZE,
            device=device,
        )
        with torch.no_grad():
            functional_output = merger(
                student_input.index_select(0, replay_indices),
                rope=rope_device,
            )
            replay_output = replay_block(
                student_input.index_select(0, replay_indices),
                rope=rope_device,
            )
        replay_error = float((functional_output - replay_output).abs().max())
        coefficient_summary = merger.coefficient_summary()
        coefficients_finite = all(
            np.isfinite(float(coefficient_summary[endpoint][field]))
            for endpoint in ("endpoint_a", "endpoint_b")
            for field in ("mean", "min", "max")
        )
        checks = {
            "initial_midpoint_state": initial_state_error <= MAX_REPLAY_ABS,
            "finite_positive_initial_loss": math_is_finite_positive(initial_loss),
            "loss_nonincrease": np.isfinite(final_loss)
            and final_loss <= initial_loss * MAX_FORMAL_LAYER_FINAL_TO_INITIAL,
            "coefficients_finite": coefficients_finite,
            "materialized_replay": replay_error <= MAX_REPLAY_ABS,
        }
        if not all(checks.values()):
            raise B43ContractError(
                f"B43 block {block_index} mechanics failed: {checks}"
            )
        _replace_block_state(candidate_state, block_index, materialized)
        student.blocks[block_index].load_state_dict(materialized, strict=True)
        layer_records.append(
            {
                "block_index": block_index,
                "checks": checks,
                "initial_loss": initial_loss,
                "epoch_losses": epoch_losses,
                "final_loss": final_loss,
                "final_to_initial": final_loss / initial_loss,
                "initial_state_max_abs": initial_state_error,
                "materialized_replay_max_abs": replay_error,
                "coefficient_summary": coefficient_summary,
            }
        )
        del (
            optimizer,
            merger,
            replay_block,
            materialized,
            student_input,
            target,
            functional_output,
            replay_output,
        )
        gc.collect()
        torch.cuda.empty_cache()

    student_state_sha256 = b21.state_sha256(student.state_dict())
    candidate_state_sha256 = b21.state_sha256(candidate_state)
    if student_state_sha256 != candidate_state_sha256:
        raise B43ContractError("Materialized candidate/student states differ.")
    torch.cuda.synchronize()
    peak_cuda_bytes = int(torch.cuda.max_memory_allocated())
    if peak_cuda_bytes > MAX_FORMAL_CUDA_BYTES:
        raise B43ContractError(
            f"B43 consolidation exceeded VRAM contract: {peak_cuda_bytes}."
        )
    telemetry = {
        "method": "progressive_elementwise_task_vector_activation_matching",
        "initial_non_block_state": "fixed_arithmetic_midpoint",
        "block_order": list(range(12)),
        "coefficient_init": COEFFICIENT_INIT,
        "optimizer": "Adam",
        "learning_rate": LEARNING_RATE,
        "layer_epochs": FORMAL_LAYER_EPOCHS,
        "per_source_batch_size": FORMAL_PER_SOURCE_BATCH_SIZE,
        "teacher_target_sha256": teacher_target_sha256,
        "average_stats": average_stats,
        "layers": layer_records,
        "all_layers_passed": all(
            all(record["checks"].values()) for record in layer_records
        ),
        "peak_cuda_bytes": peak_cuda_bytes,
        "peak_cuda_gib": peak_cuda_bytes / float(1024**3),
        "maximum_cuda_bytes": MAX_FORMAL_CUDA_BYTES,
        "candidate_state_sha256": candidate_state_sha256,
    }
    del student, base_model, control_targets, robust_targets, all_views
    gc.collect()
    torch.cuda.empty_cache()
    return candidate_state, telemetry


def _formal(
    args: argparse.Namespace,
    output: Path,
    *,
    repo: Path,
    git: Mapping[str, Any],
) -> dict[str, Any]:
    if args.preflight_artifact is None:
        raise B43ContractError("Formal B43 requires --preflight-artifact.")
    accepted_preflight = _read_formal_preflight(args.preflight_artifact)
    assignment = b21.read_locked_assignment(args.assignment_csv)
    train_root = b21.read_locked_train_root(b21.EXPECTED_DATA_YAML)
    train_fit, train_held, held_paths, fit_labels = b21._build_datasets(
        train_root,
        assignment,
        fold=FORMAL_FOLD,
    )
    class_counts = b39._fit_counts(assignment, fold=FORMAL_FOLD)
    control_training = b21._train_arm(
        mode=DIRECT_MODE,
        dino_weight=args.dino_weight,
        fit_dataset=train_fit,
        held_dataset=train_held,
        held_paths=held_paths,
        fit_labels=fit_labels,
        output_dir=output,
        epochs=FORMAL_EPOCHS,
        scheduler_horizon=FORMAL_SCHEDULER_HORIZON,
        seed=FORMAL_SEED,
        protocol_id=PROTOCOL_ID,
        checkpoint_stem="b43_control_ema",
        state_hash_epochs=(FORMAL_START_EPOCH - 1,),
    )
    objective = b41._objective_factory(class_counts)
    robust_training = b21._train_arm(
        mode=DIRECT_MODE,
        dino_weight=args.dino_weight,
        fit_dataset=train_fit,
        held_dataset=train_held,
        held_paths=held_paths,
        fit_labels=fit_labels,
        output_dir=output,
        epochs=FORMAL_EPOCHS,
        scheduler_horizon=FORMAL_SCHEDULER_HORIZON,
        seed=FORMAL_SEED,
        protocol_id=PROTOCOL_ID,
        checkpoint_stem="b43_backbone_routed_pair_mean_ema",
        auxiliary_loss_fn=objective,
        auxiliary_loss_weight=b41.AUXILIARY_WEIGHT,
        state_hash_epochs=(FORMAL_START_EPOCH - 1,),
    )
    warmup_replay = b38._matched_warmup(control_training, robust_training)
    if not warmup_replay["passed"]:
        raise B43ContractError(
            f"B43 fresh endpoints diverged before auxiliary start: {warmup_replay}"
        )
    control_record = control_training.get("checkpoint")
    robust_record = robust_training.get("checkpoint")
    if not isinstance(control_record, Mapping) or not isinstance(robust_record, Mapping):
        raise B43ContractError("B43 fresh endpoint checkpoint is missing.")
    control_state = _load_checkpoint_record(control_record)
    robust_state = _load_checkpoint_record(robust_record)

    clean, robust, calibration_records = _build_calibration_views(
        train_root,
        assignment,
        excluded_fold=FORMAL_FOLD,
        bases_per_class=FORMAL_BASES_PER_CLASS,
    )
    candidate_state, consolidation = _progressively_consolidate(
        dino_weight=args.dino_weight,
        control_state=control_state,
        robust_state=robust_state,
        clean=clean,
        robust=robust,
    )
    candidate_record = _save_safetensors_atomic(
        output / "b43_progressive_ema.safetensors",
        candidate_state,
    )

    datasets = {
        name: b36._held_subset(
            train_root,
            assignment,
            brightness=brightness,
            contrast=contrast,
            fold=FORMAL_FOLD,
        )
        for name, (brightness, contrast) in b41.CONDITIONS.items()
    }
    control_metrics, _control_features, control_logits = b36._evaluate_checkpoint(
        dino_weight=args.dino_weight,
        checkpoint=Path(str(control_record["path"])),
        expected_state_sha256=str(control_record["state_sha256"]),
        datasets=datasets,
        seed_offset=900,
    )
    robust_metrics, _robust_features, robust_logits = b36._evaluate_checkpoint(
        dino_weight=args.dino_weight,
        checkpoint=Path(str(robust_record["path"])),
        expected_state_sha256=str(robust_record["state_sha256"]),
        datasets=datasets,
        seed_offset=1000,
    )
    candidate_metrics, _candidate_features, candidate_logits = b36._evaluate_checkpoint(
        dino_weight=args.dino_weight,
        checkpoint=Path(str(candidate_record["path"])),
        expected_state_sha256=str(candidate_record["state_sha256"]),
        datasets=datasets,
        seed_offset=1100,
    )
    labels = np.asarray(control_training["held_labels"], dtype=np.int64)
    if not np.array_equal(labels, np.asarray(robust_training["held_labels"], dtype=np.int64)):
        raise B43ContractError("B43 fresh endpoint held-label order changed.")
    control_replay = float(
        np.max(
            np.abs(
                control_logits["clean"].numpy()
                - np.asarray(control_training["held_logits"], dtype=np.float32)
            )
        )
    )
    robust_replay = float(
        np.max(
            np.abs(
                robust_logits["clean"].numpy()
                - np.asarray(robust_training["held_logits"], dtype=np.float32)
            )
        )
    )
    if control_replay != 0.0 or robust_replay != 0.0:
        raise B43ContractError(
            f"B43 endpoint clean replay changed: {control_replay}, {robust_replay}"
        )
    gate = b41._gate(control_metrics, candidate_metrics)
    gate["next_permission"] = (
        "replicate_progressive_function_merge_on_remaining_train_folds"
        if gate["passed"]
        else "close_exact_b43_progressive_function_merge"
    )
    score_path = output / "held_condition_scores.npz"
    arrays: dict[str, np.ndarray] = {"labels": labels}
    for condition in b41.CONDITIONS:
        arrays[f"control_{condition}"] = control_logits[condition].numpy()
        arrays[f"raw_robust_{condition}"] = robust_logits[condition].numpy()
        arrays[f"progressive_{condition}"] = candidate_logits[condition].numpy()
    b36._atomic_npz(score_path, **arrays)
    bright_ema = objective.bright_state.ema_confusion
    dim_ema = objective.dim_state.ema_confusion
    if bright_ema is None or dim_ema is None or objective.active_calls <= 0:
        raise B43ContractError("B43 robust endpoint objective never became active.")

    summary = {
        "schema_version": 1,
        "protocol_id": PROTOCOL_ID,
        "git": dict(git),
        "source_hashes": _source_hashes(repo),
        "accepted_preflight": accepted_preflight,
        "dataset": {
            "fold": FORMAL_FOLD,
            "fit_rows": len(train_fit),
            "held_rows": len(train_held),
            "fit_class_counts": class_counts,
            "train_split_used": True,
            "validation_split_used": False,
            "test_split_used": False,
        },
        "endpoint_schedule": {
            "seed": FORMAL_SEED,
            "epochs": FORMAL_EPOCHS,
            "scheduler_horizon": FORMAL_SCHEDULER_HORIZON,
            "warmup_replay": warmup_replay,
            "objective": b41._objective_contract(),
        },
        "control_training": b21._sanitize_arm_result(control_training),
        "raw_robust_training": b21._sanitize_arm_result(robust_training),
        "calibration": {
            "labels_used_for_balancing_only": True,
            "classification_loss_or_metric_used": False,
            "bases_per_class_per_source": FORMAL_BASES_PER_CLASS,
            "control_views": int(clean.size(0)),
            "robust_views": int(robust.size(0)),
            "dim_views": sum(row["robust_view"] == "dim" for row in calibration_records),
            "bright_views": sum(
                row["robust_view"] == "bright" for row in calibration_records
            ),
            "excluded_fold": FORMAL_FOLD,
            "records": calibration_records,
        },
        "consolidation": consolidation,
        "candidate_checkpoint": candidate_record,
        "auxiliary_state": {
            "active_calls": objective.active_calls,
            "bright_updates": objective.bright_state.updates,
            "dim_updates": objective.dim_state.updates,
            "bright_ema_confusion": bright_ema.detach().float().cpu().tolist(),
            "dim_ema_confusion": dim_ema.detach().float().cpu().tolist(),
        },
        "metrics": {
            "control": control_metrics,
            "raw_robust": robust_metrics,
            "progressive": candidate_metrics,
        },
        "strict_replay_max_abs": {
            "control": control_replay,
            "raw_robust": robust_replay,
        },
        "scores_sha256": b21.sha256_file(score_path),
        "gate": gate,
        "interpretation_guard": (
            "Fresh held TRAIN component fold 4 only. Progressive calibration uses "
            "fit-side eval views and activation MSE, never held labels or classifier "
            "metrics. Validation/test remain closed."
        ),
    }
    b21._atomic_json(output / "summary.json", summary)
    return summary


def main(argv: Sequence[str] | None = None) -> int:
    args = _args(argv)
    args.assignment_csv = args.assignment_csv.expanduser().resolve(strict=True)
    args.dino_weight = b21.logical_absolute_path(args.dino_weight)
    if b21.sha256_file(args.dino_weight) != b21.DINO_SHA256:
        raise B43ContractError("DINOv3 weight changed.")
    output = args.output_dir.expanduser().resolve()
    if output.exists():
        raise FileExistsError(f"Output must not already exist: {output}")
    repo = Path(__file__).resolve().parents[2]
    git = b21._git_snapshot(repo)
    b21._configure_determinism()
    output.mkdir(parents=True, exist_ok=False)
    if args.preflight_only:
        checkpoint_args = (
            args.control_checkpoint,
            args.robust_checkpoint,
            args.midpoint_checkpoint,
        )
        if any(path is None for path in checkpoint_args):
            raise B43ContractError(
                "Preflight requires control, robust, and midpoint checkpoints."
            )
        payload = _preflight(args, repo=repo, git=git)
        artifact = output / "preflight.json"
        b21._atomic_json(artifact, payload)
        print(
            json.dumps(
                {"passed": payload["passed"], "checks": payload["checks"]},
                indent=2,
            )
        )
        return 0 if payload["passed"] else 2
    summary = _formal(args, output, repo=repo, git=git)
    print(json.dumps(summary["gate"], indent=2))
    return 0 if summary["gate"]["passed"] else 2


if __name__ == "__main__":
    raise SystemExit(main())
