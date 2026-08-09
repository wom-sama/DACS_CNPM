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
from pathlib import Path
from typing import Any, Mapping, Sequence

import numpy as np
import torch
from safetensors.torch import load_file
from torch import Tensor, nn
from torch.nn import functional as F

from trkh.data.dataset import ClassificationFolderDataset
from trkh.models.dinov3_multidepth_convpass_b21 import DIRECT_MODE
from trkh.tools import run_dinov3_convpass_b21_train_fold as b21
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
    parser.add_argument("--control-checkpoint", type=Path, required=True)
    parser.add_argument("--robust-checkpoint", type=Path, required=True)
    parser.add_argument("--midpoint-checkpoint", type=Path, required=True)
    parser.add_argument("--preflight-only", action="store_true")
    return parser.parse_args(argv)


def _source_hashes(repo: Path) -> dict[str, str]:
    paths = {
        "runner": Path("trkh/tools/run_b43_progressive_function_merge.py"),
        "primitive": Path("trkh/training/progressive_task_vector.py"),
        "model": Path("trkh/models/dinov3_multidepth_convpass_b21.py"),
        "model_builder": Path("trkh/tools/run_dinov3_convpass_b21_train_fold.py"),
        "relighting": Path("trkh/training/relighting.py"),
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
    positions = _select_calibration_positions(assignment)
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


def main(argv: Sequence[str] | None = None) -> int:
    args = _args(argv)
    if not args.preflight_only:
        raise B43ContractError(
            "Only the metric-free B43 preflight is implemented; formal fold 4 is not authorized."
        )
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
    payload = _preflight(args, repo=repo, git=git)
    artifact = output / "preflight.json"
    b21._atomic_json(artifact, payload)
    print(json.dumps({"passed": payload["passed"], "checks": payload["checks"]}, indent=2))
    return 0 if payload["passed"] else 2


if __name__ == "__main__":
    raise SystemExit(main())
