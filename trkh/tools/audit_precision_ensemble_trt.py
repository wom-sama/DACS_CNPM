from __future__ import annotations

import argparse
import json
import time
from pathlib import Path
from typing import Dict, List, Mapping, Sequence

import numpy as np
import tensorrt as trt
import torch
from torch.utils.data import DataLoader, Subset

from trkh.core.utils import load_checkpoint, set_seed
from trkh.data.dataset import build_train_collate_fn
from trkh.inference.stream_infer_trt import TensorRTHybridModel
from trkh.models.model import build_model_from_checkpoint
from trkh.models.precision_ensemble import PrecisionEnsembleClassifier
from trkh.tools.apply_frozen_precision_ensemble import _load_frozen_protocol
from trkh.tools.audit_precision_ensemble_checkpoint import (
    _eval_dataset,
    _max_metric_error,
)
from trkh.tools.audit_precision_ensemble_onnx import _write_csv
from trkh.tools.audit_precision_ensemble_readiness import (
    _aligned_arrays,
    _metric_summary,
    _sha256,
    _validate_prediction_row_split,
    apply_precision_ensemble,
)
from trkh.tools.build_precision_ensemble_checkpoint import _verify_file_hash
from trkh.tools.soft_ensemble_predictions import _read_table, _source_path


def _engine_schema(classifier: TensorRTHybridModel) -> Dict[str, object]:
    expected_inputs = {
        "images": [1, 3, 256, 256],
        "image_valid_mask": [1, 256, 256],
        "bbox": [1, 4],
    }
    observed_inputs = {
        name: [int(dim) for dim in classifier.engine.get_tensor_shape(name)]
        for name in classifier.input_names
    }
    if observed_inputs != expected_inputs:
        raise ValueError(
            f"TensorRT input schema mismatch: expected={expected_inputs}, got={observed_inputs}"
        )
    expected_output_names = list(
        PrecisionEnsembleClassifier.deployment_output_names
    )
    if classifier.output_names != expected_output_names:
        raise ValueError(
            "TensorRT precision-ensemble audit outputs differ from the certified "
            f"contract: expected={expected_output_names}, got={classifier.output_names}"
        )
    observed_outputs = {
        name: [int(dim) for dim in classifier.engine.get_tensor_shape(name)]
        for name in classifier.output_names
    }
    invalid_outputs = {
        name: shape for name, shape in observed_outputs.items() if shape != [1, 5]
    }
    if invalid_outputs:
        raise ValueError(
            "TensorRT precision-ensemble outputs must all have shape [1,5]: "
            f"{invalid_outputs}"
        )
    return {
        "inputs": observed_inputs,
        "input_dtypes": {
            name: str(classifier.input_dtypes[name])
            for name in classifier.input_names
        },
        "outputs": observed_outputs,
        "output_dtypes": {
            name: str(classifier.output_dtypes[name])
            for name in classifier.output_names
        },
    }


def _resolve_audit_indices(
    *,
    dataset_size: int,
    max_samples: int,
    sample_indices: Sequence[int],
) -> List[int]:
    dataset_size = int(dataset_size)
    max_samples = int(max_samples)
    if dataset_size <= 0:
        raise ValueError("TensorRT audit dataset must not be empty.")
    if max_samples < 0:
        raise ValueError("max_samples must be nonnegative.")
    if sample_indices and max_samples > 0:
        raise ValueError("Use either --sample-indices or --max-samples, not both.")
    if sample_indices:
        resolved = [int(value) for value in sample_indices]
        if len(set(resolved)) != len(resolved):
            raise ValueError("sample_indices must be unique.")
        invalid = [value for value in resolved if not 0 <= value < dataset_size]
        if invalid:
            raise ValueError(
                f"sample_indices outside [0, {dataset_size - 1}]: {invalid}"
            )
        return resolved
    if max_samples > 0:
        return list(range(min(max_samples, dataset_size)))
    return list(range(dataset_size))


def _parse_sample_indices(value: str) -> List[int]:
    tokens = [token.strip() for token in str(value or "").split(",")]
    return [int(token) for token in tokens if token]


def audit_trt(
    *,
    checkpoint_path: Path,
    engine_path: Path,
    frozen_summary_path: Path,
    data_yaml: Path,
    output_dir: Path,
    expected_checkpoint_sha256: str,
    expected_engine_sha256: str,
    expected_summary_sha256: str,
    max_samples: int,
    probability_tolerance: float,
    cuda_device: int,
    sample_indices: Sequence[int] = (),
) -> Dict[str, object]:
    if not torch.cuda.is_available():
        raise RuntimeError("TensorRT parity requires CUDA.")
    output_dir = Path(output_dir)
    if output_dir.exists() and any(output_dir.iterdir()):
        raise FileExistsError(f"Audit output directory is nonempty: {output_dir}")
    checkpoint_hash = _verify_file_hash(
        checkpoint_path,
        expected_checkpoint_sha256,
        name="expected_checkpoint_sha256",
    )
    engine_hash = _verify_file_hash(
        engine_path,
        expected_engine_sha256,
        name="expected_engine_sha256",
    )
    summary_hash = _verify_file_hash(
        frozen_summary_path,
        expected_summary_sha256,
        name="expected_summary_sha256",
    )

    protocol = _load_frozen_protocol(frozen_summary_path)
    protocol_inputs = protocol["inputs"]
    assert isinstance(protocol_inputs, Mapping)
    keeper_table = _read_table(
        "keeper",
        Path(str(protocol_inputs["keeper_csv"])),
        "sample_index",
    )
    candidate_table = _read_table(
        "candidate",
        Path(str(protocol_inputs["candidate_csv"])),
        "sample_index",
    )
    (
        expected_keys,
        expected_targets,
        _expected_groups,
        expected_keeper_probabilities,
        expected_candidate_probabilities,
        expected_base_rows,
    ) = _aligned_arrays(keeper_table, candidate_table)
    _validate_prediction_row_split(expected_base_rows, expected_split="val")
    locked = protocol["locked"]
    assert isinstance(locked, Mapping)
    _, expected_predictions, _ = apply_precision_ensemble(
        expected_keeper_probabilities,
        expected_candidate_probabilities,
        candidate_weight=float(locked["candidate_weight"]),
        focus_margin_offset=float(locked["focus_margin_offset"]),
        focus_class=int(protocol["focus_class"]),
    )

    checkpoint = load_checkpoint(checkpoint_path, map_location="cpu")
    provenance = checkpoint.get("precision_ensemble_provenance")
    if not isinstance(provenance, Mapping):
        raise ValueError("Packaged checkpoint has no precision-ensemble provenance.")
    if str(provenance.get("frozen_protocol_sha256", "")).lower() != summary_hash:
        raise ValueError("Packaged checkpoint points to another frozen protocol.")
    device = torch.device(f"cuda:{int(cuda_device)}")
    model = build_model_from_checkpoint(checkpoint).to(device).eval()
    if not isinstance(model, PrecisionEnsembleClassifier):
        raise TypeError("Checkpoint did not load as PrecisionEnsembleClassifier.")
    classifier = TensorRTHybridModel(
        engine_path=engine_path,
        cuda_device=int(cuda_device),
        profile_index=0,
    )
    schema = _engine_schema(classifier)

    dataset = _eval_dataset(checkpoint, data_yaml)
    if len(dataset) != len(expected_keys):
        raise ValueError(
            f"Dataset/protocol row mismatch: dataset={len(dataset)}, protocol={len(expected_keys)}"
        )
    audit_indices = _resolve_audit_indices(
        dataset_size=len(dataset),
        max_samples=max_samples,
        sample_indices=sample_indices,
    )
    full_audit = audit_indices == list(range(len(dataset)))
    loader = DataLoader(
        Subset(dataset, audit_indices),
        batch_size=1,
        shuffle=False,
        num_workers=0,
        collate_fn=build_train_collate_fn(
            num_classes=len(checkpoint["class_names"]),
            batch_mix_probability=0.0,
        ),
    )

    target_rows: List[np.ndarray] = []
    pytorch_rows: List[np.ndarray] = []
    trt_rows: List[np.ndarray] = []
    audit_output_max_errors = {
        name: 0.0 for name in PrecisionEnsembleClassifier.deployment_output_names
    }
    logits_probability_consistency_max_abs = 0.0
    pytorch_seconds = 0.0
    trt_seconds = 0.0
    processed = 0
    started = time.perf_counter()
    with torch.inference_mode():
        for images, labels, metadata in loader:
            images = images.to(device=device, dtype=torch.float32)
            masks = metadata["image_mask"].to(device=device, dtype=torch.float32)
            bboxes = metadata["bbox"].to(device=device, dtype=torch.float32)

            reference_start = time.perf_counter()
            pytorch_outputs = model.audit_outputs(
                images,
                image_valid_mask=masks,
                bbox=bboxes,
            )
            pytorch_probabilities = pytorch_outputs["probabilities"]
            torch.cuda.synchronize(device)
            pytorch_seconds += time.perf_counter() - reference_start

            backend_start = time.perf_counter()
            trt_outputs = classifier.infer(
                {
                    "images": images,
                    "image_valid_mask": masks,
                    "bbox": bboxes,
                }
            )
            trt_probabilities = trt_outputs["deployment_probabilities"].float()
            trt_logits_probabilities = trt_outputs["logits"].float().softmax(1)
            logits_probability_consistency_max_abs = max(
                logits_probability_consistency_max_abs,
                float(
                    (trt_logits_probabilities - trt_probabilities)
                    .abs()
                    .max()
                    .item()
                ),
            )
            pytorch_deployment_outputs = {
                "logits": pytorch_outputs["logits"],
                "keeper_probabilities": pytorch_outputs["keeper_probabilities"],
                "candidate_probabilities": pytorch_outputs[
                    "candidate_probabilities"
                ],
                "raw_blend": pytorch_outputs["raw_blend"],
                "decision_scores": pytorch_outputs["decision_scores"],
                "deployment_probabilities": pytorch_outputs["probabilities"],
            }
            for name, reference in pytorch_deployment_outputs.items():
                audit_output_max_errors[name] = max(
                    audit_output_max_errors[name],
                    float((trt_outputs[name] - reference).abs().max().item()),
                )
            torch.cuda.synchronize(device)
            trt_seconds += time.perf_counter() - backend_start

            target_rows.append(labels.to(dtype=torch.long).numpy())
            pytorch_rows.append(pytorch_probabilities.cpu().numpy())
            trt_rows.append(trt_probabilities.cpu().numpy())
            processed += 1
            if processed % 256 == 0 or processed == len(audit_indices):
                print(
                    json.dumps(
                        {
                            "processed": processed,
                            "elapsed_seconds": time.perf_counter() - started,
                            "trt_seconds": trt_seconds,
                        }
                    ),
                    flush=True,
                )

    targets = np.concatenate(target_rows, axis=0)
    pytorch_probabilities = np.concatenate(pytorch_rows, axis=0)
    trt_probabilities = np.concatenate(trt_rows, axis=0)
    expected_count = processed
    expected_targets = expected_targets[audit_indices]
    expected_predictions = expected_predictions[audit_indices]
    if not np.array_equal(targets, expected_targets):
        raise ValueError("Dataset targets differ from frozen validation inputs.")

    pytorch_predictions = pytorch_probabilities.argmax(axis=1)
    trt_predictions = trt_probabilities.argmax(axis=1)
    row_errors = np.max(
        np.abs(trt_probabilities - pytorch_probabilities),
        axis=1,
    )
    probability_error = float(np.max(row_errors))
    backend_mismatches = int(np.sum(trt_predictions != pytorch_predictions))
    frozen_mismatches = int(np.sum(trt_predictions != expected_predictions))
    reference_frozen_mismatches = int(
        np.sum(pytorch_predictions != expected_predictions)
    )
    class_names = list(checkpoint["class_names"])
    metrics = _metric_summary(
        targets,
        trt_predictions,
        class_names,
        int(protocol["focus_class"]),
    )
    expected_metrics = locked["metrics"]
    assert isinstance(expected_metrics, Mapping)
    metric_error = (
        _max_metric_error(metrics, expected_metrics)
        if full_audit
        else None
    )
    gates: Dict[str, bool] = {
        "validation_only": True,
        "fixed_batch_one_schema": True,
        "audit_outputs_exposed": True,
        "audit_outputs_tolerance": max(audit_output_max_errors.values())
        <= float(probability_tolerance),
        "logits_probability_consistency": (
            logits_probability_consistency_max_abs <= 1e-6
        ),
        "full_support": full_audit and expected_count == len(expected_keys) == 2606,
        "targets_exact": True,
        "probability_tolerance": probability_error <= float(probability_tolerance),
        "backend_argmax_exact": backend_mismatches == 0,
        "frozen_argmax_exact": frozen_mismatches == 0,
        "reference_frozen_argmax_exact": reference_frozen_mismatches == 0,
        "metrics_exact": metric_error is not None and metric_error <= 1e-9,
    }

    rows: List[Dict[str, object]] = []
    for index, dataset_index in enumerate(audit_indices):
        row: Dict[str, object] = {
            "sample_index": expected_keys[dataset_index],
            "dataset_index": dataset_index,
            "image_path": _source_path(expected_base_rows[dataset_index]),
            "target_index": int(targets[index]),
            "expected_prediction_index": int(expected_predictions[index]),
            "pytorch_prediction_index": int(pytorch_predictions[index]),
            "trt_prediction_index": int(trt_predictions[index]),
            "max_probability_error": float(row_errors[index]),
        }
        for class_index, class_name in enumerate(class_names):
            row[f"pytorch_prob_{class_index}_{class_name}"] = float(
                pytorch_probabilities[index, class_index]
            )
            row[f"trt_prob_{class_index}_{class_name}"] = float(
                trt_probabilities[index, class_index]
            )
        rows.append(row)

    output_dir.mkdir(parents=True, exist_ok=True)
    predictions_path = output_dir / "predictions_detailed.csv"
    mismatches_path = output_dir / "mismatches.csv"
    _write_csv(predictions_path, rows)
    _write_csv(
        mismatches_path,
        [
            row
            for row in rows
            if int(row["trt_prediction_index"])
            != int(row["pytorch_prediction_index"])
            or int(row["trt_prediction_index"])
            != int(row["expected_prediction_index"])
        ],
    )
    scope_gates = {
        key: value
        for key, value in gates.items()
        if full_audit or key not in {"full_support", "metrics_exact"}
    }
    summary: Dict[str, object] = {
        "mode": (
            "precision_ensemble_tensorrt_full_validation_parity"
            if full_audit
            else "precision_ensemble_tensorrt_validation_probe"
        ),
        "test_data_used": False,
        "checkpoint": str(checkpoint_path.resolve()),
        "checkpoint_sha256": checkpoint_hash,
        "engine": str(engine_path.resolve()),
        "engine_sha256": engine_hash,
        "frozen_protocol": str(frozen_summary_path.resolve()),
        "frozen_protocol_sha256": summary_hash,
        "data_yaml": str(data_yaml.resolve()),
        "device": str(device),
        "gpu_name": torch.cuda.get_device_name(device),
        "tensorrt_version": trt.__version__,
        "schema": schema,
        "rows": expected_count,
        "full_expected_rows": len(expected_keys),
        "audit_scope": "full" if full_audit else "probe",
        "audit_indices": audit_indices,
        "probability_tolerance": float(probability_tolerance),
        "probability_max_abs": probability_error,
        "audit_output_max_abs": audit_output_max_errors,
        "logits_probability_consistency_max_abs": (
            logits_probability_consistency_max_abs
        ),
        "mismatches": {
            "trt_vs_pytorch": backend_mismatches,
            "trt_vs_frozen": frozen_mismatches,
            "pytorch_vs_frozen": reference_frozen_mismatches,
        },
        "timings_seconds": {
            "total": time.perf_counter() - started,
            "pytorch": pytorch_seconds,
            "tensorrt": trt_seconds,
        },
        "metrics": metrics,
        "metric_max_abs": metric_error,
        "gates": gates,
        "scope_gates": scope_gates,
        "scope_gates_passed": all(scope_gates.values()),
        "all_gates_passed": full_audit and all(gates.values()),
        "artifacts": {
            "predictions": predictions_path.name,
            "predictions_sha256": _sha256(predictions_path),
            "mismatches": mismatches_path.name,
            "mismatches_sha256": _sha256(mismatches_path),
        },
    }
    summary_path = output_dir / "summary.json"
    summary_path.write_text(
        json.dumps(summary, indent=2, ensure_ascii=True),
        encoding="utf-8",
    )
    return summary


def parse_args(argv: Sequence[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Audit TensorRT parity of a packaged precision ensemble."
    )
    parser.add_argument("--checkpoint", type=Path, required=True)
    parser.add_argument("--engine", type=Path, required=True)
    parser.add_argument("--frozen-summary", type=Path, required=True)
    parser.add_argument("--data", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--expected-checkpoint-sha256", required=True)
    parser.add_argument("--expected-engine-sha256", required=True)
    parser.add_argument("--expected-summary-sha256", required=True)
    parser.add_argument("--max-samples", type=int, default=0)
    parser.add_argument("--probability-tolerance", type=float, default=5e-3)
    parser.add_argument("--cuda-device", type=int, default=0)
    parser.add_argument(
        "--sample-indices",
        default="",
        help="Comma-separated validation dataset indices for a targeted probe.",
    )
    return parser.parse_args(argv)


def main(argv: Sequence[str] | None = None) -> int:
    args = parse_args(argv)
    set_seed(42, deterministic=True)
    summary = audit_trt(
        checkpoint_path=args.checkpoint.expanduser().resolve(),
        engine_path=args.engine.expanduser().resolve(),
        frozen_summary_path=args.frozen_summary.expanduser().resolve(),
        data_yaml=args.data.expanduser().resolve(),
        output_dir=args.output_dir.expanduser().resolve(),
        expected_checkpoint_sha256=args.expected_checkpoint_sha256,
        expected_engine_sha256=args.expected_engine_sha256,
        expected_summary_sha256=args.expected_summary_sha256,
        max_samples=args.max_samples,
        probability_tolerance=args.probability_tolerance,
        cuda_device=args.cuda_device,
        sample_indices=_parse_sample_indices(args.sample_indices),
    )
    print(
        json.dumps(
            {
                "rows": summary["rows"],
                "probability_max_abs": summary["probability_max_abs"],
                "mismatches": summary["mismatches"],
                "audit_scope": summary["audit_scope"],
                "scope_gates_passed": summary["scope_gates_passed"],
                "all_gates_passed": summary["all_gates_passed"],
            }
        )
    )
    return 0 if bool(summary["scope_gates_passed"]) else 2


if __name__ == "__main__":
    raise SystemExit(main())
