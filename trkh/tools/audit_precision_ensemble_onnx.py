from __future__ import annotations

import argparse
import csv
import json
import os
import time
from pathlib import Path
from typing import Dict, List, Mapping, Sequence

import numpy as np
import onnx
import onnxruntime as ort
import torch
from torch.utils.data import DataLoader, Subset

from trkh.core.utils import load_checkpoint, set_seed
from trkh.data.dataset import build_train_collate_fn
from trkh.models.model import build_model_from_checkpoint
from trkh.models.precision_ensemble import PrecisionEnsembleClassifier
from trkh.tools.apply_frozen_precision_ensemble import _load_frozen_protocol
from trkh.tools.audit_precision_ensemble_checkpoint import (
    _eval_dataset,
    _max_metric_error,
)
from trkh.tools.audit_precision_ensemble_readiness import (
    _aligned_arrays,
    _metric_summary,
    _sha256,
    _validate_prediction_row_split,
    apply_precision_ensemble,
)
from trkh.tools.build_precision_ensemble_checkpoint import _verify_file_hash
from trkh.tools.soft_ensemble_predictions import _read_table, _source_path


_DLL_DIRECTORY_HANDLES: List[object] = []


def _write_csv(path: Path, rows: Sequence[Mapping[str, object]]) -> None:
    fieldnames: List[str] = []
    for row in rows:
        for key in row:
            if key not in fieldnames:
                fieldnames.append(str(key))
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)


def _configure_onnxruntime_cuda_dlls() -> List[str]:
    if os.name != "nt":
        return []
    torch_lib = Path(torch.__file__).resolve().parent / "lib"
    if not torch_lib.is_dir():
        raise FileNotFoundError(f"PyTorch CUDA DLL directory not found: {torch_lib}")
    torch_lib_text = str(torch_lib)
    path_entries = os.environ.get("PATH", "").split(os.pathsep)
    if torch_lib_text.lower() not in {entry.lower() for entry in path_entries}:
        os.environ["PATH"] = torch_lib_text + os.pathsep + os.environ.get("PATH", "")
    add_dll_directory = getattr(os, "add_dll_directory", None)
    if callable(add_dll_directory):
        _DLL_DIRECTORY_HANDLES.append(add_dll_directory(torch_lib_text))
    return [torch_lib_text]


def _ort_provider_configuration(provider: str) -> Dict[str, object]:
    requested = str(provider).strip().lower()
    available = list(ort.get_available_providers())
    if requested == "auto":
        requested = (
            "cuda" if "CUDAExecutionProvider" in available else "cpu"
        )
    if requested == "cuda":
        if "CUDAExecutionProvider" not in available:
            raise RuntimeError(
                "ONNX Runtime CUDAExecutionProvider is not installed; "
                f"available providers: {available}"
            )
        dll_directories = _configure_onnxruntime_cuda_dlls()
        providers = ["CUDAExecutionProvider", "CPUExecutionProvider"]
    elif requested == "cpu":
        dll_directories = []
        providers = ["CPUExecutionProvider"]
    else:
        raise ValueError(f"Unsupported ONNX Runtime provider: {provider}")
    return {
        "requested": requested,
        "available": available,
        "providers": providers,
        "dll_directories": dll_directories,
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
        raise ValueError("ONNX audit dataset must not be empty.")
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


def _fixed_batch_one_schema(
    session: ort.InferenceSession,
) -> Dict[str, Dict[str, List[int]]]:
    expected = {
        "images": [1, 3, 256, 256],
        "image_valid_mask": [1, 256, 256],
        "bbox": [1, 4],
    }
    observed = {item.name: list(item.shape) for item in session.get_inputs()}
    if observed != expected:
        raise ValueError(f"ONNX input schema mismatch: expected={expected}, got={observed}")
    expected_outputs = {
        name: [1, 5]
        for name in PrecisionEnsembleClassifier.deployment_output_names
    }
    observed_outputs = {
        item.name: list(item.shape) for item in session.get_outputs()
    }
    if observed_outputs != expected_outputs:
        raise ValueError(
            "ONNX audit-output schema mismatch: "
            f"expected={expected_outputs}, got={observed_outputs}"
        )
    return {"inputs": observed, "outputs": observed_outputs}


def audit_onnx(
    *,
    checkpoint_path: Path,
    onnx_path: Path,
    frozen_summary_path: Path,
    data_yaml: Path,
    output_dir: Path,
    expected_checkpoint_sha256: str,
    expected_onnx_sha256: str,
    expected_summary_sha256: str,
    reference_batch_size: int,
    max_samples: int,
    probability_tolerance: float,
    device: torch.device,
    onnx_provider: str,
    sample_indices: Sequence[int] = (),
) -> Dict[str, object]:
    output_dir = Path(output_dir)
    if output_dir.exists() and any(output_dir.iterdir()):
        raise FileExistsError(f"Audit output directory is nonempty: {output_dir}")
    checkpoint_hash = _verify_file_hash(
        checkpoint_path,
        expected_checkpoint_sha256,
        name="expected_checkpoint_sha256",
    )
    onnx_hash = _verify_file_hash(
        onnx_path,
        expected_onnx_sha256,
        name="expected_onnx_sha256",
    )
    summary_hash = _verify_file_hash(
        frozen_summary_path,
        expected_summary_sha256,
        name="expected_summary_sha256",
    )

    graph = onnx.load(str(onnx_path), load_external_data=False)
    onnx.checker.check_model(graph)
    provider_config = _ort_provider_configuration(onnx_provider)
    session_options = ort.SessionOptions()
    session_options.graph_optimization_level = ort.GraphOptimizationLevel.ORT_ENABLE_ALL
    session_options.log_severity_level = 3
    session = ort.InferenceSession(
        str(onnx_path),
        sess_options=session_options,
        providers=provider_config["providers"],
    )
    active_providers = list(session.get_providers())
    if (
        provider_config["requested"] == "cuda"
        and (
            not active_providers
            or active_providers[0] != "CUDAExecutionProvider"
        )
    ):
        raise RuntimeError(
            "ONNX Runtime did not activate CUDAExecutionProvider first: "
            f"{active_providers}"
        )
    io_schema = _fixed_batch_one_schema(session)
    output_names = list(PrecisionEnsembleClassifier.deployment_output_names)

    protocol = _load_frozen_protocol(frozen_summary_path)
    inputs = protocol["inputs"]
    assert isinstance(inputs, Mapping)
    keeper_table = _read_table(
        "keeper",
        Path(str(inputs["keeper_csv"])),
        "sample_index",
    )
    candidate_table = _read_table(
        "candidate",
        Path(str(inputs["candidate_csv"])),
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
    model = build_model_from_checkpoint(checkpoint).to(device).eval()
    if not isinstance(model, PrecisionEnsembleClassifier):
        raise TypeError("Checkpoint did not load as PrecisionEnsembleClassifier.")
    if bool(getattr(model, "supports_dynamic_batch", True)):
        raise ValueError("Precision-ensemble deployment contract must be fixed batch 1.")
    if int(reference_batch_size) != 1:
        raise ValueError(
            "ONNX parity reference must use deployment batch geometry 1; "
            "routing top-k can change at floating-point ties across batch kernels."
        )

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
        batch_size=max(1, int(reference_batch_size)),
        shuffle=False,
        num_workers=0,
        collate_fn=build_train_collate_fn(
            num_classes=len(checkpoint["class_names"]),
            batch_mix_probability=0.0,
        ),
    )

    target_rows: List[np.ndarray] = []
    pytorch_rows: List[np.ndarray] = []
    onnx_rows: List[np.ndarray] = []
    audit_output_max_errors = {name: 0.0 for name in output_names}
    pytorch_seconds = 0.0
    onnx_seconds = 0.0
    processed = 0
    started = time.perf_counter()
    with torch.inference_mode():
        for images, labels, metadata in loader:
            take = len(images)
            images = images[:take].to(dtype=torch.float32)
            labels = labels[:take].to(dtype=torch.long)
            masks = metadata["image_mask"][:take].to(dtype=torch.float32)
            bboxes = metadata["bbox"][:take].to(dtype=torch.float32)

            reference_start = time.perf_counter()
            pytorch_outputs = model.audit_outputs(
                images.to(device=device),
                image_valid_mask=masks.to(device=device),
                bbox=bboxes.to(device=device),
            )
            if device.type == "cuda":
                torch.cuda.synchronize(device)
            pytorch_seconds += time.perf_counter() - reference_start
            pytorch_deployment_outputs = {
                "logits": pytorch_outputs["logits"].detach().cpu().numpy(),
                "keeper_probabilities": pytorch_outputs[
                    "keeper_probabilities"
                ].detach().cpu().numpy(),
                "candidate_probabilities": pytorch_outputs[
                    "candidate_probabilities"
                ].detach().cpu().numpy(),
                "raw_blend": pytorch_outputs["raw_blend"].detach().cpu().numpy(),
                "decision_scores": pytorch_outputs[
                    "decision_scores"
                ].detach().cpu().numpy(),
                "deployment_probabilities": pytorch_outputs[
                    "probabilities"
                ].detach().cpu().numpy(),
            }
            pytorch_probabilities = pytorch_deployment_outputs[
                "deployment_probabilities"
            ]

            images_array = images.numpy()
            masks_array = masks.numpy()
            bboxes_array = bboxes.numpy()
            backend_batch: List[np.ndarray] = []
            backend_start = time.perf_counter()
            for index in range(take):
                values = session.run(
                    output_names,
                    {
                        "images": images_array[index : index + 1],
                        "image_valid_mask": masks_array[index : index + 1],
                        "bbox": bboxes_array[index : index + 1],
                    },
                )
                backend_outputs = dict(zip(output_names, values))
                for name in output_names:
                    reference = pytorch_deployment_outputs[name][index : index + 1]
                    audit_output_max_errors[name] = max(
                        audit_output_max_errors[name],
                        float(np.max(np.abs(backend_outputs[name] - reference))),
                    )
                backend_batch.append(backend_outputs["deployment_probabilities"])
            onnx_seconds += time.perf_counter() - backend_start

            target_rows.append(labels.numpy())
            pytorch_rows.append(pytorch_probabilities)
            onnx_rows.append(np.concatenate(backend_batch, axis=0))
            processed += take
            if processed % 256 < take or processed == len(audit_indices):
                print(
                    json.dumps(
                        {
                            "processed": processed,
                            "elapsed_seconds": time.perf_counter() - started,
                            "onnx_seconds": onnx_seconds,
                        }
                    ),
                    flush=True,
                )

    targets = np.concatenate(target_rows, axis=0)
    pytorch_probabilities = np.concatenate(pytorch_rows, axis=0)
    onnx_probabilities = np.concatenate(onnx_rows, axis=0)
    expected_count = processed
    selected_expected_targets = expected_targets[audit_indices]
    selected_expected_predictions = expected_predictions[audit_indices]
    if not np.array_equal(targets, selected_expected_targets):
        raise ValueError("Dataset targets differ from frozen validation inputs.")

    pytorch_predictions = pytorch_probabilities.argmax(axis=1)
    onnx_predictions = onnx_probabilities.argmax(axis=1)
    row_errors = np.max(
        np.abs(onnx_probabilities - pytorch_probabilities),
        axis=1,
    )
    probability_error = float(np.max(row_errors))
    backend_mismatches = int(np.sum(onnx_predictions != pytorch_predictions))
    frozen_mismatches = int(
        np.sum(onnx_predictions != selected_expected_predictions)
    )
    reference_frozen_mismatches = int(
        np.sum(pytorch_predictions != selected_expected_predictions)
    )
    class_names = list(checkpoint["class_names"])
    metrics = _metric_summary(
        targets,
        onnx_predictions,
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
        "onnx_checker": True,
        "fixed_batch_one_schema": True,
        "audit_outputs_exposed": True,
        "audit_outputs_tolerance": max(audit_output_max_errors.values())
        <= float(probability_tolerance),
        "reference_batch_geometry_exact": int(reference_batch_size) == 1,
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
            "expected_prediction_index": int(
                selected_expected_predictions[index]
            ),
            "pytorch_prediction_index": int(pytorch_predictions[index]),
            "onnx_prediction_index": int(onnx_predictions[index]),
            "max_probability_error": float(row_errors[index]),
        }
        for class_index, class_name in enumerate(class_names):
            row[f"pytorch_prob_{class_index}_{class_name}"] = float(
                pytorch_probabilities[index, class_index]
            )
            row[f"onnx_prob_{class_index}_{class_name}"] = float(
                onnx_probabilities[index, class_index]
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
            if int(row["onnx_prediction_index"])
            != int(row["pytorch_prediction_index"])
            or int(row["onnx_prediction_index"])
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
            "precision_ensemble_onnx_full_validation_parity"
            if full_audit
            else "precision_ensemble_onnx_validation_probe"
        ),
        "test_data_used": False,
        "checkpoint": str(checkpoint_path.resolve()),
        "checkpoint_sha256": checkpoint_hash,
        "onnx": str(onnx_path.resolve()),
        "onnx_sha256": onnx_hash,
        "frozen_protocol": str(frozen_summary_path.resolve()),
        "frozen_protocol_sha256": summary_hash,
        "data_yaml": str(data_yaml.resolve()),
        "device": str(device),
        "pytorch_reference_runtime": {
            "cudnn_deterministic": bool(torch.backends.cudnn.deterministic),
            "cudnn_benchmark": bool(torch.backends.cudnn.benchmark),
            "batch_size": int(reference_batch_size),
        },
        "onnxruntime_provider_requested": provider_config["requested"],
        "onnxruntime_provider_request_order": provider_config["providers"],
        "onnxruntime_providers_available": provider_config["available"],
        "onnxruntime_providers_active": active_providers,
        "onnxruntime_cuda_dll_directories": provider_config["dll_directories"],
        "onnx_version": onnx.__version__,
        "onnxruntime_version": ort.__version__,
        "io_schema": io_schema,
        "audit_output_max_abs": audit_output_max_errors,
        "rows": expected_count,
        "full_expected_rows": len(expected_keys),
        "audit_scope": "full" if full_audit else "probe",
        "audit_indices": audit_indices,
        "probability_tolerance": float(probability_tolerance),
        "probability_max_abs": probability_error,
        "mismatches": {
            "onnx_vs_pytorch": backend_mismatches,
            "onnx_vs_frozen": frozen_mismatches,
            "pytorch_vs_frozen": reference_frozen_mismatches,
        },
        "timings_seconds": {
            "total": time.perf_counter() - started,
            "pytorch": pytorch_seconds,
            "onnx": onnx_seconds,
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
        description="Audit fixed-batch ONNX parity of a packaged precision ensemble."
    )
    parser.add_argument("--checkpoint", type=Path, required=True)
    parser.add_argument("--onnx", type=Path, required=True)
    parser.add_argument("--frozen-summary", type=Path, required=True)
    parser.add_argument("--data", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--expected-checkpoint-sha256", required=True)
    parser.add_argument("--expected-onnx-sha256", required=True)
    parser.add_argument("--expected-summary-sha256", required=True)
    parser.add_argument("--reference-batch-size", type=int, default=1)
    parser.add_argument("--max-samples", type=int, default=0)
    parser.add_argument("--probability-tolerance", type=float, default=1e-5)
    parser.add_argument("--device", choices=("auto", "cpu", "cuda"), default="auto")
    parser.add_argument(
        "--onnx-provider",
        choices=("auto", "cpu", "cuda"),
        default="cpu",
        help="ONNX Runtime execution provider; CUDA falls back per-node to CPU only.",
    )
    parser.add_argument(
        "--sample-indices",
        default="",
        help="Comma-separated validation dataset indices for a targeted probe.",
    )
    return parser.parse_args(argv)


def main(argv: Sequence[str] | None = None) -> int:
    args = parse_args(argv)
    set_seed(42, deterministic=True)
    if args.device == "auto":
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    else:
        device = torch.device(args.device)
    summary = audit_onnx(
        checkpoint_path=args.checkpoint.expanduser().resolve(),
        onnx_path=args.onnx.expanduser().resolve(),
        frozen_summary_path=args.frozen_summary.expanduser().resolve(),
        data_yaml=args.data.expanduser().resolve(),
        output_dir=args.output_dir.expanduser().resolve(),
        expected_checkpoint_sha256=args.expected_checkpoint_sha256,
        expected_onnx_sha256=args.expected_onnx_sha256,
        expected_summary_sha256=args.expected_summary_sha256,
        reference_batch_size=args.reference_batch_size,
        max_samples=args.max_samples,
        probability_tolerance=args.probability_tolerance,
        device=device,
        onnx_provider=args.onnx_provider,
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
