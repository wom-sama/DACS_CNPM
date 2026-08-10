from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path
from typing import Dict, List, Mapping, Sequence

import numpy as np
import torch
from torch.utils.data import DataLoader

from trkh.core.config import load_data_spec
from trkh.core.utils import load_checkpoint, set_seed
from trkh.data.dataset import (
    MangoYOLOCropDataset,
    build_eval_transform,
    build_train_collate_fn,
)
from trkh.evaluation.evaluate import (
    resolve_classification_object_crops,
    resolve_crop_to_primary_object,
)
from trkh.evaluation.input_normalization import checkpoint_input_normalization
from trkh.models.model import build_model_from_checkpoint
from trkh.models.precision_ensemble import (
    PrecisionEnsembleClassifier,
    _classification_member_logits,
)
from trkh.tools.apply_frozen_precision_ensemble import _load_frozen_protocol
from trkh.tools.audit_precision_ensemble_readiness import (
    _aligned_arrays,
    _metric_summary,
    _sha256,
    _validate_prediction_row_split,
    apply_precision_ensemble,
)
from trkh.tools.build_precision_ensemble_checkpoint import _verify_file_hash
from trkh.tools.soft_ensemble_predictions import _read_table, _source_path


def _eval_dataset(
    checkpoint: Mapping[str, object],
    data_yaml: Path,
) -> MangoYOLOCropDataset:
    data_spec = load_data_spec(
        data_yaml,
        class_name_mode="raw",
        expected_num_classes=len(checkpoint["class_names"]),
    )
    augmentation = checkpoint.get("augmentation_config", {})
    if not isinstance(augmentation, Mapping):
        augmentation = {}
    image_size = int(checkpoint["model_config"].get("image_size", 256))
    input_mean, input_std = checkpoint_input_normalization(dict(checkpoint))
    transform = build_eval_transform(
        image_size=image_size,
        resize_mode=str(augmentation.get("resize_mode", "pad")),
        illumination_normalization=bool(
            augmentation.get("illumination_normalization", False)
        ),
        illumination_normalization_strength=float(
            augmentation.get("illumination_normalization_strength", 0.0) or 0.0
        ),
        foreground_crop_mode=str(augmentation.get("foreground_crop_mode", "none")),
        foreground_crop_margin_ratio=float(
            augmentation.get("foreground_crop_margin_ratio", 0.08) or 0.08
        ),
        foreground_crop_min_mask_area_ratio=float(
            augmentation.get("foreground_crop_min_mask_area_ratio", 0.03) or 0.03
        ),
        foreground_crop_max_mask_area_ratio=float(
            augmentation.get("foreground_crop_max_mask_area_ratio", 0.92) or 0.92
        ),
        foreground_crop_max_crop_area_ratio=float(
            augmentation.get("foreground_crop_max_crop_area_ratio", 0.98) or 0.98
        ),
        background_suppression_mode=str(
            augmentation.get("background_suppression_mode", "none")
        ),
        background_suppression_margin=float(
            augmentation.get("background_suppression_margin", 0.08) or 0.08
        ),
        background_suppression_blur_radius=float(
            augmentation.get("background_suppression_blur_radius", 7.0) or 7.0
        ),
        surface_detail_amplification_mode=str(
            augmentation.get("surface_detail_amplification_mode", "none")
        ),
        surface_detail_amplification_strength=float(
            augmentation.get("surface_detail_amplification_strength", 0.0) or 0.0
        ),
        surface_detail_amplification_blur_radius=float(
            augmentation.get("surface_detail_amplification_blur_radius", 1.25) or 1.25
        ),
        surface_detail_amplification_foreground_weight=float(
            augmentation.get("surface_detail_amplification_foreground_weight", 0.85)
            or 0.85
        ),
        eval_surface_detail_amplification=bool(
            augmentation.get("eval_surface_detail_amplification", False)
        ),
        mean=input_mean,
        std=input_std,
    )
    return MangoYOLOCropDataset.from_data_spec(
        data_spec=data_spec,
        split="val",
        transform=transform,
        crop_margin_ratio=float(augmentation.get("crop_margin_ratio", 0.05)),
        crop_to_primary_object=resolve_crop_to_primary_object(dict(checkpoint)),
        classification_target=True,
        classification_object_crops=resolve_classification_object_crops(
            dict(checkpoint)
        ),
        classification_bbox_metadata=True,
    )


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


def _max_metric_error(
    observed: Mapping[str, object],
    expected: Mapping[str, object],
) -> float:
    errors: List[float] = []
    for key in ("accuracy", "macro_f1", "weighted_f1"):
        errors.append(abs(float(observed[key]) - float(expected[key])))
    observed_focus = observed["focus"]
    expected_focus = expected["focus"]
    assert isinstance(observed_focus, Mapping)
    assert isinstance(expected_focus, Mapping)
    for key in ("precision", "recall", "f1"):
        errors.append(abs(float(observed_focus[key]) - float(expected_focus[key])))
    observed_per_class = observed["per_class"]
    expected_per_class = expected["per_class"]
    assert isinstance(observed_per_class, Sequence)
    assert isinstance(expected_per_class, Sequence)
    for observed_row, expected_row in zip(observed_per_class, expected_per_class):
        assert isinstance(observed_row, Mapping)
        assert isinstance(expected_row, Mapping)
        for key in ("precision", "recall", "f1"):
            errors.append(abs(float(observed_row[key]) - float(expected_row[key])))
    return max(errors, default=0.0)


def audit_checkpoint(
    *,
    checkpoint_path: Path,
    keeper_path: Path,
    candidate_path: Path,
    frozen_summary_path: Path,
    data_yaml: Path,
    output_dir: Path,
    expected_checkpoint_sha256: str,
    expected_keeper_sha256: str,
    expected_candidate_sha256: str,
    expected_summary_sha256: str,
    batch_size: int,
    max_samples: int,
    device: torch.device,
) -> Dict[str, object]:
    output_dir = Path(output_dir)
    if output_dir.exists() and any(output_dir.iterdir()):
        raise FileExistsError(f"Audit output directory is nonempty: {output_dir}")
    checkpoint_hash = _verify_file_hash(
        checkpoint_path,
        expected_checkpoint_sha256,
        name="expected_checkpoint_sha256",
    )
    keeper_hash = _verify_file_hash(
        keeper_path,
        expected_keeper_sha256,
        name="expected_keeper_sha256",
    )
    candidate_hash = _verify_file_hash(
        candidate_path,
        expected_candidate_sha256,
        name="expected_candidate_sha256",
    )
    summary_hash = _verify_file_hash(
        frozen_summary_path,
        expected_summary_sha256,
        name="expected_summary_sha256",
    )
    protocol = _load_frozen_protocol(frozen_summary_path)
    inputs = protocol["inputs"]
    assert isinstance(inputs, Mapping)
    keeper_table = _read_table("keeper", Path(str(inputs["keeper_csv"])), "sample_index")
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
    expected_raw_blend, expected_predictions, expected_focus_margin = (
        apply_precision_ensemble(
            expected_keeper_probabilities,
            expected_candidate_probabilities,
            candidate_weight=float(locked["candidate_weight"]),
            focus_margin_offset=float(locked["focus_margin_offset"]),
            focus_class=int(protocol["focus_class"]),
        )
    )

    packaged_checkpoint = load_checkpoint(checkpoint_path, map_location="cpu")
    provenance = packaged_checkpoint.get("precision_ensemble_provenance")
    if not isinstance(provenance, Mapping):
        raise ValueError("Packaged checkpoint has no precision-ensemble provenance.")
    provenance_expectations = {
        "keeper_checkpoint_sha256": keeper_hash,
        "candidate_checkpoint_sha256": candidate_hash,
        "frozen_protocol_sha256": summary_hash,
    }
    for key, expected in provenance_expectations.items():
        if str(provenance.get(key, "")).lower() != expected:
            raise ValueError(f"Packaged provenance mismatch for {key}.")

    packaged_model = build_model_from_checkpoint(packaged_checkpoint).to(device).eval()
    if not isinstance(packaged_model, PrecisionEnsembleClassifier):
        raise TypeError("Checkpoint did not load as PrecisionEnsembleClassifier.")
    keeper_checkpoint = load_checkpoint(keeper_path, map_location="cpu")
    candidate_checkpoint = load_checkpoint(candidate_path, map_location="cpu")
    keeper_model = build_model_from_checkpoint(keeper_checkpoint).to(device).eval()
    candidate_model = build_model_from_checkpoint(candidate_checkpoint).to(device).eval()

    dataset = _eval_dataset(packaged_checkpoint, data_yaml)
    if len(dataset) != len(expected_keys):
        raise ValueError(
            f"Dataset/protocol row mismatch: dataset={len(dataset)}, protocol={len(expected_keys)}"
        )
    loader = DataLoader(
        dataset,
        batch_size=max(1, int(batch_size)),
        shuffle=False,
        num_workers=0,
        collate_fn=build_train_collate_fn(
            num_classes=len(packaged_checkpoint["class_names"]),
            batch_mix_probability=0.0,
        ),
    )

    observed_targets: List[np.ndarray] = []
    packaged_keeper_rows: List[np.ndarray] = []
    packaged_candidate_rows: List[np.ndarray] = []
    independent_keeper_rows: List[np.ndarray] = []
    independent_candidate_rows: List[np.ndarray] = []
    raw_blend_rows: List[np.ndarray] = []
    decision_score_rows: List[np.ndarray] = []
    probability_rows: List[np.ndarray] = []
    processed = 0
    with torch.inference_mode():
        for images, labels, metadata in loader:
            if int(max_samples) > 0 and processed >= int(max_samples):
                break
            remaining = int(max_samples) - processed if int(max_samples) > 0 else len(images)
            take = min(len(images), remaining)
            images = images[:take].to(device=device, dtype=torch.float32)
            labels = labels[:take].to(dtype=torch.long)
            image_valid_mask = metadata["image_mask"][:take].to(device=device)
            bbox = metadata["bbox"][:take].to(device=device, dtype=torch.float32)
            packaged = packaged_model.audit_outputs(
                images,
                image_valid_mask=image_valid_mask,
                bbox=bbox,
            )
            independent_keeper_logits = _classification_member_logits(
                keeper_model,
                images,
                image_valid_mask > 0.5,
                bbox,
            )
            independent_candidate_logits = _classification_member_logits(
                candidate_model,
                images,
                image_valid_mask > 0.5,
                bbox,
            )
            observed_targets.append(labels.numpy())
            packaged_keeper_rows.append(
                packaged["keeper_probabilities"].detach().cpu().numpy()
            )
            packaged_candidate_rows.append(
                packaged["candidate_probabilities"].detach().cpu().numpy()
            )
            independent_keeper_rows.append(
                independent_keeper_logits.float().softmax(1).detach().cpu().numpy()
            )
            independent_candidate_rows.append(
                independent_candidate_logits.float().softmax(1).detach().cpu().numpy()
            )
            raw_blend_rows.append(packaged["raw_blend"].detach().cpu().numpy())
            decision_score_rows.append(
                packaged["decision_scores"].detach().cpu().numpy()
            )
            probability_rows.append(packaged["probabilities"].detach().cpu().numpy())
            processed += take

    observed_targets_array = np.concatenate(observed_targets, axis=0)
    packaged_keeper = np.concatenate(packaged_keeper_rows, axis=0)
    packaged_candidate = np.concatenate(packaged_candidate_rows, axis=0)
    independent_keeper = np.concatenate(independent_keeper_rows, axis=0)
    independent_candidate = np.concatenate(independent_candidate_rows, axis=0)
    raw_blend = np.concatenate(raw_blend_rows, axis=0)
    decision_scores = np.concatenate(decision_score_rows, axis=0)
    deployment_probabilities = np.concatenate(probability_rows, axis=0)
    expected_count = processed
    expected_targets = expected_targets[:expected_count]
    expected_keeper_probabilities = expected_keeper_probabilities[:expected_count]
    expected_candidate_probabilities = expected_candidate_probabilities[:expected_count]
    expected_raw_blend = expected_raw_blend[:expected_count]
    expected_predictions = expected_predictions[:expected_count]
    expected_focus_margin = expected_focus_margin[:expected_count]
    if not np.array_equal(observed_targets_array, expected_targets):
        raise ValueError("Dataset targets differ from frozen validation inputs.")

    packaged_predictions = deployment_probabilities.argmax(axis=1)
    decision_predictions = decision_scores.argmax(axis=1)
    candidate_weight = float(locked["candidate_weight"])
    recomputed_raw_blend = (
        (1.0 - candidate_weight) * packaged_keeper
        + candidate_weight * packaged_candidate
    )
    recomputed_decision_scores = recomputed_raw_blend.copy()
    recomputed_decision_scores[:, int(protocol["focus_class"])] -= float(
        locked["focus_margin_offset"]
    )
    recomputed_safe_scores = np.maximum(recomputed_decision_scores, 1e-8)
    recomputed_deployment_probabilities = recomputed_safe_scores / np.maximum(
        recomputed_safe_scores.sum(axis=1, keepdims=True),
        1e-8,
    )
    raw_blend_formula_error = float(
        np.max(np.abs(raw_blend - recomputed_raw_blend))
    )
    decision_score_formula_error = float(
        np.max(np.abs(decision_scores - recomputed_decision_scores))
    )
    deployment_formula_error = float(
        np.max(
            np.abs(
                deployment_probabilities - recomputed_deployment_probabilities
            )
        )
    )
    class_names = list(packaged_checkpoint["class_names"])
    metrics = _metric_summary(
        observed_targets_array,
        packaged_predictions,
        class_names,
        int(protocol["focus_class"]),
    )
    expected_metrics = locked["metrics"]
    assert isinstance(expected_metrics, Mapping)
    metric_error = (
        _max_metric_error(metrics, expected_metrics)
        if expected_count == len(expected_keys)
        else None
    )
    member_keeper_error = float(np.max(np.abs(packaged_keeper - independent_keeper)))
    member_candidate_error = float(
        np.max(np.abs(packaged_candidate - independent_candidate))
    )
    keeper_csv_error = float(
        np.max(np.abs(packaged_keeper - expected_keeper_probabilities))
    )
    candidate_csv_error = float(
        np.max(np.abs(packaged_candidate - expected_candidate_probabilities))
    )
    raw_blend_error = float(np.max(np.abs(raw_blend - expected_raw_blend)))
    decision_mismatches = int(np.sum(decision_predictions != expected_predictions))
    deployment_mismatches = int(np.sum(packaged_predictions != expected_predictions))
    softmax_decision_mismatches = int(
        np.sum(packaged_predictions != decision_predictions)
    )

    gates: Dict[str, bool] = {
        "validation_only": True,
        "full_support": expected_count == len(expected_keys) == 2606,
        "targets_exact": True,
        "packaged_keeper_exact": member_keeper_error <= 1e-6,
        "packaged_candidate_exact": member_candidate_error <= 1e-6,
        "raw_blend_formula_exact": raw_blend_formula_error <= 1e-7,
        "decision_score_formula_exact": decision_score_formula_error <= 1e-7,
        "deployment_probability_formula_exact": deployment_formula_error <= 1e-7,
        "decision_argmax_exact": decision_mismatches == 0,
        "deployment_argmax_exact": deployment_mismatches == 0,
        "softmax_preserves_decision": softmax_decision_mismatches == 0,
        "metrics_exact": metric_error is not None and metric_error <= 1e-9,
    }
    rows: List[Dict[str, object]] = []
    for index in range(expected_count):
        expected_row = expected_base_rows[index]
        row: Dict[str, object] = {
            "sample_index": expected_keys[index],
            "image_path": _source_path(expected_row),
            "target_index": int(observed_targets_array[index]),
            "expected_prediction_index": int(expected_predictions[index]),
            "packaged_prediction_index": int(packaged_predictions[index]),
            "decision_prediction_index": int(decision_predictions[index]),
            "focus_margin_expected": float(expected_focus_margin[index]),
            "max_member_error": float(
                max(
                    np.max(np.abs(packaged_keeper[index] - independent_keeper[index])),
                    np.max(
                        np.abs(packaged_candidate[index] - independent_candidate[index])
                    ),
                )
            ),
            "max_raw_blend_error": float(
                np.max(np.abs(raw_blend[index] - expected_raw_blend[index]))
            ),
        }
        for class_index, class_name in enumerate(class_names):
            row[f"raw_blend_{class_index}_{class_name}"] = float(
                raw_blend[index, class_index]
            )
            row[f"deployment_prob_{class_index}_{class_name}"] = float(
                deployment_probabilities[index, class_index]
            )
        rows.append(row)

    output_dir.mkdir(parents=True, exist_ok=True)
    predictions_path = output_dir / "predictions_detailed.csv"
    mismatch_path = output_dir / "mismatches.csv"
    _write_csv(predictions_path, rows)
    mismatch_rows = [
        row
        for row in rows
        if int(row["expected_prediction_index"])
        != int(row["packaged_prediction_index"])
    ]
    _write_csv(mismatch_path, mismatch_rows)
    summary: Dict[str, object] = {
        "mode": "precision_ensemble_checkpoint_pytorch_parity",
        "test_data_used": False,
        "checkpoint": str(Path(checkpoint_path).resolve()),
        "checkpoint_sha256": checkpoint_hash,
        "keeper_checkpoint_sha256": keeper_hash,
        "candidate_checkpoint_sha256": candidate_hash,
        "frozen_protocol_sha256": summary_hash,
        "data_yaml": str(Path(data_yaml).resolve()),
        "device": str(device),
        "rows": expected_count,
        "full_expected_rows": len(expected_keys),
        "errors": {
            "packaged_vs_independent_keeper_max_abs": member_keeper_error,
            "packaged_vs_independent_candidate_max_abs": member_candidate_error,
            "keeper_vs_frozen_csv_max_abs": keeper_csv_error,
            "candidate_vs_frozen_csv_max_abs": candidate_csv_error,
            "raw_blend_vs_frozen_max_abs": raw_blend_error,
            "raw_blend_formula_max_abs": raw_blend_formula_error,
            "decision_score_formula_max_abs": decision_score_formula_error,
            "deployment_probability_formula_max_abs": deployment_formula_error,
            "metric_max_abs": metric_error,
        },
        "mismatches": {
            "decision_vs_frozen": decision_mismatches,
            "deployment_vs_frozen": deployment_mismatches,
            "deployment_vs_decision": softmax_decision_mismatches,
        },
        "metrics": metrics,
        "historical_probability_replay": {
            "gate": False,
            "reason": (
                "Archived CSV probabilities are runtime/kernel telemetry. Exact packaged-vs-"
                "independent member parity, frozen decisions, and metrics are the gates."
            ),
            "keeper_max_abs": keeper_csv_error,
            "candidate_max_abs": candidate_csv_error,
            "raw_blend_max_abs": raw_blend_error,
        },
        "gates": gates,
        "all_gates_passed": all(gates.values()),
        "artifacts": {
            "predictions": predictions_path.name,
            "predictions_sha256": _sha256(predictions_path),
            "mismatches": mismatch_path.name,
            "mismatches_sha256": _sha256(mismatch_path),
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
        description="Audit exact PyTorch parity of a packaged precision ensemble."
    )
    parser.add_argument("--checkpoint", type=Path, required=True)
    parser.add_argument("--keeper", type=Path, required=True)
    parser.add_argument("--candidate", type=Path, required=True)
    parser.add_argument("--frozen-summary", type=Path, required=True)
    parser.add_argument("--data", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--expected-checkpoint-sha256", required=True)
    parser.add_argument("--expected-keeper-sha256", required=True)
    parser.add_argument("--expected-candidate-sha256", required=True)
    parser.add_argument("--expected-summary-sha256", required=True)
    parser.add_argument("--batch-size", type=int, default=64)
    parser.add_argument("--max-samples", type=int, default=0)
    parser.add_argument("--device", choices=("auto", "cpu", "cuda"), default="auto")
    return parser.parse_args(argv)


def main(argv: Sequence[str] | None = None) -> int:
    args = parse_args(argv)
    set_seed(42, deterministic=False)
    if args.device == "auto":
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    else:
        device = torch.device(args.device)
    summary = audit_checkpoint(
        checkpoint_path=args.checkpoint.expanduser().resolve(),
        keeper_path=args.keeper.expanduser().resolve(),
        candidate_path=args.candidate.expanduser().resolve(),
        frozen_summary_path=args.frozen_summary.expanduser().resolve(),
        data_yaml=args.data.expanduser().resolve(),
        output_dir=args.output_dir.expanduser().resolve(),
        expected_checkpoint_sha256=args.expected_checkpoint_sha256,
        expected_keeper_sha256=args.expected_keeper_sha256,
        expected_candidate_sha256=args.expected_candidate_sha256,
        expected_summary_sha256=args.expected_summary_sha256,
        batch_size=args.batch_size,
        max_samples=args.max_samples,
        device=device,
    )
    print(
        json.dumps(
            {
                "rows": summary["rows"],
                "all_gates_passed": summary["all_gates_passed"],
                "errors": summary["errors"],
                "mismatches": summary["mismatches"],
            },
            ensure_ascii=True,
        )
    )
    return 0 if bool(summary["all_gates_passed"]) else 2


if __name__ == "__main__":
    raise SystemExit(main())
