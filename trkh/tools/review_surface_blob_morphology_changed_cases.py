from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path
from typing import Dict, List, Mapping, Optional, Sequence, Tuple

import numpy as np
import torch
from PIL import Image, ImageDraw, ImageFont
from torchvision.ops import roi_align

from trkh.evaluation.input_normalization import checkpoint_input_normalization
from trkh.tools.audit_lbp_surface_texture_readiness import (
    _is_relative_to,
    _normalize_path,
)
from trkh.tools.audit_surface_blob_morphology_readiness import (
    _heatmap,
    _write_artifact_manifest,
    interior_roi_boxes,
    surface_blob_descriptor,
)
from trkh.tools.probe_embedding_prototypes import (
    _build_dataset,
    _collate_classification,
)


DEFAULT_MAX_REVIEW_ROWS = 24
DEFAULT_REPRODUCTION_BATCH_SIZE = 64


def parse_args(argv: Optional[Sequence[str]] = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Review locked validation transitions from the surface-blob morphology "
            "audit and exactly reproduce selected descriptors. No fit, test access, "
            "raw-data edit, model, or checkpoint."
        )
    )
    parser.add_argument("--data", type=Path, required=True)
    parser.add_argument("--checkpoint", type=Path, required=True)
    parser.add_argument("--val-predictions", type=Path, required=True)
    parser.add_argument("--descriptor-cache", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--class-name-mode", type=str, default="raw")
    parser.add_argument("--max-review-rows", type=int, default=DEFAULT_MAX_REVIEW_ROWS)
    parser.add_argument(
        "--reproduction-batch-size",
        type=int,
        default=DEFAULT_REPRODUCTION_BATCH_SIZE,
        help="Original full-audit batch size used to reproduce CUDA ROI extraction.",
    )
    parser.add_argument("--device", choices=("cuda", "cpu"), default="cuda")
    parser.add_argument("--roi-size", type=int, default=128)
    parser.add_argument("--interior-erode-ratio", type=float, default=0.15)
    return parser.parse_args(argv)


def transition_kind(row: Mapping[str, object]) -> str:
    target = int(row["target_index"])
    keeper = int(row["keeper_prediction_index"])
    candidate = int(row["keeper_candidate_residual_prediction_index"])
    if target == 1 and keeper != 1 and candidate == 1:
        return "class1_fn_rescued"
    if target == 1 and keeper == 1 and candidate != 1:
        return "class1_tp_broken"
    if target != 1 and keeper == 1 and candidate != 1:
        return "class1_fp_removed"
    if target != 1 and keeper != 1 and candidate == 1:
        return "class1_fp_created"
    if keeper != target and candidate == target:
        return "correction"
    if keeper == target and candidate != target:
        return "harm"
    return "neutral"


def load_changed_rows(path: Path) -> List[Dict[str, object]]:
    with Path(path).open("r", encoding="utf-8-sig", newline="") as handle:
        raw_rows = [dict(row) for row in csv.DictReader(handle)]
    required = {
        "split",
        "sample_index",
        "target_index",
        "keeper_prediction_index",
        "keeper_control_residual_prediction_index",
        "keeper_candidate_residual_prediction_index",
    }
    if not raw_rows or not required.issubset(raw_rows[0]):
        raise ValueError("validation predictions are missing locked columns")
    changed = []
    for raw in raw_rows:
        if str(raw["split"]).strip().lower() != "val":
            continue
        keeper = int(raw["keeper_prediction_index"])
        candidate = int(raw["keeper_candidate_residual_prediction_index"])
        if keeper == candidate:
            continue
        row: Dict[str, object] = dict(raw)
        row["sample_index"] = int(raw["sample_index"])
        row["target_index"] = int(raw["target_index"])
        row["keeper_prediction_index"] = keeper
        row["keeper_control_residual_prediction_index"] = int(
            raw["keeper_control_residual_prediction_index"]
        )
        row["keeper_candidate_residual_prediction_index"] = candidate
        row["transition_kind"] = transition_kind(row)
        changed.append(row)
    if not changed:
        raise ValueError("validation predictions contain no candidate-vs-keeper changes")
    indices = [int(row["sample_index"]) for row in changed]
    if len(set(indices)) != len(indices):
        raise ValueError("changed sample_index values must be unique")
    return changed


def transition_summary(rows: Sequence[Mapping[str, object]]) -> Dict[str, int]:
    result = {
        "changed": int(len(rows)),
        "corrections": 0,
        "harms": 0,
        "neutral": 0,
        "class1_fn_rescued": 0,
        "class1_tp_broken": 0,
        "class1_fp_removed": 0,
        "class1_fp_created": 0,
    }
    for row in rows:
        target = int(row["target_index"])
        keeper = int(row["keeper_prediction_index"])
        candidate = int(row["keeper_candidate_residual_prediction_index"])
        if keeper != target and candidate == target:
            result["corrections"] += 1
        elif keeper == target and candidate != target:
            result["harms"] += 1
        else:
            result["neutral"] += 1
        kind = transition_kind(row)
        if kind.startswith("class1_") and kind in result:
            result[kind] += 1
    return result


def select_review_rows(
    rows: Sequence[Mapping[str, object]],
    *,
    maximum: int = DEFAULT_MAX_REVIEW_ROWS,
) -> List[Dict[str, object]]:
    if int(maximum) <= 0:
        raise ValueError("maximum must be positive")
    order = (
        "class1_fn_rescued",
        "class1_tp_broken",
        "class1_fp_removed",
        "class1_fp_created",
        "correction",
        "harm",
        "neutral",
    )
    buckets = {
        name: sorted(
            [dict(row) for row in rows if transition_kind(row) == name],
            key=lambda row: int(row["sample_index"]),
        )
        for name in order
    }
    selected: List[Dict[str, object]] = []
    while len(selected) < int(maximum) and any(buckets.values()):
        for name in order:
            if buckets[name] and len(selected) < int(maximum):
                selected.append(buckets[name].pop(0))
    return selected


def reproduction_batch_windows(
    sample_indices: Sequence[int],
    *,
    dataset_size: int,
    batch_size: int,
) -> List[Tuple[int, int, List[int]]]:
    if int(dataset_size) <= 0 or int(batch_size) <= 0:
        raise ValueError("dataset_size and batch_size must be positive")
    grouped: Dict[int, List[int]] = {}
    for raw_index in sample_indices:
        index = int(raw_index)
        if index < 0 or index >= int(dataset_size):
            raise ValueError(f"sample_index is outside the dataset: {index}")
        start = (index // int(batch_size)) * int(batch_size)
        grouped.setdefault(start, []).append(index)
    return [
        (
            int(start),
            min(int(start) + int(batch_size), int(dataset_size)),
            sorted(indices),
        )
        for start, indices in sorted(grouped.items())
    ]


def _write_csv(path: Path, rows: Sequence[Mapping[str, object]]) -> None:
    values = [dict(row) for row in rows]
    fieldnames: List[str] = []
    for row in values:
        for key in row:
            if key not in fieldnames:
                fieldnames.append(key)
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(values)


def _render_contact_sheet(
    path: Path,
    rows: Sequence[Mapping[str, object]],
) -> None:
    tile = 150
    header = 48
    columns = 5
    canvas = Image.new("RGB", (columns * tile, len(rows) * (tile + header)), "white")
    draw = ImageDraw.Draw(canvas)
    font = ImageFont.load_default()
    for row_index, row in enumerate(rows):
        y = row_index * (tile + header)
        rgb = Image.fromarray(np.asarray(row["rgb"], dtype=np.uint8))
        canvas.paste(rgb.resize((tile, tile), Image.Resampling.BILINEAR), (0, y))
        maps = row["maps"]
        highlight = np.asarray(row["rgb"], dtype=np.float32).copy()
        highlight_mask = np.asarray(maps["highlight"], dtype=bool)
        highlight[highlight_mask] = 0.45 * highlight[highlight_mask] + 0.55 * np.asarray(
            (255.0, 40.0, 40.0), dtype=np.float32
        )
        canvas.paste(
            Image.fromarray(highlight.clip(0, 255).astype(np.uint8)).resize((tile, tile)),
            (tile, y),
        )
        for column, key in enumerate(("dark_blob", "bright_blob", "chroma_blob"), start=2):
            canvas.paste(
                _heatmap(np.asarray(maps[key])).resize((tile, tile), Image.Resampling.BILINEAR),
                (column * tile, y),
            )
        draw.text(
            (4, y + tile + 3),
            (
                f"{row['transition_kind']} idx={row['sample_index']} y={row['target_index']} "
                f"keeper={row['keeper_prediction_index']} control={row['keeper_control_residual_prediction_index']} "
                f"candidate={row['keeper_candidate_residual_prediction_index']}"
            )[:125],
            fill=(0, 0, 0),
            font=font,
        )
        draw.text(
            (4, y + tile + 21),
            (
                f"RGB | highlight={row['highlight_fraction']:.3f} | dark | bright | chroma; "
                f"max descriptor diff={row['max_descriptor_abs_difference']:.2e}"
            ),
            fill=(0, 0, 0),
            font=font,
        )
    canvas.save(path)


def run_review(args: argparse.Namespace) -> Dict[str, object]:
    if (
        int(args.max_review_rows) <= 0
        or int(args.reproduction_batch_size) <= 0
        or int(args.roi_size) < 64
    ):
        raise ValueError("review/ROI settings are invalid")
    if str(args.device) == "cuda" and not torch.cuda.is_available():
        raise RuntimeError("CUDA is required to reproduce the locked full-audit ROI extraction")
    device = torch.device(str(args.device))
    output_dir = Path(args.output_dir).resolve()
    data_path = Path(args.data).resolve()
    if _is_relative_to(output_dir, data_path.parent):
        raise ValueError("output directory must stay outside the raw dataset")
    if output_dir.exists() and any(output_dir.iterdir()):
        raise FileExistsError(f"output directory must be empty: {output_dir}")
    output_dir.mkdir(parents=True, exist_ok=True)
    changed_rows = load_changed_rows(Path(args.val_predictions))
    selected_rows = select_review_rows(changed_rows, maximum=int(args.max_review_rows))

    checkpoint = torch.load(Path(args.checkpoint), map_location="cpu", weights_only=False)
    if not isinstance(checkpoint, Mapping):
        raise ValueError("checkpoint payload must be a mapping")
    dataset, _class_names = _build_dataset(
        data_yaml=data_path,
        split="val",
        checkpoint=checkpoint,
        class_name_mode=str(args.class_name_mode),
        max_samples=0,
    )
    # Paths are stored as NumPy object strings by the locked full audit.
    descriptor_cache = np.load(Path(args.descriptor_cache), allow_pickle=True)
    required_cache = {
        "val_control",
        "val_morphology",
        "val_labels",
        "val_sample_index",
        "val_paths",
    }
    if not required_cache.issubset(descriptor_cache.files):
        raise ValueError("descriptor cache is missing validation arrays")
    if len(dataset) != int(descriptor_cache["val_labels"].shape[0]):
        raise ValueError("dataset and descriptor cache row counts differ")
    if not np.array_equal(
        np.asarray(descriptor_cache["val_sample_index"], dtype=np.int64),
        np.arange(len(dataset), dtype=np.int64),
    ):
        raise ValueError("descriptor cache sample_index is not deterministic")
    dataset_labels = np.asarray(dataset.labels(), dtype=np.int64)
    if not np.array_equal(dataset_labels, np.asarray(descriptor_cache["val_labels"], dtype=np.int64)):
        raise ValueError("dataset and descriptor cache labels differ")
    dataset_paths = [_normalize_path(value) for value in dataset.sample_paths()]
    cache_paths = [_normalize_path(value) for value in descriptor_cache["val_paths"]]
    if dataset_paths != cache_paths:
        raise ValueError("dataset and descriptor cache paths differ")

    mean, std = checkpoint_input_normalization(checkpoint)
    mean_tensor = torch.tensor(mean, dtype=torch.float32, device=device).view(1, 3, 1, 1)
    std_tensor = torch.tensor(std, dtype=torch.float32, device=device).view(1, 3, 1, 1)
    selected_by_index = {int(row["sample_index"]): dict(row) for row in selected_rows}
    reproduced_by_index: Dict[int, Dict[str, object]] = {}
    maximum_difference = 0.0
    batch_windows = reproduction_batch_windows(
        list(selected_by_index),
        dataset_size=len(dataset),
        batch_size=int(args.reproduction_batch_size),
    )
    batch_context_rows = 0
    for batch_start, batch_stop, selected_indices in batch_windows:
        images, targets, metadata = _collate_classification(
            [dataset[index] for index in range(batch_start, batch_stop)]
        )
        if not isinstance(metadata, Mapping) or not torch.is_tensor(metadata.get("crop_bbox")):
            raise ValueError("classification crop_bbox metadata is required")
        images = images.to(device=device, dtype=torch.float32, non_blocking=True)
        crop_bbox = metadata["crop_bbox"].to(
            device=device,
            dtype=torch.float32,
            non_blocking=True,
        )
        rgb = (images * std_tensor + mean_tensor).clamp(0.0, 1.0)
        boxes = interior_roi_boxes(
            crop_bbox,
            image_height=int(rgb.size(2)),
            image_width=int(rgb.size(3)),
            erode_ratio=float(args.interior_erode_ratio),
        )
        roi = roi_align(
            rgb,
            boxes,
            output_size=(int(args.roi_size), int(args.roi_size)),
            spatial_scale=1.0,
            sampling_ratio=2,
            aligned=True,
        )
        rgb_uint8_batch = (
            roi.mul(255.0)
            .round()
            .clamp(0.0, 255.0)
            .to(dtype=torch.uint8)
            .permute(0, 2, 3, 1)
            .cpu()
            .numpy()
        )
        batch_context_rows += int(batch_stop - batch_start)
        for index in selected_indices:
            row = selected_by_index[index]
            local_index = int(index - batch_start)
            if int(targets[local_index]) != int(row["target_index"]):
                raise ValueError("selected target differs from runtime dataset")
            rgb_uint8 = rgb_uint8_batch[local_index]
            control, morphology, maps, telemetry = surface_blob_descriptor(rgb_uint8)
            control_difference = float(
                np.max(
                    np.abs(
                        control
                        - np.asarray(descriptor_cache["val_control"][index], dtype=np.float32)
                    )
                )
            )
            morphology_difference = float(
                np.max(
                    np.abs(
                        morphology
                        - np.asarray(
                            descriptor_cache["val_morphology"][index], dtype=np.float32
                        )
                    )
                )
            )
            difference = max(control_difference, morphology_difference)
            maximum_difference = max(maximum_difference, difference)
            if difference > 1e-7:
                raise ValueError(
                    f"descriptor reproduction failed for sample_index={index}: {difference}"
                )
            review_row = dict(row)
            review_row.update(
                {
                    "control_max_abs_difference": control_difference,
                    "morphology_max_abs_difference": morphology_difference,
                    "max_descriptor_abs_difference": difference,
                    "highlight_fraction": float(telemetry["highlight_fraction"]),
                    "dark_peak_density_sum": float(telemetry["dark_peak_density_sum"]),
                    "bright_peak_density_sum": float(telemetry["bright_peak_density_sum"]),
                    "chroma_peak_density_sum": float(telemetry["chroma_peak_density_sum"]),
                    "rgb": rgb_uint8,
                    "maps": maps,
                }
            )
            reproduced_by_index[index] = review_row
    reproduced = [reproduced_by_index[int(row["sample_index"])] for row in selected_rows]

    all_changed_export = [
        {key: value for key, value in row.items() if key not in {"rgb", "maps"}}
        for row in changed_rows
    ]
    selected_export = [
        {key: value for key, value in row.items() if key not in {"rgb", "maps"}}
        for row in reproduced
    ]
    _write_csv(output_dir / "all_changed_cases.csv", all_changed_export)
    _write_csv(output_dir / "selected_reproduced_cases.csv", selected_export)
    _render_contact_sheet(output_dir / "selected_changed_cases_contact_sheet.png", reproduced)
    summary = {
        "mode": "surface_blob_morphology_changed_case_review",
        "guardrail": (
            "Locked validation review only; no fitting, test access, threshold/feature "
            "sweep, raw-data edit, model, checkpoint, or trainable manifest."
        ),
        "source_val_predictions": str(Path(args.val_predictions).resolve()),
        "source_descriptor_cache": str(Path(args.descriptor_cache).resolve()),
        "data": str(data_path),
        "checkpoint": str(Path(args.checkpoint).resolve()),
        "all_changed_transitions": transition_summary(changed_rows),
        "selected_rows": int(len(reproduced)),
        "reproduction_device": str(device),
        "reproduction_batch_size": int(args.reproduction_batch_size),
        "reproduction_batch_windows": int(len(batch_windows)),
        "reproduction_batch_context_rows": int(batch_context_rows),
        "selected_kind_counts": {
            name: int(sum(str(row["transition_kind"]) == name for row in reproduced))
            for name in sorted({str(row["transition_kind"]) for row in reproduced})
        },
        "all_selected_descriptors_reproduced": True,
        "maximum_descriptor_abs_difference": float(maximum_difference),
        "test_split_used": False,
        "raw_dataset_modified": False,
        "model_or_checkpoint_written": False,
    }
    (output_dir / "summary.json").write_text(json.dumps(summary, indent=2), encoding="utf-8")
    (output_dir / "README.md").write_text(
        "# Surface-Blob Morphology Changed-Case Review\n\n"
        f"- Audited all `{len(changed_rows)}` candidate-vs-keeper validation transitions.\n"
        f"- Recomputed `{len(reproduced)}` prioritized ROI descriptors with max abs difference `{maximum_difference:.3e}`.\n"
        f"- Recreated `{len(batch_windows)}` original `{args.reproduction_batch_size}`-row `{device}` batch windows.\n"
        "- No fitting, test access, raw-data edit, model, checkpoint, or training manifest.\n",
        encoding="utf-8",
    )
    _write_artifact_manifest(
        output_dir,
        mode="surface_blob_morphology_changed_case_review_manifest",
    )
    return summary


def main(argv: Optional[Sequence[str]] = None) -> int:
    summary = run_review(parse_args(argv))
    print(json.dumps(summary, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
