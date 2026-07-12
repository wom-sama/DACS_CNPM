from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path
from typing import Dict, List, Mapping, Optional, Sequence

import numpy as np
import torch
from PIL import Image, ImageDraw
from torch import nn
from torch.utils.data import Dataset

from trkh.core.config import load_data_spec
from trkh.core.utils import ensure_dir, json_dump, load_checkpoint, set_seed
from trkh.evaluation.input_normalization import checkpoint_input_normalization
from trkh.models.feature_hooks import resolve_feature_hook
from trkh.models.model import build_model_from_checkpoint
from trkh.tools.audit_finer_cam_class1_readiness import (
    FEATURE_SOURCE,
    _checkpoint_sha256,
    _direction_summary,
    _disable_inplace_modules,
    _group_summary,
    _is_relative_to,
    _masked_preview,
    _overlay,
    _tensor_to_image,
    _write_artifact_manifest,
    collect_split,
)
from trkh.tools.evaluate_paired_view_fusion import (
    _build_classification_dataset,
    _build_eval_transform_from_checkpoint,
)


class SelectedIndexDataset(Dataset):
    def __init__(self, dataset: Dataset, indices: Sequence[int]) -> None:
        self.dataset = dataset
        self.indices = [int(value) for value in indices]
        if not self.indices:
            raise ValueError("at least one review index is required")
        if min(self.indices) < 0 or max(self.indices) >= len(dataset):
            raise IndexError("review index is outside the validation dataset")
        if len(set(self.indices)) != len(self.indices):
            raise ValueError("review indices must be unique")
        sample_paths = getattr(dataset, "sample_paths", None)
        if not callable(sample_paths):
            raise TypeError("validation dataset must expose sample_paths()")
        self._all_paths = [str(value) for value in sample_paths()]

    def __len__(self) -> int:
        return len(self.indices)

    def __getitem__(self, position: int):
        return self.dataset[self.indices[int(position)]]

    def sample_paths(self) -> List[str]:
        return [self._all_paths[index] for index in self.indices]


def parse_args(argv: Optional[Sequence[str]] = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Recompute and render every changed validation case from a locked "
            "Finer-CAM readiness CSV. No fitting, test access, or raw-data edit."
        )
    )
    parser.add_argument("--checkpoint", type=Path, required=True)
    parser.add_argument("--data", type=Path, required=True)
    parser.add_argument("--case-csv", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--batch-size", type=int, default=16)
    parser.add_argument("--num-workers", type=int, default=0)
    parser.add_argument("--seed", type=int, default=20260712)
    return parser.parse_args(argv)


def load_changed_case_rows(path: Path) -> List[Dict[str, object]]:
    with Path(path).open("r", encoding="utf-8-sig", newline="") as handle:
        rows = [dict(row) for row in csv.DictReader(handle)]
    required = {
        "split",
        "sample_index",
        "target_index",
        "global_prediction_index",
        "candidate_prediction_index",
    }
    if not rows or not required.issubset(rows[0]):
        raise ValueError("case CSV is missing the locked prediction columns")
    changed = [
        row
        for row in rows
        if str(row["split"]).strip().lower() == "val"
        and int(row["global_prediction_index"]) != int(row["candidate_prediction_index"])
    ]
    if not changed:
        raise ValueError("case CSV contains no changed validation rows")
    indices = [int(row["sample_index"]) for row in changed]
    if len(set(indices)) != len(indices):
        raise ValueError("changed validation sample_index values must be unique")
    return changed


def expand_batch_context_indices(
    changed_indices: Sequence[int],
    *,
    dataset_length: int,
    batch_size: int,
) -> List[int]:
    if int(dataset_length) <= 0 or int(batch_size) <= 0:
        raise ValueError("dataset_length and batch_size must be positive")
    result: set[int] = set()
    for value in changed_indices:
        index = int(value)
        if index < 0 or index >= int(dataset_length):
            raise IndexError("changed index is outside the validation dataset")
        start = (index // int(batch_size)) * int(batch_size)
        stop = min(start + int(batch_size), int(dataset_length))
        result.update(range(start, stop))
    return sorted(result)


def _render_all_cases(
    path: Path,
    *,
    dataset: SelectedIndexDataset,
    result: Mapping[str, object],
    expected_rows: Sequence[Mapping[str, object]],
    selected_positions: Sequence[int],
    mean: Sequence[float],
    std: Sequence[float],
) -> None:
    rows = result["rows"]
    standard_cams = result["standard_cams"]
    finer_cams = result["finer_cams"]
    thumb = 150
    header = 38
    columns = ("input", "Grad-CAM", "Finer-CAM", "Grad mask 5%", "Finer mask 5%")
    canvas = Image.new(
        "RGB",
        (thumb * len(columns), (thumb + header) * len(selected_positions)),
        "white",
    )
    draw = ImageDraw.Draw(canvas)
    for display_position, (context_position, expected) in enumerate(
        zip(selected_positions, expected_rows)
    ):
        row = rows[int(context_position)]
        image_tensor, _, metadata = dataset[int(context_position)]
        image = _tensor_to_image(image_tensor, mean, std)
        valid_mask = metadata.get("image_mask") if isinstance(metadata, Mapping) else None
        if not torch.is_tensor(valid_mask):
            valid_mask = torch.ones(image.height, image.width, dtype=torch.bool)
        if valid_mask.ndim == 3 and int(valid_mask.size(0)) == 1:
            valid_mask = valid_mask[0]
        standard = np.asarray(standard_cams[int(context_position)], dtype=np.float32)
        finer = np.asarray(finer_cams[int(context_position)], dtype=np.float32)
        views = (
            image,
            _overlay(image, standard),
            _overlay(image, finer),
            _masked_preview(image, standard, valid_mask, mean),
            _masked_preview(image, finer, valid_mask, mean),
        )
        y = display_position * (thumb + header)
        title = (
            f"idx={int(expected['sample_index'])} y={int(row['target_index'])} "
            f"g={int(row['global_prediction_index'])} "
            f"c={int(row['candidate_prediction_index'])} "
            f"dRD={float(row['finer_minus_standard_relative_drop']):+.3f}"
        )
        draw.text((3, y + 2), title, fill="black")
        for column_index, (name, view) in enumerate(zip(columns, views)):
            rendered = view.copy()
            rendered.thumbnail((thumb - 4, thumb - 20), Image.Resampling.LANCZOS)
            x = column_index * thumb
            draw.text((x + 3, y + header), name, fill="black")
            image_y = y + header + 18
            canvas.paste(rendered, (x + (thumb - rendered.width) // 2, image_y))
    canvas.save(path)


def _assert_reproduction(
    result_rows: Sequence[Mapping[str, object]],
    expected_rows: Sequence[Mapping[str, object]],
    context_indices: Sequence[int],
    *,
    numeric_tolerance: float = 1e-5,
) -> tuple[List[Dict[str, object]], List[int], float]:
    if len(result_rows) != len(context_indices):
        raise ValueError("review row count differs from batch-context indices")
    position_by_index = {int(index): position for position, index in enumerate(context_indices)}
    output: List[Dict[str, object]] = []
    selected_positions: List[int] = []
    maximum_numeric_difference = 0.0
    numeric_keys = (
        "standard_relative_drop",
        "finer_relative_drop",
        "finer_minus_standard_relative_drop",
    )
    for expected in expected_rows:
        original_index = int(expected["sample_index"])
        if original_index not in position_by_index:
            raise ValueError(f"missing batch context for sample_index={original_index}")
        position = int(position_by_index[original_index])
        actual = result_rows[position]
        checks = {
            "target": int(actual["target_index"]) == int(expected["target_index"]),
            "global_prediction": int(actual["global_prediction_index"])
            == int(expected["global_prediction_index"]),
            "candidate_prediction": int(actual["candidate_prediction_index"])
            == int(expected["candidate_prediction_index"]),
        }
        if not all(checks.values()):
            raise ValueError(
                f"changed-case reproduction failed at position {position}: {checks}"
            )
        for key in numeric_keys:
            difference = abs(float(actual[key]) - float(expected[key]))
            maximum_numeric_difference = max(maximum_numeric_difference, difference)
            if difference > float(numeric_tolerance):
                raise ValueError(
                    f"changed-case numeric reproduction failed for {original_index}/{key}: "
                    f"difference={difference}"
                )
        row = dict(actual)
        row["sample_index"] = original_index
        row["review_position"] = int(position)
        output.append(row)
        selected_positions.append(position)
    return output, selected_positions, float(maximum_numeric_difference)


def _transition_summary(rows: Sequence[Mapping[str, object]]) -> Dict[str, int]:
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
        global_prediction = int(row["global_prediction_index"])
        candidate_prediction = int(row["candidate_prediction_index"])
        global_correct = global_prediction == target
        candidate_correct = candidate_prediction == target
        if not global_correct and candidate_correct:
            result["corrections"] += 1
        elif global_correct and not candidate_correct:
            result["harms"] += 1
        else:
            result["neutral"] += 1
        if target == 1 and global_prediction != 1 and candidate_prediction == 1:
            result["class1_fn_rescued"] += 1
        if target == 1 and global_prediction == 1 and candidate_prediction != 1:
            result["class1_tp_broken"] += 1
        if target != 1 and global_prediction == 1 and candidate_prediction != 1:
            result["class1_fp_removed"] += 1
        if target != 1 and global_prediction != 1 and candidate_prediction == 1:
            result["class1_fp_created"] += 1
    return result


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


def run_review(args: argparse.Namespace) -> Dict[str, object]:
    if int(args.batch_size) < 1 or int(args.num_workers) < 0:
        raise ValueError("invalid batch/worker settings")
    output_dir = Path(args.output_dir).resolve()
    data_path = Path(args.data).resolve()
    if _is_relative_to(output_dir, data_path.parent):
        raise ValueError("output directory must stay outside the raw dataset")
    if output_dir.exists() and any(output_dir.iterdir()):
        raise FileExistsError(f"output directory is not empty: {output_dir}")
    ensure_dir(output_dir)
    set_seed(int(args.seed))
    expected_rows = load_changed_case_rows(Path(args.case_csv))
    indices = [int(row["sample_index"]) for row in expected_rows]

    checkpoint_path = Path(args.checkpoint).resolve()
    checkpoint = load_checkpoint(checkpoint_path, map_location="cpu")
    data_spec = load_data_spec(data_path, class_name_mode="raw", expected_num_classes=5)
    class_names = list(checkpoint.get("class_names", data_spec.class_names))
    if class_names != list(data_spec.class_names):
        raise ValueError("checkpoint and dataset class orders differ")
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = build_model_from_checkpoint(dict(checkpoint), num_classes=len(class_names))
    model.to(device)
    model.eval()
    disabled_inplace_modules = _disable_inplace_modules(model)
    feature_spec = resolve_feature_hook(model, feature_source=FEATURE_SOURCE)
    image_size = int(checkpoint.get("model_config", {}).get("image_size", 256))
    transform = _build_eval_transform_from_checkpoint(dict(checkpoint), image_size=image_size)
    base_dataset = _build_classification_dataset(
        data_spec=data_spec,
        split="val",
        transform=transform,
        checkpoint=dict(checkpoint),
    )
    context_indices = expand_batch_context_indices(
        indices,
        dataset_length=len(base_dataset),
        batch_size=int(args.batch_size),
    )
    dataset = SelectedIndexDataset(base_dataset, context_indices)
    bbox_token_prior_source = str(
        checkpoint.get("model_config", {}).get("bbox_token_prior_source", "crop_bbox")
        or "crop_bbox"
    )
    result = collect_split(
        split="val_changed_review",
        dataset=dataset,
        model=model,
        feature_module=feature_spec.module,
        class_names=class_names,
        device=device,
        batch_size=int(args.batch_size),
        num_workers=int(args.num_workers),
        bbox_token_prior_source=bbox_token_prior_source,
    )
    reproduced_rows, selected_positions, maximum_numeric_difference = _assert_reproduction(
        result["rows"],
        expected_rows,
        context_indices,
    )
    _write_csv(output_dir / "changed_cases_reproduced.csv", reproduced_rows)
    mean, std = checkpoint_input_normalization(checkpoint)
    _render_all_cases(
        output_dir / "all_changed_cases_finer_cam_contact_sheet.png",
        dataset=dataset,
        result=result,
        expected_rows=expected_rows,
        selected_positions=selected_positions,
        mean=mean,
        std=std,
    )
    summary = {
        "mode": "finer_cam_changed_case_review",
        "guardrail": (
            "Validation changed-case XAI reproduction only; no test, fit, sweep, "
            "raw-data edit, model, checkpoint, or trainable manifest."
        ),
        "checkpoint": str(checkpoint_path),
        "checkpoint_sha256": _checkpoint_sha256(checkpoint_path),
        "data": str(data_path),
        "source_case_csv": str(Path(args.case_csv).resolve()),
        "source_case_csv_sha256": _checkpoint_sha256(Path(args.case_csv).resolve()),
        "feature_source": feature_spec.source,
        "sample_count": int(len(indices)),
        "sample_indices": indices,
        "batch_context_sample_count": int(len(context_indices)),
        "batch_context_indices": context_indices,
        "batch_size_reproduced": int(args.batch_size),
        "all_predictions_reproduced": True,
        "relative_drop_numeric_tolerance": 1e-5,
        "maximum_relative_drop_abs_difference": maximum_numeric_difference,
        "transitions": _transition_summary(reproduced_rows),
        "direction": _direction_summary(reproduced_rows),
        "groups": _group_summary(reproduced_rows),
        "disabled_inplace_modules_for_gradient_hooks": disabled_inplace_modules,
        "test_split_used": False,
        "raw_dataset_modified": False,
        "model_or_checkpoint_written": False,
    }
    json_dump(output_dir / "summary.json", summary)
    (output_dir / "README.md").write_text(
        "# Finer-CAM Changed-Case Review\n\n"
        f"- Reproduced all `{len(indices)}` changed validation rows from the locked full audit.\n"
        "- Includes every transition in one Grad-CAM/Finer-CAM/masked-view contact sheet.\n"
        "- No test access, fitting, sweep, raw-data edit, model, or checkpoint.\n",
        encoding="utf-8",
    )
    _write_artifact_manifest(output_dir)
    return summary


def main() -> None:
    summary = run_review(parse_args())
    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()
