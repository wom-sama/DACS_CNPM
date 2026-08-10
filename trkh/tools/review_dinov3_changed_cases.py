from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path
from typing import Dict, List, Mapping, Optional, Sequence, Tuple

import numpy as np
import torch
from PIL import Image, ImageDraw, ImageFont

from trkh.tools.audit_dinov2_dense_patch_readiness import (
    PathImageFolder,
    _patch_energy_map,
)
from trkh.tools.audit_dinov3_dense_patch_readiness import (
    _build_model_and_transform,
    _resolve_device,
)
from trkh.tools.audit_multistage_teacher_feature_readiness import (
    _alignment_plan,
    _heat_overlay,
    _imagefolder_rows,
)
from trkh.tools.audit_wavelet_scattering_readiness import (
    _assert_output_outside_datasets,
)
from trkh.tools.probe_api_pairwise_interaction_readiness import (
    _write_artifact_manifest,
)


TRANSITION_ORDER = (
    "class1_fn_rescued",
    "class1_tp_broken",
    "class1_fp_removed",
    "class1_fp_created",
)
DEFAULT_PER_TRANSITION = 8


def _parse_args(argv: Optional[Sequence[str]] = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Validation-only visual review of keeper-to-DINOv3 class-1 changes "
            "with matched FP32 DINOv2/DINOv3 patch-energy maps."
        )
    )
    parser.add_argument("--classification-root", type=Path, required=True)
    parser.add_argument("--yolo-data", type=Path, required=True)
    parser.add_argument("--val-audit-csv", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--class-name-mode", choices=("raw", "mango"), default="raw")
    parser.add_argument("--per-transition", type=int, default=DEFAULT_PER_TRANSITION)
    parser.add_argument("--device", type=str, default="")
    return parser.parse_args(argv)


def transition_kind(
    *,
    target: int,
    keeper_prediction: int,
    candidate_prediction: int,
    focus_class: int = 1,
) -> str:
    if target == focus_class and keeper_prediction != focus_class and candidate_prediction == focus_class:
        return "class1_fn_rescued"
    if target == focus_class and keeper_prediction == focus_class and candidate_prediction != focus_class:
        return "class1_tp_broken"
    if target != focus_class and keeper_prediction == focus_class and candidate_prediction != focus_class:
        return "class1_fp_removed"
    if target != focus_class and keeper_prediction != focus_class and candidate_prediction == focus_class:
        return "class1_fp_created"
    if keeper_prediction != target and candidate_prediction == target:
        return "correction"
    if keeper_prediction == target and candidate_prediction != target:
        return "harm"
    return "neutral"


def _load_changed_rows(path: Path) -> Tuple[List[Dict[str, object]], Dict[str, int]]:
    with Path(path).open("r", newline="", encoding="utf-8-sig") as handle:
        reader = csv.DictReader(handle)
        rows = list(reader)
        fields = set(reader.fieldnames or [])
    required = {
        "sample_index",
        "target",
        "path",
        "base_prediction",
        "base_prob_1",
        "dinov3_selected_prediction",
        "dinov3_selected_prob_1",
    }
    missing = sorted(required.difference(fields))
    if missing:
        raise ValueError(f"Changed-case audit CSV is missing fields: {missing}")
    indexed: Dict[int, Dict[str, object]] = {}
    summary = {
        "validation_rows": int(len(rows)),
        "changed": 0,
        "corrections": 0,
        "harms": 0,
        "neutral": 0,
        **{name: 0 for name in TRANSITION_ORDER},
    }
    for raw in rows:
        sample_index = int(raw["sample_index"])
        if sample_index in indexed:
            raise ValueError(f"Duplicate sample_index={sample_index} in {path}")
        target = int(raw["target"])
        keeper = int(raw["base_prediction"])
        candidate = int(raw["dinov3_selected_prediction"])
        if keeper == candidate:
            continue
        kind = transition_kind(
            target=target,
            keeper_prediction=keeper,
            candidate_prediction=candidate,
        )
        row: Dict[str, object] = {
            "sample_index": sample_index,
            "path": str(raw["path"]),
            "target": target,
            "keeper_prediction": keeper,
            "candidate_prediction": candidate,
            "keeper_p1": float(raw["base_prob_1"]),
            "candidate_p1": float(raw["dinov3_selected_prob_1"]),
            "p1_delta": float(raw["dinov3_selected_prob_1"])
            - float(raw["base_prob_1"]),
            "transition_kind": kind,
        }
        indexed[sample_index] = row
        summary["changed"] += 1
        if keeper != target and candidate == target:
            summary["corrections"] += 1
        elif keeper == target and candidate != target:
            summary["harms"] += 1
        else:
            summary["neutral"] += 1
        if kind in TRANSITION_ORDER:
            summary[kind] += 1
    if not indexed:
        raise ValueError("No keeper-to-DINOv3 validation changes were found")
    return [indexed[index] for index in sorted(indexed)], summary


def select_review_rows(
    rows: Sequence[Mapping[str, object]],
    *,
    per_transition: int = DEFAULT_PER_TRANSITION,
) -> List[Dict[str, object]]:
    if int(per_transition) <= 0:
        raise ValueError("per_transition must be positive")
    selected: List[Dict[str, object]] = []
    for kind in TRANSITION_ORDER:
        bucket = [dict(row) for row in rows if str(row["transition_kind"]) == kind]
        bucket.sort(
            key=lambda row: (
                -abs(float(row["p1_delta"])),
                int(row["sample_index"]),
            )
        )
        selected.extend(bucket[: int(per_transition)])
    if not selected:
        raise ValueError("No class-1 transition rows are available for review")
    return selected


def _build_val_alignment(args: argparse.Namespace) -> Dict[str, object]:
    rows, classes = _imagefolder_rows(Path(args.classification_root), "val")
    return _alignment_plan(
        classification_rows=rows,
        classification_classes=classes,
        yolo_data=Path(args.yolo_data),
        split="val",
        class_name_mode=str(args.class_name_mode),
    )


def _extract_model_views(
    *,
    model_key: str,
    selected_rows: Sequence[Mapping[str, object]],
    alignment: Mapping[str, object],
    classification_root: Path,
    device: torch.device,
) -> Tuple[Dict[int, Dict[str, object]], Dict[str, object]]:
    model, transform, metadata = _build_model_and_transform(model_key, device=device)
    dataset = PathImageFolder(Path(classification_root) / "val", transform=transform)
    row_indices = np.asarray(alignment["row_indices"], dtype=np.int64)
    labels = np.asarray(alignment["labels"], dtype=np.int64)
    sample_indices = [int(row["sample_index"]) for row in selected_rows]
    images: List[torch.Tensor] = []
    for sample_index in sample_indices:
        if sample_index < 0 or sample_index >= row_indices.size:
            raise ValueError(f"sample_index outside validation alignment: {sample_index}")
        image, folder_target, _path = dataset[int(row_indices[sample_index])]
        class_name = str(dataset.classes[int(folder_target)])
        target_name = str(alignment["class_names"][int(labels[sample_index])])
        if class_name != target_name:
            raise ValueError(f"Class identity drift at sample_index={sample_index}")
        images.append(image)
    batch = torch.stack(images).to(device=device, dtype=torch.float32)
    with torch.inference_mode():
        tokens = model.forward_features(batch)
    energy = _patch_energy_map(
        tokens,
        grid_size=tuple(int(value) for value in metadata["patch_grid"]),
        prefix_tokens=int(metadata["prefix_tokens"]),
    ).cpu().numpy()
    mean = torch.tensor(metadata["mean"], dtype=torch.float32).view(1, 3, 1, 1)
    std = torch.tensor(metadata["std"], dtype=torch.float32).view(1, 3, 1, 1)
    rgb = (
        (batch.detach().float().cpu() * std + mean)
        .clamp(0.0, 1.0)
        .permute(0, 2, 3, 1)
        .numpy()
    )
    result: Dict[int, Dict[str, object]] = {}
    for position, sample_index in enumerate(sample_indices):
        rgb_uint8 = (rgb[position] * 255.0).round().astype(np.uint8)
        result[sample_index] = {
            "rgb": rgb_uint8,
            "energy": np.asarray(energy[position], dtype=np.float32),
            "overlay": _heat_overlay(rgb_uint8, energy[position]),
        }
    del model, batch, tokens
    if device.type == "cuda":
        torch.cuda.empty_cache()
    return result, metadata


def _write_csv(path: Path, rows: Sequence[Mapping[str, object]]) -> None:
    values = [dict(row) for row in rows]
    fieldnames: List[str] = []
    for row in values:
        for key in row:
            if key not in fieldnames:
                fieldnames.append(key)
    with Path(path).open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(values)


def _render_contact_sheet(
    path: Path,
    *,
    selected_rows: Sequence[Mapping[str, object]],
    views: Mapping[str, Mapping[int, Mapping[str, object]]],
    per_transition: int,
) -> None:
    tile = 116
    item_header = 34
    group_header = 42
    group_width = tile * 3
    canvas = Image.new(
        "RGB",
        (group_width * len(TRANSITION_ORDER), group_header + per_transition * (tile + item_header)),
        "white",
    )
    draw = ImageDraw.Draw(canvas)
    font = ImageFont.load_default()
    resampling = getattr(Image, "Resampling", Image)
    by_kind = {
        kind: [row for row in selected_rows if str(row["transition_kind"]) == kind]
        for kind in TRANSITION_ORDER
    }
    for group_index, kind in enumerate(TRANSITION_ORDER):
        x = group_index * group_width
        draw.text((x + 4, 4), kind, fill=(0, 0, 0), font=font)
        draw.text((x + 4, 20), "RGB | DINOv2 energy | DINOv3 energy", fill=(0, 0, 0), font=font)
        for row_index, row in enumerate(by_kind[kind]):
            sample_index = int(row["sample_index"])
            y = group_header + row_index * (tile + item_header)
            rgb = Image.fromarray(np.asarray(views["dinov3"][sample_index]["rgb"], dtype=np.uint8))
            dino2 = Image.fromarray(np.asarray(views["dinov2"][sample_index]["overlay"], dtype=np.uint8))
            dino3 = Image.fromarray(np.asarray(views["dinov3"][sample_index]["overlay"], dtype=np.uint8))
            for column, image in enumerate((rgb, dino2, dino3)):
                canvas.paste(
                    image.resize((tile, tile), resampling.BILINEAR),
                    (x + column * tile, y),
                )
            draw.text(
                (x + 3, y + tile + 2),
                (
                    f"idx={sample_index} y={row['target']} "
                    f"{row['keeper_prediction']}->{row['candidate_prediction']}"
                ),
                fill=(0, 0, 0),
                font=font,
            )
            draw.text(
                (x + 3, y + tile + 16),
                f"p1 {float(row['keeper_p1']):.3f}->{float(row['candidate_p1']):.3f}",
                fill=(0, 0, 0),
                font=font,
            )
    canvas.save(path, quality=92)


def run_review(args: argparse.Namespace) -> Dict[str, object]:
    if int(args.per_transition) <= 0:
        raise ValueError("per-transition must be positive")
    for required in (
        Path(args.classification_root) / "val",
        Path(args.yolo_data),
        Path(args.val_audit_csv),
    ):
        if not required.exists():
            raise FileNotFoundError(required)
    _assert_output_outside_datasets(
        Path(args.output_dir),
        classification_root=Path(args.classification_root),
        yolo_data=Path(args.yolo_data),
    )
    output_dir = Path(args.output_dir)
    if output_dir.exists() and any(output_dir.iterdir()):
        raise FileExistsError(f"Output directory must be empty: {output_dir}")
    output_dir.mkdir(parents=True, exist_ok=True)
    changed_rows, transition_summary = _load_changed_rows(Path(args.val_audit_csv))
    selected_rows = select_review_rows(
        changed_rows, per_transition=int(args.per_transition)
    )
    alignment = _build_val_alignment(args)
    if int(alignment["yolo_rows"]) != 2606:
        raise ValueError("Locked review requires the full 2606-row validation split")
    for row in selected_rows:
        sample_index = int(row["sample_index"])
        if int(alignment["labels"][sample_index]) != int(row["target"]):
            raise ValueError(f"Target mismatch at sample_index={sample_index}")
        if str(alignment["paths"][sample_index]).replace("/", "\\").casefold() != str(
            row["path"]
        ).replace("/", "\\").casefold():
            raise ValueError(f"Path mismatch at sample_index={sample_index}")
    device = _resolve_device(str(args.device or ""))
    views: Dict[str, Dict[int, Dict[str, object]]] = {}
    model_metadata: Dict[str, Dict[str, object]] = {}
    for model_key in ("dinov2", "dinov3"):
        views[model_key], model_metadata[model_key] = _extract_model_views(
            model_key=model_key,
            selected_rows=selected_rows,
            alignment=alignment,
            classification_root=Path(args.classification_root),
            device=device,
        )
    _write_csv(output_dir / "selected_changed_cases.csv", selected_rows)
    _render_contact_sheet(
        output_dir / "changed_case_contact_sheet.jpg",
        selected_rows=selected_rows,
        views=views,
        per_transition=int(args.per_transition),
    )
    selected_counts = {
        kind: sum(str(row["transition_kind"]) == kind for row in selected_rows)
        for kind in TRANSITION_ORDER
    }
    summary = {
        "mode": "dinov3_changed_case_review",
        "guardrail": (
            "Validation-only visual diagnostic using OOF-selected frozen readouts. "
            "No test, fit, threshold, raw-data edit, checkpoint, or trainable manifest."
        ),
        "source_val_audit_csv": str(Path(args.val_audit_csv).resolve()),
        "source_validation_rows": int(transition_summary["validation_rows"]),
        "changed_rows": int(len(changed_rows)),
        "all_transition_summary": transition_summary,
        "selection": (
            f"Top {int(args.per_transition)} absolute keeper-to-DINOv3 class1 "
            "probability changes per class1 transition type"
        ),
        "selected_rows": int(len(selected_rows)),
        "selected_counts": selected_counts,
        "alignment": {
            "classification_rows": int(alignment["classification_rows"]),
            "yolo_rows": int(alignment["yolo_rows"]),
            "source_groups": int(alignment["source_groups"]),
            "missing_rows": int(alignment["missing_rows"]),
            "label_mismatches": int(alignment["label_mismatches"]),
        },
        "model_weights": {
            key: value["weight"] for key, value in model_metadata.items()
        },
        "device": str(device),
        "descriptor_precision": "fp32",
        "test_split_used": False,
        "raw_dataset_touched": False,
        "model_or_checkpoint_written": False,
    }
    (output_dir / "summary.json").write_text(
        json.dumps(summary, indent=2), encoding="utf-8"
    )
    (output_dir / "README.md").write_text(
        "# DINOv3 Changed-Case Review\n\n"
        f"- Reviewed `{len(selected_rows)}` rows from `{len(changed_rows)}` keeper changes across `{transition_summary['validation_rows']}` validation rows.\n"
        f"- Selected counts: `{selected_counts}`.\n"
        "- Columns are RGB crop, DINOv2 patch-deviation energy, and DINOv3 patch-deviation energy.\n"
        "- No test, fit, threshold, raw-data edit, checkpoint, or trainable manifest.\n",
        encoding="utf-8",
    )
    _write_artifact_manifest(
        output_dir,
        mode="dinov3_changed_case_review_manifest",
    )
    return summary


def main(argv: Optional[Sequence[str]] = None) -> int:
    summary = run_review(_parse_args(argv))
    print(json.dumps(summary, indent=2), flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
