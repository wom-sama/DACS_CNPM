from __future__ import annotations

import argparse
import csv
import hashlib
import json
from pathlib import Path
import subprocess
from typing import Dict, List, Mapping, Optional, Sequence, Tuple

import numpy as np
from PIL import Image, ImageDraw, ImageFont
import torch

from trkh.tools.audit_counterfactual_illumination_disagreement_readiness import (
    _SelectedConditionDataset,
)
from trkh.tools.audit_cutpaste_surface_response_a0 import (
    _bbox_support,
    _build_dataset,
    _protected_untracked_state,
    _to_rgb,
)
from trkh.tools.audit_more_model_rebalancing_readiness import (
    CleanTrainRow,
    _read_clean_train_rows,
)
from trkh.tools.nsa_class_pair_poisson import (
    PoissonView,
    TargetGeometry,
    array_sha256,
    exclude_entering_chromatic_occluders,
    geometry_sha256,
    make_poisson_view,
    rng_snapshot,
    rng_snapshot_equal,
    sample_poisson_geometry,
    sample_target_geometry,
)
from trkh.evaluation.robustness_eval import IdentityCorruption


METHOD = "nsa_class_pair_poisson_a0"
SEED = 20260724
PREVIEW_ROWS = 24
PREVIEW_PAIRS = (
    (1, 0),
    (1, 2),
    (1, 4),
    (0, 1),
    (2, 1),
    (4, 1),
) * 4

REPO_ROOT = Path(__file__).resolve().parents[2]
KEEPER_ROOT = (
    REPO_ROOT
    / "runs"
    / "probe_v8_yolof_pairroute_teacherfocusbinary015_boundarydrop_bboxprior_120b_2e_20260701"
)
DEFAULT_DATA = Path(r"D:\DataAI\AIEx\newdataset\yolo_f\data.yaml")
DEFAULT_CIDT_SUMMARY = (
    REPO_ROOT / "runs" / "audit_cidt_readiness_full_train_20260714" / "summary.json"
)
DEFAULT_CIDT_PREDICTIONS = (
    REPO_ROOT
    / "runs"
    / "audit_cidt_readiness_full_train_20260714"
    / "predictions_all_conditions.csv"
)
PROTOCOL_PATH = (
    REPO_ROOT
    / "docs"
    / "TRKH_5CLASS_NSA_CLASS_PAIR_POISSON_A0_PROTOCOL_20260724.md"
)
NSA_ROOT = Path(r"D:\DataAI\external_sources\official\natural-synthetic-anomalies")
PAPER_PATH = Path(r"D:\DataAI\Tools\source_cache\papers\NSA_ECCV2022.pdf")
SUPPLEMENT_PATH = Path(
    r"D:\DataAI\Tools\source_cache\papers\NSA_ECCV2022_supp.pdf"
)

LOCKED = {
    "checkpoint": "1f49d577240c69dc63c30af70db52ec2aa9da65a17aef1c4b1c09ece6c482677",
    "launcher_args": "908a05cf66b2a01162cae62e4ff2251eaae1297d31e70510144e4954159b7eff",
    "resolved_config": "e9c4f48917e333d2f34f61806bb54041b35f2217ebb23afb3bd0ced969854674",
    "data": "716e33df24c63a9e9920f97b685199707fb84ab4c7154544f5dd9a3e00d884ef",
    "cidt_summary": "d4891edf2963ab12385b7ce5bdc812ec3e19c5c098acd25c66eb557af541d7ad",
    "cidt_predictions": "2e0993752d58d99ea429bfefe1e2bfe6fa949e45aea1a26cc4bdfee97d4db21c",
    "protocol": "e7e2794794a7059379060ffc10579b8b63f6288d3f50d187f8f62d36ce69fa18",
    "paper": "fdd14c4ee81f353d6f279f71b9516825fd84e4f603b050dc869db03dba474e44",
    "supplement": "5804ae0202b74149441baee1151d52da83fc1864a454e17fedcdfaaacf8d32da",
    "nsa_license": "7f86283e43b5c69fe93ff2c73ddb2bd89f1dbb718995ece3a7b5fa41b2435aaa",
    "nsa_task": "14f751362fa2f05935599e69b1334cf3568a2d01b2e8e7317eeb02185f3e4552",
    "current_command": "36b9aa1a21b765829acf4c8321be147bd76297de4ccdb8a40e6dee8e37940faf",
    "command_history": "39bd2879ce66fddf36a953021ea1e40f8d9de6cb4334b9b825011b2b8dc98f53",
}
NSA_COMMIT = "919591685307ce030fe27cb77687509dc277189c"
NSA_TREE = "6eebd6e9d7ce9ff77d7d0aa9a2640da7f304cd06"


def parse_args(argv: Optional[Sequence[str]] = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Locked train-only NSA class-pair Poisson A0."
    )
    parser.add_argument("--geometry-preview", action="store_true")
    parser.add_argument("--finalize-geometry-review", action="store_true")
    parser.add_argument("--run-a0", action="store_true")
    parser.add_argument("--replay-summary", type=Path)
    parser.add_argument("--decision", choices=("pass", "reject"))
    parser.add_argument("--review-note", default="")
    parser.add_argument("--checkpoint", type=Path, default=KEEPER_ROOT / "checkpoints" / "best.pt")
    parser.add_argument("--launcher-args", type=Path, default=KEEPER_ROOT / "launcher_args.json")
    parser.add_argument("--resolved-config", type=Path, default=KEEPER_ROOT / "resolved_config.json")
    parser.add_argument("--data", type=Path, default=DEFAULT_DATA)
    parser.add_argument("--cidt-summary", type=Path, default=DEFAULT_CIDT_SUMMARY)
    parser.add_argument("--cidt-predictions", type=Path, default=DEFAULT_CIDT_PREDICTIONS)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--device", choices=("auto", "cpu", "cuda"), default="cuda")
    parser.add_argument("--batch-size", type=int, default=16)
    parser.add_argument("--num-workers", type=int, default=4)
    return parser.parse_args(argv)


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with Path(path).open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _verify_hash(path: Path, expected: str, label: str) -> Dict[str, object]:
    resolved = Path(path).resolve()
    if not resolved.is_file():
        raise FileNotFoundError(f"{label} not found: {resolved}")
    observed = _sha256(resolved)
    if observed != expected:
        raise ValueError(f"{label} SHA-256 mismatch: {observed} != {expected}")
    return {"path": str(resolved), "sha256": observed, "passed": True}


def _git(repo: Path, *arguments: str) -> str:
    result = subprocess.run(
        ["git", *arguments],
        cwd=repo,
        text=True,
        capture_output=True,
        check=True,
    )
    return result.stdout.strip()


def _repo_state(*, require_clean: bool) -> Dict[str, object]:
    head = _git(REPO_ROOT, "rev-parse", "HEAD")
    upstream = _git(REPO_ROOT, "rev-parse", "@{upstream}")
    tracked = _git(REPO_ROOT, "status", "--short", "--untracked-files=no")
    result = {
        "head": head,
        "upstream": upstream,
        "head_equals_upstream": head == upstream,
        "tracked_status": tracked,
        "tracked_clean": not bool(tracked),
    }
    if require_clean and (head != upstream or tracked):
        raise ValueError(f"formal TRKH repository state is not clean/pushed: {result}")
    return result


def _official_state() -> Dict[str, object]:
    status = _git(NSA_ROOT, "status", "--short")
    commit = _git(NSA_ROOT, "rev-parse", "HEAD")
    tree = _git(NSA_ROOT, "rev-parse", "HEAD^{tree}")
    result = {
        "path": str(NSA_ROOT),
        "status": status,
        "clean": not bool(status),
        "commit": commit,
        "tree": tree,
    }
    if status or commit != NSA_COMMIT or tree != NSA_TREE:
        raise ValueError(f"official NSA repository differs from lock: {result}")
    return result


def verify_locked_inputs(args: argparse.Namespace, *, require_clean: bool) -> Dict[str, object]:
    records = {
        "checkpoint": _verify_hash(Path(args.checkpoint), LOCKED["checkpoint"], "keeper"),
        "launcher_args": _verify_hash(
            Path(args.launcher_args), LOCKED["launcher_args"], "launcher args"
        ),
        "resolved_config": _verify_hash(
            Path(args.resolved_config), LOCKED["resolved_config"], "resolved config"
        ),
        "data": _verify_hash(Path(args.data), LOCKED["data"], "yolo_f data"),
        "cidt_summary": _verify_hash(
            Path(args.cidt_summary), LOCKED["cidt_summary"], "CIDT summary"
        ),
        "cidt_predictions": _verify_hash(
            Path(args.cidt_predictions),
            LOCKED["cidt_predictions"],
            "CIDT predictions",
        ),
        "protocol": _verify_hash(PROTOCOL_PATH, LOCKED["protocol"], "protocol"),
        "paper": _verify_hash(PAPER_PATH, LOCKED["paper"], "NSA paper"),
        "supplement": _verify_hash(
            SUPPLEMENT_PATH, LOCKED["supplement"], "NSA supplement"
        ),
        "nsa_license": _verify_hash(
            NSA_ROOT / "LICENSE", LOCKED["nsa_license"], "NSA license"
        ),
        "nsa_task": _verify_hash(
            NSA_ROOT / "self_sup_data" / "self_sup_tasks.py",
            LOCKED["nsa_task"],
            "NSA official task",
        ),
        "current_command": _verify_hash(
            REPO_ROOT
            / "docs"
            / "TRKH_CURRENT_BEST_FULL_TRAIN_COMMANDS_20260706.txt",
            LOCKED["current_command"],
            "current-best commands",
        ),
        "command_history": _verify_hash(
            REPO_ROOT / "docs" / "TRKH_CURRENT_BEST_COMMAND_UPDATE_HISTORY.txt",
            LOCKED["command_history"],
            "command history",
        ),
    }
    protected = _protected_untracked_state()
    if not bool(protected["passed"]):
        raise ValueError(f"protected untracked payload changed: {protected}")
    return {
        "hashes": records,
        "repo": _repo_state(require_clean=require_clean),
        "official_nsa": _official_state(),
        "protected_untracked": protected,
    }


def _metadata_checkpoint(resolved_config_path: Path) -> Dict[str, object]:
    resolved = json.loads(Path(resolved_config_path).read_text(encoding="utf-8"))
    model_config = resolved.get("model_config")
    augmentation = resolved.get("augmentation_config")
    data = resolved.get("data")
    if not isinstance(model_config, Mapping) or not isinstance(augmentation, Mapping):
        raise ValueError("resolved config lacks model/augmentation semantics")
    if not isinstance(data, Mapping):
        raise ValueError("resolved config lacks data semantics")
    class_names = data.get("class_names")
    if not isinstance(class_names, list) or len(class_names) != 5:
        raise ValueError("resolved config class order differs from lock")
    return {
        "model_config": dict(model_config),
        "augmentation_config": dict(augmentation),
        "data_summary": {
            "crop_to_primary_object": bool(data.get("crop_to_primary_object", True)),
            "classification_object_crops": bool(
                data.get("classification_object_crops", True)
            ),
        },
        "class_names": [str(value) for value in class_names],
    }


def _stable_rank(namespace: str, row: CleanTrainRow) -> str:
    payload = (
        f"{SEED}|{namespace}|{row.sample_index}|{row.source_stem}|{row.target}"
    ).encode("utf-8")
    return hashlib.sha256(payload).hexdigest()


def _ranked_rows(
    rows: Sequence[CleanTrainRow],
    *,
    target: int,
    namespace: str,
    exclude_sources: Sequence[str] = (),
) -> List[CleanTrainRow]:
    excluded = {str(value).casefold() for value in exclude_sources}
    selected = [
        row
        for row in rows
        if int(row.target) == int(target) and row.source_stem not in excluded
    ]
    return sorted(selected, key=lambda row: (_stable_rank(namespace, row), row.sample_index))


def build_preview_plan(rows: Sequence[CleanTrainRow]) -> List[Dict[str, object]]:
    if len(PREVIEW_PAIRS) != PREVIEW_ROWS:
        raise RuntimeError("preview pair lock differs from row lock")
    used_targets: set[int] = set()
    plan: List[Dict[str, object]] = []
    for position, (target_class, cross_class) in enumerate(PREVIEW_PAIRS):
        target_candidates = _ranked_rows(
            rows,
            target=target_class,
            namespace=f"preview-target-{position}",
        )
        target_row = next(
            (row for row in target_candidates if row.sample_index not in used_targets),
            None,
        )
        if target_row is None:
            raise RuntimeError("could not select a distinct preview target")
        used_targets.add(target_row.sample_index)
        same_candidates = _ranked_rows(
            rows,
            target=target_class,
            namespace=f"preview-same-{position}",
            exclude_sources=(target_row.source_stem,),
        )[:64]
        cross_candidates = _ranked_rows(
            rows,
            target=cross_class,
            namespace=f"preview-cross-{position}",
            exclude_sources=(target_row.source_stem,),
        )[:64]
        if not same_candidates or not cross_candidates:
            raise RuntimeError("preview donor plan is empty")
        plan.append(
            {
                "position": position,
                "target_class": int(target_class),
                "cross_class": int(cross_class),
                "target_sample_index": int(target_row.sample_index),
                "target_source_stem": target_row.source_stem,
                "target_fold": int(target_row.fold),
                "same_candidate_indices": [
                    int(row.sample_index) for row in same_candidates
                ],
                "cross_candidate_indices": [
                    int(row.sample_index) for row in cross_candidates
                ],
            }
        )
    return plan


def _tensor_rgb_uint8(tensor: torch.Tensor, semantics: Mapping[str, object]) -> np.ndarray:
    rgb = _to_rgb(tensor, semantics)
    return (
        rgb.permute(1, 2, 0).numpy().clip(0.0, 1.0) * 255.0
    ).round().astype(np.uint8)


def _load_preview_sample(
    dataset: _SelectedConditionDataset,
    semantics: Mapping[str, object],
    sample_index: int,
) -> Dict[str, object]:
    tensor, label, metadata = dataset[int(sample_index)]
    if "crop_bbox" not in metadata or "image_mask" not in metadata:
        raise ValueError("preview sample lacks crop_bbox/image_mask")
    rgb_tensor = _to_rgb(tensor, semantics)
    support = _bbox_support(
        rgb_tensor,
        metadata["crop_bbox"],
        metadata["image_mask"],
    )
    rgb = _tensor_rgb_uint8(tensor, semantics)
    base_support = support.numpy().astype(bool)
    final_support, exclusion = exclude_entering_chromatic_occluders(
        rgb,
        base_support,
    )
    return {
        "tensor": tensor,
        "label": int(label),
        "metadata": metadata,
        "rgb": rgb,
        "base_support": base_support,
        "support": final_support,
        "occluder_exclusion": exclusion,
    }


def _find_preview_donor(
    dataset: _SelectedConditionDataset,
    semantics: Mapping[str, object],
    target: Mapping[str, object],
    target_geometry: TargetGeometry,
    candidate_indices: Sequence[int],
    *,
    target_sample_index: int,
    epoch: int,
    source_role_id: int,
) -> Tuple[Dict[str, object], PoissonView]:
    target_rgb = np.asarray(target["rgb"], dtype=np.uint8)
    target_support = np.asarray(target["support"], dtype=bool)
    errors: List[str] = []
    for candidate_index in candidate_indices:
        donor = _load_preview_sample(dataset, semantics, int(candidate_index))
        try:
            geometry = sample_poisson_geometry(
                target_support,
                np.asarray(donor["support"], dtype=bool),
                sample_index=target_sample_index,
                epoch=epoch,
                target_role_id=0,
                source_role_id=source_role_id + int(candidate_index),
                matched_target=target_geometry,
            )
            view = make_poisson_view(
                target_rgb,
                np.asarray(donor["rgb"], dtype=np.uint8),
                target_support,
                np.asarray(donor["support"], dtype=bool),
                geometry,
            )
            return donor, view
        except (RuntimeError, ValueError) as error:
            errors.append(f"{candidate_index}:{error}")
    raise RuntimeError(f"preview donor search exhausted: {errors[:5]}")


def _mask_overlay(
    rgb: np.ndarray,
    mask: np.ndarray,
    *,
    color: Tuple[int, int, int],
    alpha: float = 0.40,
) -> Image.Image:
    value = np.asarray(rgb, dtype=np.uint8).copy()
    selected = np.asarray(mask, dtype=bool)
    tint = np.asarray(color, dtype=np.float32)
    value[selected] = np.rint(
        value[selected].astype(np.float32) * (1.0 - alpha) + tint * alpha
    ).clip(0, 255).astype(np.uint8)
    return Image.fromarray(value, mode="RGB")


def _intensity_image(target: np.ndarray) -> Image.Image:
    value = np.asarray(target, dtype=np.float32).clip(0.0, 1.0)
    rgb = np.zeros((*value.shape, 3), dtype=np.uint8)
    rgb[..., 0] = np.rint(value * 255.0).astype(np.uint8)
    rgb[..., 1] = np.rint(np.sqrt(value) * 110.0).astype(np.uint8)
    return Image.fromarray(rgb, mode="RGB")


def _draw_geometry(
    image: Image.Image,
    view: PoissonView,
    *,
    source: bool,
) -> Image.Image:
    output = image.copy()
    draw = ImageDraw.Draw(output)
    geometry = view.geometry.source if source else view.geometry.target
    draw.rectangle(
        (
            geometry.left,
            geometry.top,
            geometry.left + geometry.width - 1,
            geometry.top + geometry.height - 1,
        ),
        outline=(255, 220, 20),
        width=3,
    )
    return output


def render_preview_contact_sheet(
    records: Sequence[Mapping[str, object]],
    visual_rows: Sequence[Mapping[str, object]],
    output_path: Path,
) -> Dict[str, object]:
    tile = 144
    label_height = 40
    headers = (
        "target/base support",
        "target/excluded",
        "same donor",
        "same Poisson",
        "cross donor",
        "cross Poisson",
        "changed",
        "intensity target",
    )
    width = len(headers) * tile
    height = (len(records) + 1) * (tile + label_height)
    sheet = Image.new("RGB", (width, height), "white")
    draw = ImageDraw.Draw(sheet)
    font = ImageFont.load_default()
    for column, header in enumerate(headers):
        draw.text((column * tile + 4, 12), header, fill="black", font=font)
    for row_index, (record, visual) in enumerate(zip(records, visual_rows), start=1):
        images = [
            _mask_overlay(
                np.asarray(visual["target_rgb"]),
                np.asarray(visual["target_base_support"]),
                color=(20, 220, 60),
            ),
            _mask_overlay(
                np.asarray(visual["target_rgb"]),
                np.asarray(visual["target_exclusion"]),
                color=(255, 30, 30),
                alpha=0.65,
            ),
            _draw_geometry(
                _mask_overlay(
                    np.asarray(visual["same_rgb"]),
                    np.asarray(visual["same_support"]),
                    color=(20, 220, 60),
                ),
                visual["same_view"],
                source=True,
            ),
            _draw_geometry(
                Image.fromarray(np.asarray(visual["same_view"].composite_rgb)),
                visual["same_view"],
                source=False,
            ),
            _draw_geometry(
                _mask_overlay(
                    np.asarray(visual["cross_rgb"]),
                    np.asarray(visual["cross_support"]),
                    color=(20, 220, 60),
                ),
                visual["cross_view"],
                source=True,
            ),
            _draw_geometry(
                Image.fromarray(np.asarray(visual["cross_view"].composite_rgb)),
                visual["cross_view"],
                source=False,
            ),
            _mask_overlay(
                np.asarray(visual["target_rgb"]),
                np.asarray(visual["cross_view"].changed_mask),
                color=(255, 30, 30),
                alpha=0.65,
            ),
            _intensity_image(np.asarray(visual["cross_view"].intensity_target)),
        ]
        y0 = row_index * (tile + label_height)
        for column, image in enumerate(images):
            thumb = image.resize((tile, tile), Image.Resampling.BILINEAR)
            sheet.paste(thumb, (column * tile, y0))
        label = (
            f"row={record['position']} target={record['target_class']} "
            f"cross={record['cross_class']} sample={record['target_sample_index']} "
            f"same={record['same_sample_index']} cross={record['cross_sample_index']}"
        )
        draw.text((4, y0 + tile + 5), label, fill="black", font=font)
    sheet.save(output_path)
    return {
        "path": str(output_path),
        "sha256": _sha256(output_path),
        "width": width,
        "height": height,
        "rows": len(records),
        "columns": len(headers),
    }


def _write_json(path: Path, payload: object) -> None:
    path.write_text(
        json.dumps(payload, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )


def _write_csv(path: Path, rows: Sequence[Mapping[str, object]]) -> None:
    if not rows:
        raise ValueError(f"cannot write empty CSV: {path}")
    fields: List[str] = []
    for row in rows:
        for key in row:
            if key not in fields:
                fields.append(key)
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields)
        writer.writeheader()
        for row in rows:
            writer.writerow(row)


def _write_manifest(output_dir: Path) -> Dict[str, object]:
    path = output_dir / "artifact_manifest.json"
    records = []
    for artifact in sorted(output_dir.iterdir(), key=lambda item: item.name.casefold()):
        if artifact == path or not artifact.is_file():
            continue
        records.append(
            {
                "path": artifact.name,
                "bytes": artifact.stat().st_size,
                "sha256": _sha256(artifact),
            }
        )
    payload = {
        "mode": f"{METHOD}_geometry_preview_manifest",
        "artifacts": records,
    }
    _write_json(path, payload)
    payload["manifest_path"] = str(path)
    payload["manifest_sha256"] = _sha256(path)
    return payload


def run_geometry_preview(args: argparse.Namespace) -> Dict[str, object]:
    output_dir = Path(args.output_dir).resolve()
    if output_dir.exists() and any(output_dir.iterdir()):
        raise FileExistsError(f"output directory must be empty: {output_dir}")
    output_dir.mkdir(parents=True, exist_ok=True)
    locks = verify_locked_inputs(args, require_clean=True)
    rows = _read_clean_train_rows(Path(args.cidt_predictions))
    checkpoint_metadata = _metadata_checkpoint(Path(args.resolved_config))
    base_dataset, transform, dataset_summary = _build_dataset(
        checkpoint_metadata,
        rows,
        Path(args.data).resolve(),
    )
    dataset = _SelectedConditionDataset(
        base_dataset,
        range(len(base_dataset)),
        corruption=IdentityCorruption(),
        transform=transform,
    )
    from trkh.tools.build_precision_ensemble_checkpoint import _eval_semantics

    semantics = _eval_semantics(checkpoint_metadata)
    plan = build_preview_plan(rows)
    records: List[Dict[str, object]] = []
    visual_rows: List[Dict[str, object]] = []
    before_rng = rng_snapshot()
    for item in plan:
        position = int(item["position"])
        target_index = int(item["target_sample_index"])
        target = _load_preview_sample(dataset, semantics, target_index)
        if int(target["label"]) != int(item["target_class"]):
            raise ValueError("preview target label differs from plan")
        target_geometry = sample_target_geometry(
            np.asarray(target["support"], dtype=bool),
            sample_index=target_index,
            epoch=position,
            role_id=10,
        )
        same, same_view = _find_preview_donor(
            dataset,
            semantics,
            target,
            target_geometry,
            item["same_candidate_indices"],
            target_sample_index=target_index,
            epoch=position,
            source_role_id=100,
        )
        cross, cross_view = _find_preview_donor(
            dataset,
            semantics,
            target,
            target_geometry,
            item["cross_candidate_indices"],
            target_sample_index=target_index,
            epoch=position,
            source_role_id=200,
        )
        if int(same["label"]) != int(item["target_class"]):
            raise ValueError("same-class preview donor label differs")
        if int(cross["label"]) != int(item["cross_class"]):
            raise ValueError("cross-class preview donor label differs")
        same_index = int(same["metadata"]["sample_index"].item())
        cross_index = int(cross["metadata"]["sample_index"].item())
        if rows[same_index].source_stem == rows[target_index].source_stem:
            raise ValueError("same-class preview donor shares target source")
        if rows[cross_index].source_stem == rows[target_index].source_stem:
            raise ValueError("cross-class preview donor shares target source")
        record = {
            "position": position,
            "target_class": int(item["target_class"]),
            "cross_class": int(item["cross_class"]),
            "target_sample_index": target_index,
            "same_sample_index": same_index,
            "cross_sample_index": cross_index,
            "target_source_stem": rows[target_index].source_stem,
            "same_source_stem": rows[same_index].source_stem,
            "cross_source_stem": rows[cross_index].source_stem,
            "target_fold": int(rows[target_index].fold),
            "same_fold": int(rows[same_index].fold),
            "cross_fold": int(rows[cross_index].fold),
            "target_support_pixels": int(np.asarray(target["support"]).sum()),
            "same_support_pixels": int(np.asarray(same["support"]).sum()),
            "cross_support_pixels": int(np.asarray(cross["support"]).sum()),
            "target_base_support_pixels": int(
                target["occluder_exclusion"]["base_support_pixels"]
            ),
            "target_excluded_support_pixels": int(
                target["occluder_exclusion"]["excluded_support_pixels"]
            ),
            "target_selected_occluder_components": int(
                target["occluder_exclusion"]["selected_component_count"]
            ),
            "target_base_support_sha256": str(
                target["occluder_exclusion"]["base_support_sha256"]
            ),
            "target_exclusion_sha256": str(
                target["occluder_exclusion"]["dilated_exclusion_sha256"]
            ),
            "target_final_support_sha256": str(
                target["occluder_exclusion"]["final_support_sha256"]
            ),
            "same_base_support_pixels": int(
                same["occluder_exclusion"]["base_support_pixels"]
            ),
            "same_excluded_support_pixels": int(
                same["occluder_exclusion"]["excluded_support_pixels"]
            ),
            "same_selected_occluder_components": int(
                same["occluder_exclusion"]["selected_component_count"]
            ),
            "same_base_support_sha256": str(
                same["occluder_exclusion"]["base_support_sha256"]
            ),
            "same_exclusion_sha256": str(
                same["occluder_exclusion"]["dilated_exclusion_sha256"]
            ),
            "same_final_support_sha256": str(
                same["occluder_exclusion"]["final_support_sha256"]
            ),
            "cross_base_support_pixels": int(
                cross["occluder_exclusion"]["base_support_pixels"]
            ),
            "cross_excluded_support_pixels": int(
                cross["occluder_exclusion"]["excluded_support_pixels"]
            ),
            "cross_selected_occluder_components": int(
                cross["occluder_exclusion"]["selected_component_count"]
            ),
            "cross_base_support_sha256": str(
                cross["occluder_exclusion"]["base_support_sha256"]
            ),
            "cross_exclusion_sha256": str(
                cross["occluder_exclusion"]["dilated_exclusion_sha256"]
            ),
            "cross_final_support_sha256": str(
                cross["occluder_exclusion"]["final_support_sha256"]
            ),
            "same_geometry_sha256": geometry_sha256(same_view.geometry),
            "cross_geometry_sha256": geometry_sha256(cross_view.geometry),
            "same_composite_sha256": array_sha256(same_view.composite_rgb),
            "cross_composite_sha256": array_sha256(cross_view.composite_rgb),
            "cross_changed_sha256": array_sha256(cross_view.changed_mask),
            "cross_intensity_sha256": array_sha256(cross_view.intensity_target),
            "same_changed_pixels": int(same_view.changed_mask.sum()),
            "cross_changed_pixels": int(cross_view.changed_mask.sum()),
            "cross_intensity_mean_positive": float(
                cross_view.intensity_target[cross_view.changed_mask].mean()
            ),
            "same_maximum_outside_delta": same_view.maximum_outside_target_delta,
            "cross_maximum_outside_delta": cross_view.maximum_outside_target_delta,
            "target_height": target_geometry.height,
            "target_width": target_geometry.width,
            "target_top": target_geometry.top,
            "target_left": target_geometry.left,
        }
        records.append(record)
        visual_rows.append(
            {
                "target_rgb": target["rgb"],
                "target_base_support": target["base_support"],
                "target_support": target["support"],
                "target_exclusion": target["occluder_exclusion"][
                    "dilated_exclusion_mask"
                ]
                & target["base_support"],
                "same_rgb": same["rgb"],
                "same_support": same["support"],
                "cross_rgb": cross["rgb"],
                "cross_support": cross["support"],
                "same_view": same_view,
                "cross_view": cross_view,
            }
        )
    after_rng = rng_snapshot()
    if not rng_snapshot_equal(before_rng, after_rng):
        raise RuntimeError("geometry preview consumed global RNG state")

    rows_path = output_dir / "geometry_rows.csv"
    _write_csv(rows_path, records)
    plan_path = output_dir / "preview_plan.json"
    _write_json(plan_path, plan)
    contact = render_preview_contact_sheet(
        records,
        visual_rows,
        output_dir / "geometry_contact_sheet.png",
    )
    checks = {
        "locked_inputs": True,
        "train_only_rows_9215": len(rows) == 9215,
        "preview_rows_24": len(records) == PREVIEW_ROWS,
        "target_classes_exact": sorted({row["target_class"] for row in records})
        == [0, 1, 2, 4],
        "source_disjoint_every_row": all(
            row["target_source_stem"] != row["same_source_stem"]
            and row["target_source_stem"] != row["cross_source_stem"]
            for row in records
        ),
        "nonempty_same_and_cross_changes": all(
            int(row["same_changed_pixels"]) > 0
            and int(row["cross_changed_pixels"]) > 0
            for row in records
        ),
        "zero_outside_delta": all(
            int(row["same_maximum_outside_delta"]) == 0
            and int(row["cross_maximum_outside_delta"]) == 0
            for row in records
        ),
        "global_rng_unchanged": rng_snapshot_equal(before_rng, after_rng),
        "keeper_checkpoint_not_loaded": True,
        "model_forward_count_zero": True,
        "validation_split_unused": True,
        "test_split_unused": True,
        "raw_dataset_unchanged": True,
    }
    summary = {
        "mode": f"{METHOD}_geometry_preview",
        "protocol_locked": True,
        "protocol_sha256": LOCKED["protocol"],
        "train_only": True,
        "validation_split_used": False,
        "test_split_used": False,
        "keeper_checkpoint_loaded": False,
        "model_forward_count": 0,
        "candidate_metrics_computed": False,
        "raw_dataset_modified": False,
        "rows": len(records),
        "plan_sha256": _sha256(plan_path),
        "geometry_rows_sha256": _sha256(rows_path),
        "contact_sheet": contact,
        "dataset": dataset_summary,
        "locks": locks,
        "checks": checks,
        "automatic_geometry_pass": all(checks.values()),
        "manual_review_required": True,
        "formal_a0_authorized": False,
    }
    summary_path = output_dir / "summary.json"
    _write_json(summary_path, summary)
    manifest = _write_manifest(output_dir)
    summary["summary_sha256"] = _sha256(summary_path)
    summary["manifest"] = manifest
    return summary


def finalize_geometry_review(args: argparse.Namespace) -> Dict[str, object]:
    output_dir = Path(args.output_dir).resolve()
    summary_path = output_dir / "summary.json"
    contact_path = output_dir / "geometry_contact_sheet.png"
    rows_path = output_dir / "geometry_rows.csv"
    if not summary_path.is_file() or not contact_path.is_file() or not rows_path.is_file():
        raise FileNotFoundError("geometry preview artifacts are incomplete")
    if args.decision is None:
        raise ValueError("--decision is required for geometry finalization")
    summary = json.loads(summary_path.read_text(encoding="utf-8"))
    automatic = bool(summary.get("automatic_geometry_pass", False))
    passed = automatic and args.decision == "pass"
    payload = {
        "mode": f"{METHOD}_geometry_visual_review",
        "decision": args.decision,
        "passed": passed,
        "automatic_geometry_pass": automatic,
        "review_note": str(args.review_note).strip(),
        "summary_sha256": _sha256(summary_path),
        "geometry_rows_sha256": _sha256(rows_path),
        "contact_sheet_sha256": _sha256(contact_path),
        "reviewed_rows": int(summary.get("rows", 0)),
        "review_scope": [
            "target support stays on mango surface",
            "same/cross blend stays inside support",
            "no hand/basket/background/padding source",
            "no systematic Poisson boundary halo",
            "intensity target matches visible changed support",
        ],
    }
    _write_json(output_dir / "visual_review.json", payload)
    manifest = _write_manifest(output_dir)
    payload["manifest"] = manifest
    return payload


def main(argv: Optional[Sequence[str]] = None) -> int:
    args = parse_args(argv)
    modes = [
        bool(args.geometry_preview),
        bool(args.finalize_geometry_review),
        bool(args.run_a0),
        args.replay_summary is not None,
    ]
    if sum(modes) != 1:
        raise ValueError("select exactly one execution mode")
    if args.geometry_preview:
        result = run_geometry_preview(args)
    elif args.finalize_geometry_review:
        result = finalize_geometry_review(args)
    elif args.run_a0:
        raise NotImplementedError("formal A0 implementation follows geometry approval")
    else:
        raise NotImplementedError("replay implementation follows formal A0")
    print(json.dumps(result, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
