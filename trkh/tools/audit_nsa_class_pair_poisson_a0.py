from __future__ import annotations

import argparse
import csv
import hashlib
import json
import os
from pathlib import Path
import subprocess
import sys
import threading
import time
from typing import Dict, List, Mapping, Optional, Sequence, Tuple

os.environ.setdefault("CUBLAS_WORKSPACE_CONFIG", ":4096:8")

import numpy as np
from PIL import Image, ImageDraw, ImageFont
import psutil
import torch

from trkh.tools.audit_counterfactual_illumination_disagreement_readiness import (
    _SelectedConditionDataset,
)
from trkh.tools.audit_cutpaste_surface_response_a0 import (
    _bbox_support,
    _build_dataset,
    _cohort_from_rows,
    _model_state_sha256,
    _protected_untracked_state,
    _to_rgb,
)
from trkh.tools.audit_more_model_rebalancing_readiness import (
    CleanTrainRow,
    _read_clean_train_rows,
)
from trkh.tools.nsa_class_pair_poisson import (
    ClassQueryMismatchAdapter,
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
    state_dict_sha256,
)
from trkh.tools.nsa_class_pair_poisson_a0_engine import (
    FOLDS,
    GEOMETRY_INELIGIBLE_SAMPLE_INDICES,
    MINIMUM_SUPPORT_EXTENT,
    READOUT_ROLES,
    ROLE_NAMES,
    apply_saved_readouts,
    assess_mechanism_gate,
    build_balanced_fit_panel,
    collect_fixed_visual_rows,
    fit_source_held_readouts,
    fixed_visual_indices,
    geometry_records_sha256,
    query_metrics,
    replay_training_geometry,
    role_metrics,
    score_clean_cohort,
    score_held_diagnostics,
    train_fold,
)
from trkh.tools.build_precision_ensemble_checkpoint import _eval_semantics
from trkh.evaluation.robustness_eval import IdentityCorruption
from trkh.inference.inference import load_model


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
    "protocol": "fbd7788c5ddba0e08a8b4cf680c427674f68bb9744198f41c6c1c7e6f146254f",
    "support_geometry_scan": "3aa2d4e6f3caa2331af413911d78d5b5771fe7f3ead58b1ab7b22fa34358d248",
    "worker_benchmark": "e437d84c9d47a4b7757fb03169e3bf99b6ae4c1b7b3b7a423bee85a97b825492",
    "paper": "fdd14c4ee81f353d6f279f71b9516825fd84e4f603b050dc869db03dba474e44",
    "supplement": "5804ae0202b74149441baee1151d52da83fc1864a454e17fedcdfaaacf8d32da",
    "nsa_license": "7f86283e43b5c69fe93ff2c73ddb2bd89f1dbb718995ece3a7b5fa41b2435aaa",
    "nsa_task": "14f751362fa2f05935599e69b1334cf3568a2d01b2e8e7317eeb02185f3e4552",
    "current_command": "36b9aa1a21b765829acf4c8321be147bd76297de4ccdb8a40e6dee8e37940faf",
    "command_history": "39bd2879ce66fddf36a953021ea1e40f8d9de6cb4334b9b825011b2b8dc98f53",
}
NSA_COMMIT = "919591685307ce030fe27cb77687509dc277189c"
NSA_TREE = "6eebd6e9d7ce9ff77d7d0aa9a2640da7f304cd06"
GEOMETRY_REVIEW_ROOT = (
    REPO_ROOT
    / "runs"
    / "audit_nsa_class_pair_poisson_a0_geometry_preview_v2_20260724"
)
LOCKED_GEOMETRY = {
    "summary.json": "83a1622ebf5db934de83fb3f68baede87a498ec536657505a60d19bd6358c2a5",
    "geometry_contact_sheet.png": "8237878cfc1f9febf9603f2297ca1e7c4bef36ec25707f98068d8125e9742ee4",
    "geometry_rows.csv": "e69e4c0453add9df44c8889e81bf46223778e6adaea2fe12619bed01eda68f30",
    "visual_review.json": "ddc49b219f423aceb9c5bc3b29398f07b3040e28a0d495477d1b7339391c5a96",
    "artifact_manifest.json": "eff84c110a8216bd94360f95f4f7d4309ac9ad044365c6a1ca94ef78dd5b00c7",
}
MAX_CUDA_BYTES = int(4.5 * 1024**3)
MAX_VIRTUAL_MEMORY_FRACTION = 0.82
FORMAL_VISUAL_ROWS = 19
LOCKED_MODEL_STATE_SHA256 = (
    "522b95f1ecfa3837e5561d3306562a0d254e40aafeb68ccae5d9ccb52a3ea38c"
)
LOCKED_GEOMETRY_PROTOCOL_SHA256 = (
    "8647a74d5a3941570ab1e91e0feb63bdbaccc2cee106612a8989d2c447380290"
)
SUPPORT_GEOMETRY_SCAN_PATH = (
    REPO_ROOT
    / "docs"
    / "TRKH_5CLASS_NSA_SUPPORT_GEOMETRY_SCAN_20260724.json"
)
WORKER_BENCHMARK_PATH = (
    REPO_ROOT
    / "docs"
    / "TRKH_5CLASS_NSA_WORKER_BENCHMARK_20260724.json"
)


def parse_args(argv: Optional[Sequence[str]] = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Locked train-only NSA class-pair Poisson A0."
    )
    parser.add_argument("--geometry-preview", action="store_true")
    parser.add_argument("--finalize-geometry-review", action="store_true")
    parser.add_argument("--finalize-formal-review", action="store_true")
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
        "support_geometry_scan": _verify_hash(
            SUPPORT_GEOMETRY_SCAN_PATH,
            LOCKED["support_geometry_scan"],
            "support geometry scan",
        ),
        "worker_benchmark": _verify_hash(
            WORKER_BENCHMARK_PATH,
            LOCKED["worker_benchmark"],
            "worker benchmark",
        ),
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


def _write_jsonl(path: Path, rows: Sequence[Mapping[str, object]]) -> None:
    if not rows:
        raise ValueError(f"cannot write empty JSONL: {path}")
    with path.open("w", encoding="utf-8", newline="\n") as handle:
        for row in rows:
            handle.write(
                json.dumps(_jsonable(dict(row)), sort_keys=True, separators=(",", ":"))
            )
            handle.write("\n")


def _write_manifest(output_dir: Path) -> Dict[str, object]:
    path = output_dir / "artifact_manifest.json"
    records = []
    for artifact in sorted(
        output_dir.rglob("*"),
        key=lambda item: item.relative_to(output_dir).as_posix().casefold(),
    ):
        if artifact == path or not artifact.is_file():
            continue
        records.append(
            {
                "path": artifact.relative_to(output_dir).as_posix(),
                "bytes": artifact.stat().st_size,
                "sha256": _sha256(artifact),
            }
        )
    payload = {
        "mode": f"{METHOD}_artifact_manifest",
        "artifacts": records,
    }
    _write_json(path, payload)
    payload["manifest_path"] = str(path)
    payload["manifest_sha256"] = _sha256(path)
    return payload


def _verify_manifest(output_dir: Path) -> Dict[str, object]:
    root = Path(output_dir).resolve()
    manifest_path = root / "artifact_manifest.json"
    payload = json.loads(manifest_path.read_text(encoding="utf-8"))
    expected = {
        str(row["path"]): (int(row["bytes"]), str(row["sha256"]))
        for row in payload["artifacts"]
    }
    observed = {}
    for path in sorted(root.rglob("*")):
        if path.is_file() and path != manifest_path:
            observed[path.relative_to(root).as_posix()] = (
                int(path.stat().st_size),
                _sha256(path),
            )
    if observed != expected:
        raise ValueError("artifact manifest payload differs from disk")
    return {
        "passed": True,
        "payload_count": len(observed),
        "manifest_sha256": _sha256(manifest_path),
    }


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


def _jsonable(value: object) -> object:
    if isinstance(value, Mapping):
        return {str(key): _jsonable(item) for key, item in value.items()}
    if isinstance(value, (list, tuple)):
        return [_jsonable(item) for item in value]
    if isinstance(value, np.ndarray):
        return value.tolist()
    if isinstance(value, np.generic):
        return value.item()
    if torch.is_tensor(value):
        return value.detach().cpu().tolist()
    if isinstance(value, Path):
        return str(value)
    return value


def _array_sha256(value: np.ndarray) -> str:
    array = np.ascontiguousarray(np.asarray(value))
    digest = hashlib.sha256()
    digest.update(str(array.dtype).encode("ascii"))
    digest.update(np.asarray(array.shape, dtype=np.int64).tobytes())
    digest.update(array.tobytes())
    return digest.hexdigest()


def _resolve_device(name: str) -> torch.device:
    if name == "auto":
        name = "cuda" if torch.cuda.is_available() else "cpu"
    if name == "cuda" and not torch.cuda.is_available():
        raise RuntimeError("CUDA was requested but is unavailable")
    return torch.device(name)


class _ResourceMonitor:
    def __init__(self, interval_seconds: float = 0.10) -> None:
        self.interval_seconds = float(interval_seconds)
        self.process = psutil.Process()
        self.peak_rss_bytes = int(self.process.memory_info().rss)
        self.peak_virtual_memory_fraction = float(
            psutil.virtual_memory().percent / 100.0
        )
        self._stop = threading.Event()
        self._thread = threading.Thread(target=self._sample, daemon=True)

    def _sample(self) -> None:
        while not self._stop.wait(self.interval_seconds):
            try:
                rss = int(self.process.memory_info().rss)
            except (psutil.AccessDenied, psutil.NoSuchProcess):
                rss = self.peak_rss_bytes
            self.peak_rss_bytes = max(self.peak_rss_bytes, rss)
            self.peak_virtual_memory_fraction = max(
                self.peak_virtual_memory_fraction,
                float(psutil.virtual_memory().percent / 100.0),
            )

    def __enter__(self) -> "_ResourceMonitor":
        self._thread.start()
        return self

    def __exit__(self, exc_type, exc_value, traceback) -> None:
        self._stop.set()
        self._thread.join()
        try:
            self.peak_rss_bytes = max(
                self.peak_rss_bytes,
                int(self.process.memory_info().rss),
            )
        except (psutil.AccessDenied, psutil.NoSuchProcess):
            pass
        self.peak_virtual_memory_fraction = max(
            self.peak_virtual_memory_fraction,
            float(psutil.virtual_memory().percent / 100.0),
        )


def _process_snapshot() -> Dict[str, object]:
    current_pid = os.getpid()
    owned = {current_pid}
    try:
        process = psutil.Process(current_pid)
        owned.update(parent.pid for parent in process.parents())
        owned.update(child.pid for child in process.children(recursive=True))
    except (psutil.AccessDenied, psutil.NoSuchProcess):
        pass
    rows = []
    for process in psutil.process_iter(
        attrs=("pid", "ppid", "name", "create_time", "cmdline")
    ):
        try:
            name = str(process.info.get("name") or "").casefold()
            if name not in {"python.exe", "pythonw.exe", "trtexec.exe"}:
                continue
            pid = int(process.info["pid"])
            rows.append(
                {
                    "pid": pid,
                    "parent_pid": int(process.info.get("ppid") or 0),
                    "name": name,
                    "create_time": float(process.info.get("create_time") or 0.0),
                    "command_line": " ".join(process.info.get("cmdline") or []),
                    "owned_by_auditor_chain": pid in owned,
                }
            )
        except (psutil.AccessDenied, psutil.NoSuchProcess):
            continue
    return {
        "current_pid": current_pid,
        "owned_pids": sorted(owned),
        "processes": rows,
        "unexpected_processes": [
            row for row in rows if not bool(row["owned_by_auditor_chain"])
        ],
    }


def _gpu_snapshot() -> Dict[str, object]:
    query = subprocess.run(
        [
            "nvidia-smi",
            "--query-gpu=name,utilization.gpu,memory.used,memory.total,"
            "temperature.gpu,power.draw,pstate",
            "--format=csv,noheader,nounits",
        ],
        check=True,
        capture_output=True,
        text=True,
    )
    rows = []
    for line in query.stdout.splitlines():
        fields = [value.strip() for value in line.split(",")]
        if len(fields) != 7:
            raise ValueError(f"cannot parse nvidia-smi row: {line}")
        rows.append(
            {
                "name": fields[0],
                "utilization_percent": int(fields[1]),
                "memory_used_mib": int(fields[2]),
                "memory_total_mib": int(fields[3]),
                "temperature_c": int(fields[4]),
                "power_w": float(fields[5]),
                "pstate": fields[6],
            }
        )
    return {"gpus": rows}


def _verify_geometry_authorization() -> Dict[str, object]:
    records = {
        name: _verify_hash(
            GEOMETRY_REVIEW_ROOT / name,
            expected,
            f"geometry review {name}",
        )
        for name, expected in LOCKED_GEOMETRY.items()
    }
    summary = json.loads(
        (GEOMETRY_REVIEW_ROOT / "summary.json").read_text(encoding="utf-8")
    )
    review = json.loads(
        (GEOMETRY_REVIEW_ROOT / "visual_review.json").read_text(encoding="utf-8")
    )
    checks = {
        "automatic_geometry_pass": bool(
            summary.get("automatic_geometry_pass", False)
        ),
        "manual_review_pass": bool(review.get("passed", False)),
        "manual_decision_pass": str(review.get("decision")) == "pass",
        "reviewed_24_rows": int(review.get("reviewed_rows", 0)) == PREVIEW_ROWS,
        "protocol_sha256_matches": str(summary.get("protocol_sha256"))
        == LOCKED_GEOMETRY_PROTOCOL_SHA256,
        "keeper_not_loaded": not bool(summary.get("keeper_checkpoint_loaded", True)),
        "model_forward_zero": int(summary.get("model_forward_count", -1)) == 0,
        "validation_unused": not bool(summary.get("validation_split_used", True)),
        "test_unused": not bool(summary.get("test_split_used", True)),
    }
    if not all(checks.values()):
        raise ValueError(f"geometry review does not authorize formal A0: {checks}")
    return {"files": records, "checks": checks, "passed": True}


def _run_checked_command(
    command: Sequence[str],
    *,
    timeout_seconds: int,
) -> Dict[str, object]:
    started = time.perf_counter()
    result = subprocess.run(
        list(command),
        cwd=REPO_ROOT,
        capture_output=True,
        text=True,
        timeout=int(timeout_seconds),
        check=False,
    )
    elapsed = time.perf_counter() - started
    stdout = result.stdout or ""
    stderr = result.stderr or ""
    record = {
        "command": list(command),
        "returncode": int(result.returncode),
        "elapsed_seconds": float(elapsed),
        "stdout_sha256": hashlib.sha256(stdout.encode("utf-8")).hexdigest(),
        "stderr_sha256": hashlib.sha256(stderr.encode("utf-8")).hexdigest(),
        "stdout_tail": stdout[-4000:],
        "stderr_tail": stderr[-4000:],
        "passed": result.returncode == 0,
    }
    if result.returncode != 0:
        raise RuntimeError(f"formal preflight failed: {record}")
    return record


def _formal_preflight() -> Dict[str, object]:
    python = str(Path(sys.executable).resolve())
    paths = [
        "trkh/tools/nsa_class_pair_poisson.py",
        "trkh/tools/nsa_class_pair_poisson_a0_engine.py",
        "trkh/tools/audit_nsa_class_pair_poisson_a0.py",
        "tests/test_nsa_class_pair_poisson.py",
    ]
    commands = [
        _run_checked_command(
            [python, "-m", "py_compile", *paths],
            timeout_seconds=120,
        ),
        _run_checked_command(
            [python, "-m", "pyflakes", *paths],
            timeout_seconds=120,
        ),
        _run_checked_command(
            [python, "-m", "pytest", "tests/test_nsa_class_pair_poisson.py", "-q"],
            timeout_seconds=300,
        ),
        _run_checked_command(
            [python, "-m", "pytest", "-q"],
            timeout_seconds=1800,
        ),
        _run_checked_command(
            [
                "powershell.exe",
                "-NoProfile",
                "-Command",
                (
                    "$t=$null;$e=$null;"
                    "[void][System.Management.Automation.Language.Parser]::"
                    "ParseFile('scripts\\run_trkh_nsa_class_pair_poisson_a0.ps1',"
                    "[ref]$t,[ref]$e);"
                    "if($e.Count){$e|Out-String|Write-Error;exit 1}"
                ),
            ],
            timeout_seconds=120,
        ),
    ]
    return {
        "commands": commands,
        "passed": all(bool(record["passed"]) for record in commands),
    }


def _save_fold_scores(
    path: Path,
    scores: Mapping[str, object],
) -> Dict[str, object]:
    arrays: Dict[str, np.ndarray] = {}
    metadata = scores["metadata"]
    for key, value in metadata.items():
        arrays[f"metadata__{key}"] = np.asarray(value)
    for role, summaries in scores["roles"].items():
        for key, value in summaries.items():
            arrays[f"role__{role}__{key}"] = np.asarray(value)
    arrays["query_compatibility"] = np.asarray(scores["query_compatibility"])
    arrays["candidate_maps"] = np.asarray(scores["candidate_maps"])
    np.savez_compressed(path, **arrays)
    return {
        "path": str(path),
        "sha256": _sha256(path),
        "arrays": {
            key: {
                "shape": list(value.shape),
                "dtype": str(value.dtype),
                "sha256": _array_sha256(value),
            }
            for key, value in arrays.items()
        },
    }


def _load_fold_scores(path: Path) -> Dict[str, object]:
    with np.load(path, allow_pickle=False) as payload:
        arrays = {key: np.asarray(payload[key]).copy() for key in payload.files}
    metadata = {
        key.removeprefix("metadata__"): value
        for key, value in arrays.items()
        if key.startswith("metadata__")
    }
    roles: Dict[str, Dict[str, np.ndarray]] = {
        role: {} for role in ROLE_NAMES
    }
    for key, value in arrays.items():
        if not key.startswith("role__"):
            continue
        _, role, summary = key.split("__", 2)
        roles[role][summary] = value
    return {
        "metadata": metadata,
        "roles": roles,
        "query_compatibility": arrays["query_compatibility"],
        "candidate_maps": arrays["candidate_maps"],
    }


def _compare_fold_scores(
    expected: Mapping[str, object],
    observed: Mapping[str, object],
) -> Dict[str, object]:
    comparisons = {}
    maximum = 0.0
    exact_keys = {"sample_index", "target", "fold"}
    for key, expected_value in expected["metadata"].items():
        observed_value = np.asarray(observed["metadata"][key])
        expected_array = np.asarray(expected_value)
        if key in exact_keys:
            passed = np.array_equal(expected_array, observed_value)
            error = 0.0 if passed else float("inf")
        else:
            error = float(
                np.max(np.abs(expected_array.astype(np.float64) - observed_value))
            )
            passed = error <= 1e-6
        comparisons[f"metadata/{key}"] = {
            "maximum_absolute_error": error,
            "passed": passed,
        }
        maximum = max(maximum, error)
    for role in ROLE_NAMES:
        for key, expected_value in expected["roles"][role].items():
            observed_value = np.asarray(observed["roles"][role][key])
            error = float(
                np.max(
                    np.abs(
                        np.asarray(expected_value, dtype=np.float64)
                        - observed_value.astype(np.float64)
                    )
                )
            )
            comparisons[f"{role}/{key}"] = {
                "maximum_absolute_error": error,
                "passed": error <= 1e-6,
            }
            maximum = max(maximum, error)
    for key in ("query_compatibility", "candidate_maps"):
        error = float(
            np.max(
                np.abs(
                    np.asarray(expected[key], dtype=np.float64)
                    - np.asarray(observed[key], dtype=np.float64)
                )
            )
        )
        comparisons[key] = {
            "maximum_absolute_error": error,
            "passed": error <= 1e-6,
        }
        maximum = max(maximum, error)
    return {
        "comparisons": comparisons,
        "maximum_absolute_error": maximum,
        "passed": all(bool(item["passed"]) for item in comparisons.values()),
    }


def _flatten_visual_payload(
    rows: Sequence[Mapping[str, object]],
) -> Tuple[Dict[str, np.ndarray], List[Dict[str, object]]]:
    arrays: Dict[str, np.ndarray] = {}
    metadata: List[Dict[str, object]] = []
    for position, row in enumerate(rows):
        prefix = f"row_{position:02d}"
        for key in ("clean_rgb", "support", "same_rgb", "cross_rgb", "cross_target"):
            arrays[f"{prefix}__{key}"] = np.asarray(row[key])
        for role, value in row["role_maps"].items():
            arrays[f"{prefix}__role__{role}"] = np.asarray(value)
        for query, value in row["query_maps"].items():
            arrays[f"{prefix}__query__{query}"] = np.asarray(value)
        metadata.append(
            {
                "position": position,
                "sample_index": int(row["sample_index"]),
                "target": int(row["target"]),
                "fold": int(row["fold"]),
                "same_record": dict(row["same_record"]),
                "cross_record": dict(row["cross_record"]),
            }
        )
    return arrays, metadata


def _save_visual_payload(
    output_dir: Path,
    rows: Sequence[Mapping[str, object]],
) -> Dict[str, object]:
    arrays, metadata = _flatten_visual_payload(rows)
    arrays_path = output_dir / "fixed_visual_arrays.npz"
    metadata_path = output_dir / "fixed_visual_metadata.json"
    np.savez_compressed(arrays_path, **arrays)
    _write_json(metadata_path, metadata)
    return {
        "rows": len(rows),
        "arrays_path": str(arrays_path),
        "arrays_sha256": _sha256(arrays_path),
        "metadata_path": str(metadata_path),
        "metadata_sha256": _sha256(metadata_path),
        "array_hashes": {
            key: _array_sha256(value) for key, value in arrays.items()
        },
    }


def _compare_visual_payload(
    *,
    output_dir: Path,
    rows: Sequence[Mapping[str, object]],
) -> Dict[str, object]:
    observed_arrays, observed_metadata = _flatten_visual_payload(rows)
    with np.load(output_dir / "fixed_visual_arrays.npz", allow_pickle=False) as payload:
        expected_arrays = {
            key: np.asarray(payload[key]).copy() for key in payload.files
        }
    expected_metadata = json.loads(
        (output_dir / "fixed_visual_metadata.json").read_text(encoding="utf-8")
    )
    if set(expected_arrays) != set(observed_arrays):
        raise ValueError("fixed visual replay array keys differ")
    errors = {}
    for key in sorted(expected_arrays):
        expected = expected_arrays[key]
        observed = observed_arrays[key]
        if expected.dtype == np.bool_ or np.issubdtype(expected.dtype, np.integer):
            passed = np.array_equal(expected, observed)
            error = 0.0 if passed else float("inf")
        else:
            error = float(
                np.max(
                    np.abs(
                        expected.astype(np.float64)
                        - observed.astype(np.float64)
                    )
                )
            )
            passed = error <= 1e-6
        errors[key] = error
        if not passed:
            errors[key] = error
    metadata_match = expected_metadata == observed_metadata
    maximum = max(errors.values(), default=0.0)
    return {
        "metadata_exact": metadata_match,
        "maximum_absolute_error": maximum,
        "array_count": len(errors),
        "passed": metadata_match and maximum <= 1e-6,
    }


def _resize_rgb(value: np.ndarray, tile: int) -> Image.Image:
    image = Image.fromarray(np.asarray(value, dtype=np.uint8))
    return image.resize((tile, tile), resample=Image.Resampling.BILINEAR)


def _map_tile(value: np.ndarray, tile: int) -> Image.Image:
    array = np.asarray(value, dtype=np.float32)
    array = np.clip(array, 0.0, 1.0)
    red = np.rint(255.0 * array).astype(np.uint8)
    green = np.rint(255.0 * np.sqrt(array)).astype(np.uint8)
    blue = np.rint(255.0 * (1.0 - array)).astype(np.uint8)
    rgb = np.stack((red, green, blue), axis=2)
    return Image.fromarray(rgb).resize(
        (tile, tile), resample=Image.Resampling.NEAREST
    )


def _support_tile(
    rgb: np.ndarray,
    support: np.ndarray,
    tile: int,
) -> Image.Image:
    value = np.asarray(rgb, dtype=np.uint8).copy()
    mask = np.asarray(support, dtype=bool)
    value[~mask] = np.rint(value[~mask] * 0.20).astype(np.uint8)
    green = value[:, :, 1]
    green[mask] = np.maximum(green[mask], 210)
    return _resize_rgb(value, tile)


def render_formal_contact_sheet(
    rows: Sequence[Mapping[str, object]],
    output_path: Path,
) -> Dict[str, object]:
    if len(rows) != FORMAL_VISUAL_ROWS:
        raise ValueError("formal visual payload differs from 19-row lock")
    tile = 92
    header = 28
    label = 24
    columns = (
        "clean",
        "support",
        "same",
        "cross",
        "target",
        "candidate",
        "clean_ctl",
        "no_query",
        "permuted",
        "q0",
        "q1",
        "q2",
        "q3",
        "q4",
    )
    width = tile * len(columns)
    row_height = tile + label
    height = header + row_height * len(rows)
    sheet = Image.new("RGB", (width, height), "white")
    draw = ImageDraw.Draw(sheet)
    font = ImageFont.load_default()
    for column, name in enumerate(columns):
        draw.text((column * tile + 3, 8), name, fill="black", font=font)
    for position, row in enumerate(rows):
        y = header + position * row_height
        images = [
            _resize_rgb(row["clean_rgb"], tile),
            _support_tile(row["clean_rgb"], row["support"], tile),
            _resize_rgb(row["same_rgb"], tile),
            _resize_rgb(row["cross_rgb"], tile),
            _map_tile(row["cross_target"], tile),
            _map_tile(row["role_maps"]["poisson_query_candidate"], tile),
            _map_tile(row["role_maps"]["clean_query_control"], tile),
            _map_tile(row["role_maps"]["no_query_control"], tile),
            _map_tile(row["role_maps"]["permuted_query_control"], tile),
            *[
                _map_tile(row["query_maps"][str(query)], tile)
                for query in range(5)
            ],
        ]
        for column, image in enumerate(images):
            sheet.paste(image, (column * tile, y))
        draw.text(
            (3, y + tile + 5),
            (
                f"fold={row['fold']} target={row['target']} "
                f"sample={row['sample_index']}"
            ),
            fill="black",
            font=font,
        )
    sheet.save(output_path)
    return {
        "path": str(output_path),
        "sha256": _sha256(output_path),
        "width": width,
        "height": height,
        "rows": len(rows),
        "columns": len(columns),
    }


def _aggregate_diagnostics(
    folds: Sequence[Mapping[str, object]],
) -> Dict[str, object]:
    cross_inside = sum(float(row["cross_inside_mass"]) for row in folds)
    cross_total = sum(float(row["cross_total_mass"]) for row in folds)
    same_positive = sum(int(row["same_positive_pixels"]) for row in folds)
    same_support = sum(int(row["same_support_pixels"]) for row in folds)
    return {
        "rows": sum(int(row["rows"]) for row in folds),
        "cross_map_mass_inside_intensity": float(
            cross_inside / max(cross_total, 1e-12)
        ),
        "same_support_pixel_mismatch_rate": float(
            same_positive / max(same_support, 1)
        ),
        "cross_inside_mass": cross_inside,
        "cross_total_mass": cross_total,
        "same_positive_pixels": same_positive,
        "same_support_pixels": same_support,
    }


def _readout_metrics(
    *,
    cohort: Sequence[CleanTrainRow],
    readouts: Mapping[str, object],
) -> Dict[str, object]:
    return {
        role: role_metrics(
            cohort=cohort,
            labels=np.asarray(readouts["labels"]),
            folds=np.asarray(readouts["folds"]),
            scores=np.asarray(readouts["scores"][role]),
            actions=np.asarray(readouts["actions"][role]),
        )
        for role in READOUT_ROLES
    }


def _write_oof_rows(
    path: Path,
    *,
    cohort: Sequence[CleanTrainRow],
    readouts: Mapping[str, object],
) -> None:
    rows = []
    for position, row in enumerate(cohort):
        record = {
            "position": position,
            "sample_index": int(row.sample_index),
            "source_stem": row.source_stem,
            "fold": int(row.fold),
            "target": int(row.target),
            "positive": int(row.target == 1),
        }
        for role in READOUT_ROLES:
            record[f"{role}_score"] = float(readouts["scores"][role][position])
            record[f"{role}_action"] = int(
                bool(readouts["actions"][role][position])
            )
        rows.append(record)
    _write_csv(path, rows)


def run_formal_a0(args: argparse.Namespace) -> Dict[str, object]:
    output_dir = Path(args.output_dir).resolve()
    if output_dir.exists() and any(output_dir.iterdir()):
        raise FileExistsError(f"output directory must be empty: {output_dir}")
    output_dir.mkdir(parents=True, exist_ok=True)
    started = time.perf_counter()
    locks = verify_locked_inputs(args, require_clean=True)
    geometry_review = _verify_geometry_authorization()
    preflight = _formal_preflight()
    _write_json(output_dir / "preflight.json", preflight)
    process_before = _process_snapshot()
    gpu_before = _gpu_snapshot()
    if process_before["unexpected_processes"]:
        raise RuntimeError(
            "formal A0 found an unrelated Python/TensorRT process: "
            f"{process_before['unexpected_processes']}"
        )

    device = _resolve_device(str(args.device))
    torch.manual_seed(SEED)
    if device.type == "cuda":
        torch.cuda.manual_seed_all(SEED)
        torch.cuda.empty_cache()
        torch.cuda.reset_peak_memory_stats(device)
    torch.backends.cudnn.benchmark = False
    torch.backends.cudnn.deterministic = True
    torch.use_deterministic_algorithms(True)

    rows = _read_clean_train_rows(Path(args.cidt_predictions))
    cohort = _cohort_from_rows(rows)
    fixed_indices = fixed_visual_indices(cohort)
    support_geometry_scan = json.loads(
        SUPPORT_GEOMETRY_SCAN_PATH.read_text(encoding="utf-8")
    )
    worker_benchmark = json.loads(
        WORKER_BENCHMARK_PATH.read_text(encoding="utf-8")
    )
    if int(args.num_workers) != int(worker_benchmark["selected_workers"]):
        raise ValueError("formal worker count differs from locked benchmark")
    fold_scores: Dict[int, Dict[str, object]] = {}
    fold_train_summaries: List[Dict[str, object]] = []
    fold_diagnostics: List[Dict[str, object]] = []
    visual_rows: List[Dict[str, object]] = []
    fold_artifacts: List[Dict[str, object]] = []

    with _ResourceMonitor() as resources:
        model, checkpoint, class_names = load_model(
            Path(args.checkpoint).resolve(),
            device=device,
        )
        if len(class_names) != 5:
            raise ValueError("keeper class count differs from five-class lock")
        base_dataset, transform, dataset_summary = _build_dataset(
            checkpoint,
            rows,
            Path(args.data).resolve(),
        )
        transformed_dataset = _SelectedConditionDataset(
            base_dataset,
            range(len(base_dataset)),
            corruption=IdentityCorruption(),
            transform=transform,
        )
        semantics = _eval_semantics(checkpoint)
        keeper_state_before = _model_state_sha256(model)
        if keeper_state_before != LOCKED_MODEL_STATE_SHA256:
            raise ValueError(
                "loaded keeper state hash differs from locked checkpoint hash"
            )

        for held_fold in FOLDS:
            print(
                json.dumps(
                    {
                        "stage": "nsa_formal_fold_start",
                        "held_fold": held_fold,
                        "elapsed_seconds": time.perf_counter() - started,
                    }
                ),
                flush=True,
            )
            fold_dir = output_dir / f"fold_{held_fold}"
            fold_dir.mkdir(parents=True, exist_ok=False)
            panel, panel_summary = build_balanced_fit_panel(
                rows, held_fold=held_fold
            )
            _write_json(
                fold_dir / "fit_panel_indices.json",
                {
                    "held_fold": held_fold,
                    "indices": panel,
                    "summary": panel_summary,
                },
            )
            roles, training = train_fold(
                model=model,
                transformed_dataset=transformed_dataset,
                semantics=semantics,
                rows=rows,
                held_fold=held_fold,
                device=device,
                batch_size=int(args.batch_size),
                num_workers=int(args.num_workers),
                output_dir=output_dir,
            )
            geometry_records = training.pop("geometry_records")
            losses = training.pop("losses")
            geometry_hash = geometry_records_sha256(geometry_records)
            _write_jsonl(fold_dir / "geometry_records.jsonl", geometry_records)
            _write_csv(fold_dir / "losses.csv", losses)
            training["geometry_records_sha256"] = geometry_hash
            training["images_per_second"] = float(
                4
                * int(training["panel"]["selected_rows"])
                * 2
                / max(float(training["elapsed_seconds"]), 1e-12)
            )
            _write_json(fold_dir / "training_summary.json", training)

            scored = score_clean_cohort(
                model=model,
                roles=roles,
                transformed_dataset=transformed_dataset,
                semantics=semantics,
                rows=rows,
                cohort_indices=[row.sample_index for row in cohort],
                device=device,
                batch_size=int(args.batch_size),
                num_workers=int(args.num_workers),
            )
            fold_scores[held_fold] = scored
            scores_artifact = _save_fold_scores(
                fold_dir / "clean_cohort_scores.npz",
                scored,
            )
            diagnostic = score_held_diagnostics(
                model=model,
                candidate=roles["poisson_query_candidate"],
                transformed_dataset=transformed_dataset,
                semantics=semantics,
                rows=rows,
                held_fold=held_fold,
                device=device,
                batch_size=int(args.batch_size),
                num_workers=int(args.num_workers),
            )
            diagnostic_records = diagnostic.pop("records")
            _write_jsonl(
                fold_dir / "held_diagnostic_records.jsonl",
                diagnostic_records,
            )
            diagnostic["records_sha256"] = geometry_records_sha256(
                diagnostic_records
            )
            _write_json(fold_dir / "held_diagnostic_summary.json", diagnostic)
            fold_diagnostics.append(diagnostic)

            fold_visuals = collect_fixed_visual_rows(
                model=model,
                roles=roles,
                transformed_dataset=transformed_dataset,
                semantics=semantics,
                rows=rows,
                target_indices=fixed_indices,
                held_fold=held_fold,
                device=device,
            )
            visual_rows.extend(fold_visuals)
            fold_train_summaries.append(training)
            fold_artifacts.append(
                {
                    "held_fold": held_fold,
                    "scores": scores_artifact,
                    "training_summary_sha256": _sha256(
                        fold_dir / "training_summary.json"
                    ),
                    "geometry_records_sha256": geometry_hash,
                    "losses_sha256": _sha256(fold_dir / "losses.csv"),
                    "diagnostic_summary_sha256": _sha256(
                        fold_dir / "held_diagnostic_summary.json"
                    ),
                }
            )
            del roles
            if device.type == "cuda":
                torch.cuda.empty_cache()
            print(
                json.dumps(
                    {
                        "stage": "nsa_formal_fold_complete",
                        "held_fold": held_fold,
                        "elapsed_seconds": time.perf_counter() - started,
                        "geometry_records_sha256": geometry_hash,
                    }
                ),
                flush=True,
            )

        readouts = fit_source_held_readouts(
            cohort=cohort,
            fold_scores=fold_scores,
        )
        replayed_readouts = apply_saved_readouts(
            cohort=cohort,
            fold_scores=fold_scores,
            states=readouts["states"],
        )
        immediate_readout_errors = {
            role: float(
                np.max(
                    np.abs(
                        np.asarray(readouts["scores"][role], dtype=np.float64)
                        - np.asarray(
                            replayed_readouts["scores"][role],
                            dtype=np.float64,
                        )
                    )
                )
            )
            for role in READOUT_ROLES
        }
        immediate_actions_exact = all(
            np.array_equal(
                np.asarray(readouts["actions"][role]),
                np.asarray(replayed_readouts["actions"][role]),
            )
            for role in READOUT_ROLES
        )
        metrics = _readout_metrics(cohort=cohort, readouts=readouts)
        query = query_metrics(cohort=cohort, fold_scores=fold_scores)
        diagnostics = _aggregate_diagnostics(fold_diagnostics)
        candidate_gradients_nonzero = all(
            bool(summary["all_candidate_query_gradients_nonzero"])
            for summary in fold_train_summaries
        )
        candidate_updates_nonzero = all(
            bool(summary["all_candidate_query_updates_nonzero"])
            for summary in fold_train_summaries
        )
        mechanism = assess_mechanism_gate(
            metrics=metrics,
            query=query,
            diagnostics=diagnostics,
            candidate_query_gradients_nonzero=candidate_gradients_nonzero,
            candidate_query_updates_nonzero=candidate_updates_nonzero,
        )
        keeper_state_after = _model_state_sha256(model)

    process_after = _process_snapshot()
    gpu_after = _gpu_snapshot()
    peak_cuda_bytes = (
        int(torch.cuda.max_memory_allocated(device))
        if device.type == "cuda"
        else 0
    )
    readout_states = _jsonable(readouts["states"])
    _write_json(output_dir / "readout_states.json", readout_states)
    _write_oof_rows(
        output_dir / "oof_row_scores.csv",
        cohort=cohort,
        readouts=readouts,
    )
    visual_rows.sort(
        key=lambda row: fixed_indices.index(int(row["sample_index"]))
    )
    visual_artifact = _save_visual_payload(output_dir, visual_rows)
    contact_sheet = render_formal_contact_sheet(
        visual_rows,
        output_dir / "fixed_visual_contact_sheet.png",
    )

    readout_converged = all(
        max(int(value) for value in state["iterations"]) < 4000
        for state in readouts["states"]
    )
    all_role_values_finite = all(
        np.isfinite(np.asarray(readouts["scores"][role])).all()
        for role in READOUT_ROLES
    )
    worker_records = []
    for training in fold_train_summaries:
        worker_records.extend(training["loader"])
    worker_records.extend(
        diagnostic["loader"] for diagnostic in fold_diagnostics
    )
    worker_records.extend(
        fold_scores[fold]["loader"] for fold in FOLDS
    )
    structural_checks = {
        "locked_inputs_and_clean_pushed_repo": bool(
            locks["repo"]["head_equals_upstream"]
            and locks["repo"]["tracked_clean"]
        ),
        "geometry_preview_v2_authorized": bool(geometry_review["passed"]),
        "preflight_all_passed": bool(preflight["passed"]),
        "train_rows_exact_9215": len(rows) == 9215,
        "cohort_rows_exact_750": len(cohort) == 750,
        "fixed_visual_rows_exact_19": len(visual_rows) == FORMAL_VISUAL_ROWS,
        "five_folds_complete": len(fold_train_summaries) == len(FOLDS),
        "every_step_count_exact": all(
            int(summary["steps"]) == int(summary["expected_steps"])
            for summary in fold_train_summaries
        ),
        "zero_source_disjoint_violations": all(
            int(summary["source_disjoint_violations"]) == 0
            for summary in fold_train_summaries
        ),
        "zero_held_fold_geometry_violations": all(
            int(summary["held_fold_geometry_violations"]) == 0
            for summary in fold_train_summaries
        ),
        "all_cross_class_coverage_exact": all(
            bool(summary["cross_class_coverage_exact"])
            for summary in fold_train_summaries
        ),
        "support_geometry_scan_exact": (
            int(support_geometry_scan["rows"]) == 9215
            and int(support_geometry_scan["locked_minimum_extent"])
            == MINIMUM_SUPPORT_EXTENT
            and [
                int(row["sample_index"])
                for row in support_geometry_scan[
                    "excluded_from_synthetic_only"
                ]
            ]
            == list(GEOMETRY_INELIGIBLE_SAMPLE_INDICES)
        ),
        "geometry_ineligible_never_selected": all(
            int(summary["panel"]["geometry_ineligible_selected"]) == 0
            for summary in fold_train_summaries
        ),
        "worker_benchmark_selected_four": (
            int(worker_benchmark["selected_workers"]) == 4
            and int(args.num_workers) == 4
        ),
        "all_role_updates_nonzero": all(
            all(
                float(value) > 0.0
                for value in summary["parameter_update_l2_norms"].values()
            )
            for summary in fold_train_summaries
        ),
        "all_query_gradients_nonzero": candidate_gradients_nonzero,
        "all_query_updates_nonzero": candidate_updates_nonzero,
        "all_readouts_converged_once": readout_converged,
        "all_role_values_finite": all_role_values_finite,
        "immediate_saved_readout_replay_within_1e_10": max(
            immediate_readout_errors.values()
        )
        <= 1e-10,
        "immediate_saved_actions_exact": immediate_actions_exact,
        "diagnostic_rows_exact_375": int(diagnostics["rows"]) == 375,
        "keeper_state_bit_identical": keeper_state_before
        == keeper_state_after
        == LOCKED_MODEL_STATE_SHA256,
        "requested_effective_workers_recorded": all(
            "requested_num_workers" in record
            and "effective_num_workers" in record
            for record in worker_records
        ),
        "no_unexpected_compute_process_before": not bool(
            process_before["unexpected_processes"]
        ),
        "no_unexpected_compute_process_after": not bool(
            process_after["unexpected_processes"]
        ),
        "peak_cuda_allocation_le_4p5_gib": peak_cuda_bytes <= MAX_CUDA_BYTES,
        "peak_virtual_memory_fraction_lt_0p82": float(
            resources.peak_virtual_memory_fraction
        )
        < MAX_VIRTUAL_MEMORY_FRACTION,
        "validation_split_unused": True,
        "test_split_unused": True,
        "raw_dataset_unchanged": True,
    }
    structural_pass = all(structural_checks.values())
    automatic_pass = structural_pass and bool(mechanism["mechanism_pass"])
    summary = {
        "mode": f"{METHOD}_formal",
        "status": "automatic_pass_pending_replay_and_visual"
        if automatic_pass
        else "rejected",
        "protocol_sha256": LOCKED["protocol"],
        "seed": SEED,
        "train_only": True,
        "validation_split_used": False,
        "test_split_used": False,
        "raw_dataset_modified": False,
        "locks": locks,
        "geometry_review": geometry_review,
        "preflight": preflight,
        "dataset": dataset_summary,
        "support_geometry_scan": support_geometry_scan,
        "worker_benchmark": worker_benchmark,
        "class_names": class_names,
        "cohort": {
            "rows": len(cohort),
            "positive": sum(row.target == 1 for row in cohort),
            "restricted_false_positive": sum(row.target != 1 for row in cohort),
            "fixed_visual_indices": fixed_indices,
        },
        "folds": fold_train_summaries,
        "fold_artifacts": fold_artifacts,
        "diagnostics_by_fold": fold_diagnostics,
        "diagnostics": diagnostics,
        "readout_states_sha256": _sha256(output_dir / "readout_states.json"),
        "oof_rows_sha256": _sha256(output_dir / "oof_row_scores.csv"),
        "immediate_readout_replay": {
            "score_maximum_absolute_error": immediate_readout_errors,
            "actions_exact": immediate_actions_exact,
        },
        "metrics": metrics,
        "query_metrics": query,
        "mechanism_gate": mechanism,
        "visual_artifact": visual_artifact,
        "contact_sheet": contact_sheet,
        "structural_checks": structural_checks,
        "structural_pass": structural_pass,
        "automatic_pass": automatic_pass,
        "manual_visual_review_required": True,
        "fresh_process_replay_required": True,
        "runtime": {
            "elapsed_seconds": float(time.perf_counter() - started),
            "peak_rss_bytes": int(resources.peak_rss_bytes),
            "peak_rss_gib": float(resources.peak_rss_bytes / 1024**3),
            "peak_virtual_memory_fraction": float(
                resources.peak_virtual_memory_fraction
            ),
            "peak_cuda_bytes": peak_cuda_bytes,
            "peak_cuda_gib": float(peak_cuda_bytes / 1024**3),
            "worker_records": worker_records,
            "process_before": process_before,
            "process_after": process_after,
            "gpu_before": gpu_before,
            "gpu_after": gpu_after,
            "keeper_state_before_sha256": keeper_state_before,
            "keeper_state_after_sha256": keeper_state_after,
        },
        "authorization": {
            "default_off_production_integration": False,
            "validation_smoke": False,
            "test": False,
            "probe": False,
            "full_train": False,
            "current_best_update": False,
        },
    }
    summary_path = output_dir / "summary.json"
    _write_json(summary_path, _jsonable(summary))
    manifest = _write_manifest(output_dir)
    verified_manifest = _verify_manifest(output_dir)
    return {
        "status": summary["status"],
        "automatic_pass": automatic_pass,
        "structural_pass": structural_pass,
        "mechanism_pass": bool(mechanism["mechanism_pass"]),
        "summary_path": str(summary_path),
        "summary_sha256": _sha256(summary_path),
        "manifest_sha256": verified_manifest["manifest_sha256"],
        "manifest_payload_count": len(manifest["artifacts"]),
    }


def _load_adapter_roles(
    *,
    output_dir: Path,
    held_fold: int,
    device: torch.device,
    expected_hashes: Mapping[str, str],
) -> Tuple[Dict[str, ClassQueryMismatchAdapter], Dict[str, object]]:
    roles = {}
    records = {}
    for role in ROLE_NAMES:
        path = output_dir / f"fold_{held_fold}" / f"{role}.pt"
        payload = torch.load(path, map_location="cpu", weights_only=False)
        if int(payload.get("held_fold", -1)) != int(held_fold):
            raise ValueError(f"adapter held fold differs: {path}")
        if str(payload.get("role")) != role:
            raise ValueError(f"adapter role differs: {path}")
        module = ClassQueryMismatchAdapter()
        module.load_state_dict(payload["state_dict"], strict=True)
        module.to(device=device, dtype=torch.float32)
        module.eval()
        observed = state_dict_sha256(module)
        expected = str(expected_hashes[role])
        if observed != expected:
            raise ValueError(
                f"adapter state differs for fold={held_fold} role={role}"
            )
        roles[role] = module
        records[role] = {
            "path": str(path),
            "file_sha256": _sha256(path),
            "state_sha256": observed,
            "passed": True,
        }
    return roles, records


def _compare_nested(
    expected: object,
    observed: object,
    *,
    path: str = "root",
) -> Dict[str, object]:
    maximum = 0.0
    mismatches: List[str] = []

    def visit(left: object, right: object, location: str) -> None:
        nonlocal maximum
        if isinstance(left, Mapping):
            if not isinstance(right, Mapping) or set(left) != set(right):
                mismatches.append(f"{location}:mapping_keys")
                return
            for key in left:
                visit(left[key], right[key], f"{location}/{key}")
            return
        if isinstance(left, (list, tuple)):
            if not isinstance(right, (list, tuple)) or len(left) != len(right):
                mismatches.append(f"{location}:sequence_shape")
                return
            for index, (left_item, right_item) in enumerate(zip(left, right)):
                visit(left_item, right_item, f"{location}/{index}")
            return
        if isinstance(left, bool) or isinstance(right, bool):
            if bool(left) != bool(right):
                mismatches.append(f"{location}:bool")
            return
        if isinstance(left, (int, float, np.number)) and isinstance(
            right, (int, float, np.number)
        ):
            left_value = float(left)
            right_value = float(right)
            if np.isnan(left_value) and np.isnan(right_value):
                return
            error = abs(left_value - right_value)
            if not np.isfinite(error):
                mismatches.append(f"{location}:nonfinite")
                return
            maximum = max(maximum, error)
            return
        if left != right:
            mismatches.append(f"{location}:value")

    visit(expected, observed, path)
    return {
        "maximum_absolute_error": maximum,
        "mismatches": mismatches,
        "structure_exact": not mismatches,
    }


def _read_oof_rows(path: Path) -> Dict[str, object]:
    records = []
    with path.open("r", encoding="utf-8-sig", newline="") as handle:
        for row in csv.DictReader(handle):
            records.append(row)
    return {
        "sample_index": np.asarray(
            [int(row["sample_index"]) for row in records], dtype=np.int64
        ),
        "scores": {
            role: np.asarray(
                [float(row[f"{role}_score"]) for row in records],
                dtype=np.float64,
            )
            for role in READOUT_ROLES
        },
        "actions": {
            role: np.asarray(
                [bool(int(row[f"{role}_action"])) for row in records],
                dtype=bool,
            )
            for role in READOUT_ROLES
        },
    }


def replay_formal_a0(args: argparse.Namespace) -> Dict[str, object]:
    summary_path = Path(args.replay_summary).resolve()
    output_dir = Path(args.output_dir).resolve()
    if summary_path != output_dir / "summary.json":
        raise ValueError("replay summary must be output_dir/summary.json")
    replay_path = output_dir / "replay.json"
    if replay_path.exists():
        raise FileExistsError(f"replay already exists: {replay_path}")
    formal_manifest = _verify_manifest(output_dir)
    summary = json.loads(summary_path.read_text(encoding="utf-8"))
    if (
        str(summary.get("mode")) != f"{METHOD}_formal"
        or str(summary.get("protocol_sha256")) != LOCKED["protocol"]
    ):
        raise ValueError("formal summary differs from NSA protocol")
    locks = verify_locked_inputs(args, require_clean=True)
    geometry_review = _verify_geometry_authorization()
    process_before = _process_snapshot()
    gpu_before = _gpu_snapshot()
    if process_before["unexpected_processes"]:
        raise RuntimeError(
            "replay found an unrelated Python/TensorRT process: "
            f"{process_before['unexpected_processes']}"
        )
    device = _resolve_device(str(args.device))
    torch.manual_seed(SEED)
    if device.type == "cuda":
        torch.cuda.manual_seed_all(SEED)
        torch.cuda.empty_cache()
        torch.cuda.reset_peak_memory_stats(device)
    torch.backends.cudnn.benchmark = False
    torch.backends.cudnn.deterministic = True
    torch.use_deterministic_algorithms(True)

    started = time.perf_counter()
    rows = _read_clean_train_rows(Path(args.cidt_predictions))
    cohort = _cohort_from_rows(rows)
    fixed_indices = fixed_visual_indices(cohort)
    worker_benchmark = json.loads(
        WORKER_BENCHMARK_PATH.read_text(encoding="utf-8")
    )
    if int(args.num_workers) != int(worker_benchmark["selected_workers"]):
        raise ValueError("replay worker count differs from locked benchmark")
    fold_scores: Dict[int, Dict[str, object]] = {}
    fold_comparisons = []
    geometry_comparisons = []
    diagnostic_comparisons = []
    fold_diagnostics = []
    adapter_records = []
    visual_rows = []
    with _ResourceMonitor() as resources:
        model, checkpoint, class_names = load_model(
            Path(args.checkpoint).resolve(),
            device=device,
        )
        if len(class_names) != 5:
            raise ValueError("keeper class count differs during replay")
        base_dataset, transform, dataset_summary = _build_dataset(
            checkpoint,
            rows,
            Path(args.data).resolve(),
        )
        transformed_dataset = _SelectedConditionDataset(
            base_dataset,
            range(len(base_dataset)),
            corruption=IdentityCorruption(),
            transform=transform,
        )
        semantics = _eval_semantics(checkpoint)
        keeper_before = _model_state_sha256(model)
        if keeper_before != LOCKED_MODEL_STATE_SHA256:
            raise ValueError("replay keeper state differs from lock")
        for held_fold in FOLDS:
            print(
                json.dumps(
                    {
                        "stage": "nsa_replay_fold_start",
                        "held_fold": held_fold,
                        "elapsed_seconds": time.perf_counter() - started,
                    }
                ),
                flush=True,
            )
            expected_training = summary["folds"][held_fold]
            roles, state_records = _load_adapter_roles(
                output_dir=output_dir,
                held_fold=held_fold,
                device=device,
                expected_hashes=expected_training["final_state_sha256"],
            )
            adapter_records.append(
                {"held_fold": held_fold, "roles": state_records}
            )
            expected_scores = _load_fold_scores(
                output_dir
                / f"fold_{held_fold}"
                / "clean_cohort_scores.npz"
            )
            observed_scores = score_clean_cohort(
                model=model,
                roles=roles,
                transformed_dataset=transformed_dataset,
                semantics=semantics,
                rows=rows,
                cohort_indices=[row.sample_index for row in cohort],
                device=device,
                batch_size=int(args.batch_size),
                num_workers=int(args.num_workers),
            )
            fold_scores[held_fold] = observed_scores
            comparison = _compare_fold_scores(
                expected_scores,
                observed_scores,
            )
            fold_comparisons.append(
                {"held_fold": held_fold, **comparison}
            )
            geometry = replay_training_geometry(
                transformed_dataset=transformed_dataset,
                semantics=semantics,
                rows=rows,
                held_fold=held_fold,
            )
            expected_geometry_hash = str(
                expected_training["geometry_records_sha256"]
            )
            geometry["expected_records_sha256"] = expected_geometry_hash
            geometry["passed"] = (
                str(geometry["records_sha256"]) == expected_geometry_hash
                and int(geometry["record_count"])
                == int(expected_training["geometry_record_count"])
            )
            geometry_comparisons.append(geometry)

            diagnostic = score_held_diagnostics(
                model=model,
                candidate=roles["poisson_query_candidate"],
                transformed_dataset=transformed_dataset,
                semantics=semantics,
                rows=rows,
                held_fold=held_fold,
                device=device,
                batch_size=int(args.batch_size),
                num_workers=int(args.num_workers),
            )
            diagnostic_records = diagnostic.pop("records")
            diagnostic["records_sha256"] = geometry_records_sha256(
                diagnostic_records
            )
            fold_diagnostics.append(diagnostic)
            expected_diagnostic = summary["diagnostics_by_fold"][held_fold]
            diagnostic_difference = _compare_nested(
                expected_diagnostic,
                diagnostic,
                path=f"diagnostic_fold_{held_fold}",
            )
            diagnostic_comparisons.append(
                {
                    "held_fold": held_fold,
                    **diagnostic_difference,
                    "passed": bool(
                        diagnostic_difference["structure_exact"]
                        and diagnostic_difference["maximum_absolute_error"]
                        <= 1e-6
                    ),
                }
            )
            visual_rows.extend(
                collect_fixed_visual_rows(
                    model=model,
                    roles=roles,
                    transformed_dataset=transformed_dataset,
                    semantics=semantics,
                    rows=rows,
                    target_indices=fixed_indices,
                    held_fold=held_fold,
                    device=device,
                )
            )
            del roles
            if device.type == "cuda":
                torch.cuda.empty_cache()
        keeper_after = _model_state_sha256(model)

    visual_rows.sort(
        key=lambda row: fixed_indices.index(int(row["sample_index"]))
    )
    visual_comparison = _compare_visual_payload(
        output_dir=output_dir,
        rows=visual_rows,
    )
    states = json.loads(
        (output_dir / "readout_states.json").read_text(encoding="utf-8")
    )
    readouts = apply_saved_readouts(
        cohort=cohort,
        fold_scores=fold_scores,
        states=states,
    )
    expected_oof = _read_oof_rows(output_dir / "oof_row_scores.csv")
    ordered_indices = np.asarray(
        [row.sample_index for row in cohort], dtype=np.int64
    )
    readout_score_errors = {
        role: float(
            np.max(
                np.abs(
                    np.asarray(readouts["scores"][role], dtype=np.float64)
                    - expected_oof["scores"][role]
                )
            )
        )
        for role in READOUT_ROLES
    }
    readout_actions_exact = all(
        np.array_equal(
            np.asarray(readouts["actions"][role]),
            expected_oof["actions"][role],
        )
        for role in READOUT_ROLES
    )
    metrics = _readout_metrics(cohort=cohort, readouts=readouts)
    query = query_metrics(cohort=cohort, fold_scores=fold_scores)
    diagnostics = _aggregate_diagnostics(fold_diagnostics)
    metrics_difference = _compare_nested(
        summary["metrics"], metrics, path="metrics"
    )
    query_difference = _compare_nested(
        summary["query_metrics"], query, path="query"
    )
    diagnostics_difference = _compare_nested(
        summary["diagnostics"], diagnostics, path="diagnostics"
    )
    process_after = _process_snapshot()
    gpu_after = _gpu_snapshot()
    peak_cuda_bytes = (
        int(torch.cuda.max_memory_allocated(device))
        if device.type == "cuda"
        else 0
    )
    formal_pid = int(
        summary["runtime"]["process_before"]["current_pid"]
    )
    checks = {
        "formal_manifest_verified_before_replay": bool(formal_manifest["passed"]),
        "locked_inputs_and_geometry_review": bool(
            locks["repo"]["head_equals_upstream"]
            and geometry_review["passed"]
        ),
        "fresh_process_pid": formal_pid != os.getpid(),
        "five_adapter_folds_reloaded": len(adapter_records) == len(FOLDS),
        "all_fold_maps_within_1e_6": all(
            bool(row["passed"]) for row in fold_comparisons
        ),
        "all_geometry_hashes_exact": all(
            bool(row["passed"]) for row in geometry_comparisons
        ),
        "all_diagnostics_within_1e_6": all(
            bool(row["passed"]) for row in diagnostic_comparisons
        ),
        "fixed_visual_payload_within_1e_6": bool(
            visual_comparison["passed"]
        ),
        "oof_sample_order_exact": np.array_equal(
            ordered_indices, expected_oof["sample_index"]
        ),
        "readout_scores_within_1e_10": max(
            readout_score_errors.values()
        )
        <= 1e-10,
        "readout_actions_exact": readout_actions_exact,
        "metrics_within_1e_10": bool(
            metrics_difference["structure_exact"]
            and metrics_difference["maximum_absolute_error"] <= 1e-10
        ),
        "query_metrics_within_1e_10": bool(
            query_difference["structure_exact"]
            and query_difference["maximum_absolute_error"] <= 1e-10
        ),
        "aggregate_diagnostics_within_1e_10": bool(
            diagnostics_difference["structure_exact"]
            and diagnostics_difference["maximum_absolute_error"] <= 1e-10
        ),
        "keeper_state_bit_identical": keeper_before
        == keeper_after
        == LOCKED_MODEL_STATE_SHA256,
        "no_unexpected_compute_process_before": not bool(
            process_before["unexpected_processes"]
        ),
        "no_unexpected_compute_process_after": not bool(
            process_after["unexpected_processes"]
        ),
        "peak_cuda_allocation_le_4p5_gib": peak_cuda_bytes <= MAX_CUDA_BYTES,
        "peak_virtual_memory_fraction_lt_0p82": float(
            resources.peak_virtual_memory_fraction
        )
        < MAX_VIRTUAL_MEMORY_FRACTION,
        "no_adapter_or_readout_refit": True,
        "validation_split_unused": True,
        "test_split_unused": True,
        "raw_dataset_unchanged": True,
    }
    passed = all(checks.values())
    replay = {
        "mode": f"{METHOD}_fresh_process_replay",
        "status": "passed" if passed else "failed",
        "passed": passed,
        "protocol_sha256": LOCKED["protocol"],
        "summary_path": str(summary_path),
        "summary_sha256": _sha256(summary_path),
        "formal_manifest_before_replay": formal_manifest,
        "locks": locks,
        "geometry_review": geometry_review,
        "dataset": dataset_summary,
        "adapter_states": adapter_records,
        "fold_score_comparisons": fold_comparisons,
        "geometry_comparisons": geometry_comparisons,
        "diagnostic_comparisons": diagnostic_comparisons,
        "visual_comparison": visual_comparison,
        "readout_score_maximum_absolute_error": readout_score_errors,
        "readout_actions_exact": readout_actions_exact,
        "metrics_difference": metrics_difference,
        "query_difference": query_difference,
        "diagnostics_difference": diagnostics_difference,
        "checks": checks,
        "runtime": {
            "elapsed_seconds": float(time.perf_counter() - started),
            "peak_rss_bytes": int(resources.peak_rss_bytes),
            "peak_virtual_memory_fraction": float(
                resources.peak_virtual_memory_fraction
            ),
            "peak_cuda_bytes": peak_cuda_bytes,
            "process_before": process_before,
            "process_after": process_after,
            "gpu_before": gpu_before,
            "gpu_after": gpu_after,
            "keeper_state_before_sha256": keeper_before,
            "keeper_state_after_sha256": keeper_after,
        },
        "validation_split_used": False,
        "test_split_used": False,
        "raw_dataset_modified": False,
    }
    _write_json(replay_path, _jsonable(replay))
    manifest = _write_manifest(output_dir)
    verified_manifest = _verify_manifest(output_dir)
    return {
        "status": replay["status"],
        "passed": passed,
        "replay_path": str(replay_path),
        "replay_sha256": _sha256(replay_path),
        "manifest_sha256": verified_manifest["manifest_sha256"],
        "manifest_payload_count": len(manifest["artifacts"]),
    }


def finalize_formal_review(args: argparse.Namespace) -> Dict[str, object]:
    output_dir = Path(args.output_dir).resolve()
    summary_path = output_dir / "summary.json"
    replay_path = output_dir / "replay.json"
    contact_path = output_dir / "fixed_visual_contact_sheet.png"
    review_path = output_dir / "formal_visual_review.json"
    if review_path.exists():
        raise FileExistsError(f"formal visual review already exists: {review_path}")
    for path in (summary_path, replay_path, contact_path):
        if not path.is_file():
            raise FileNotFoundError(f"formal review artifact is missing: {path}")
    if args.decision is None:
        raise ValueError("--decision is required for formal visual review")
    manifest_before = _verify_manifest(output_dir)
    summary = json.loads(summary_path.read_text(encoding="utf-8"))
    replay = json.loads(replay_path.read_text(encoding="utf-8"))
    automatic = bool(summary.get("automatic_pass", False))
    replay_pass = bool(replay.get("passed", False))
    manual_pass = args.decision == "pass"
    passed = automatic and replay_pass and manual_pass
    payload = {
        "mode": f"{METHOD}_formal_visual_review",
        "decision": args.decision,
        "review_note": str(args.review_note).strip(),
        "passed": passed,
        "automatic_pass": automatic,
        "fresh_process_replay_pass": replay_pass,
        "contact_sheet_sha256": _sha256(contact_path),
        "summary_sha256": _sha256(summary_path),
        "replay_sha256": _sha256(replay_path),
        "manifest_before_review": manifest_before,
        "reviewed_rows": int(summary["visual_artifact"]["rows"]),
        "review_scope": [
            "mismatch mass follows fruit surface rather than clone seam",
            "no bbox edge, padding, hand, basket, or background shortcut",
            "same-class maps stay materially quieter than cross-class maps",
            "candidate differs meaningfully from capacity-matched controls",
            "five query maps contain class-conditional spatial behavior",
        ],
        "authorization": {
            "default_off_production_integration": passed,
            "validation_smoke": passed,
            "test": False,
            "probe": False,
            "full_train": False,
            "current_best_update": False,
        },
    }
    _write_json(review_path, payload)
    manifest = _write_manifest(output_dir)
    verified_manifest = _verify_manifest(output_dir)
    return {
        "passed": passed,
        "decision": args.decision,
        "review_path": str(review_path),
        "review_sha256": _sha256(review_path),
        "manifest_sha256": verified_manifest["manifest_sha256"],
        "manifest_payload_count": len(manifest["artifacts"]),
        "authorization": payload["authorization"],
    }


def main(argv: Optional[Sequence[str]] = None) -> int:
    args = parse_args(argv)
    modes = [
        bool(args.geometry_preview),
        bool(args.finalize_geometry_review),
        bool(args.finalize_formal_review),
        bool(args.run_a0),
        args.replay_summary is not None,
    ]
    if sum(modes) != 1:
        raise ValueError("select exactly one execution mode")
    if args.geometry_preview:
        result = run_geometry_preview(args)
    elif args.finalize_geometry_review:
        result = finalize_geometry_review(args)
    elif args.finalize_formal_review:
        result = finalize_formal_review(args)
    elif args.run_a0:
        result = run_formal_a0(args)
    else:
        result = replay_formal_a0(args)
    print(json.dumps(result, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
