from __future__ import annotations

import argparse
import hashlib
import json
import math
import subprocess
import time
from collections import defaultdict
from pathlib import Path
from typing import Mapping, Optional, Sequence

import numpy as np
import torch
import torch.nn.functional as F
from PIL import Image, ImageDraw, ImageFilter

from trkh.evaluation.attention_viz import (
    _capture_forward,
    prepare_image_and_tensor,
    save_heatmap_visualizations,
)
from trkh.evaluation.xai_audit import (
    _forward_logits_with_optional_bbox,
    _tensor_from_crop,
)
from trkh.inference.inference import load_model
from trkh.models.feature_hooks import count_attention_layers


REPO_ROOT = Path(__file__).resolve().parents[2]
DEFAULT_LOCK = (
    REPO_ROOT
    / "docs"
    / "TRKH_5CLASS_SASPA_F1_XAI_EXECUTION_LOCK_20260724.json"
)
DEFAULT_GENERATION_ROOT = (
    REPO_ROOT
    / "runs"
    / "audit_saspa_dual_view_synthetic_a0_20260723"
    / "f1_tiny_output_v2"
)
DEFAULT_CHECKPOINT = (
    REPO_ROOT
    / "runs"
    / "probe_v8_yolof_pairroute_teacherfocusbinary015_boundarydrop_bboxprior_120b_2e_20260701"
    / "checkpoints"
    / "best.pt"
)
DEFAULT_OUTPUT_DIR = DEFAULT_GENERATION_ROOT / "xai_audit"
MANIFEST_EXCLUSIONS = {
    "xai_manifest.json",
    "xai_replay.json",
    "xai_final_manifest.json",
}


def parse_args(argv: Optional[Sequence[str]] = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Locked diagnostic XAI for the ten SaSPA A0 F1 outputs."
    )
    parser.add_argument("--lock", type=Path, default=DEFAULT_LOCK)
    parser.add_argument(
        "--generation-root",
        type=Path,
        default=DEFAULT_GENERATION_ROOT,
    )
    parser.add_argument("--checkpoint", type=Path, default=DEFAULT_CHECKPOINT)
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT_DIR)
    mode = parser.add_mutually_exclusive_group(required=True)
    mode.add_argument("--preflight-only", action="store_true")
    mode.add_argument("--formal-audit", action="store_true")
    mode.add_argument("--replay-summary", type=Path)
    return parser.parse_args(argv)


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with Path(path).open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _read_json(path: Path) -> object:
    with Path(path).open("r", encoding="utf-8") as handle:
        return json.load(handle)


def _write_json(path: Path, value: object) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8", newline="\n") as handle:
        json.dump(value, handle, indent=2, sort_keys=True)
        handle.write("\n")


def _read_jsonl(path: Path) -> list[dict[str, object]]:
    rows: list[dict[str, object]] = []
    with Path(path).open("r", encoding="utf-8") as handle:
        for line_number, line in enumerate(handle, start=1):
            text = line.strip()
            if not text:
                continue
            value = json.loads(text)
            if not isinstance(value, dict):
                raise ValueError(f"Expected object at {path}:{line_number}.")
            rows.append(value)
    return rows


def _as_mapping(value: object, name: str) -> Mapping[str, object]:
    if not isinstance(value, Mapping):
        raise ValueError(f"{name} must be an object.")
    return value


def _resolve_inside(root: Path, relative_path: str) -> Path:
    candidate = (root / Path(relative_path)).resolve()
    resolved_root = root.resolve()
    try:
        candidate.relative_to(resolved_root)
    except ValueError as exc:
        raise ValueError(f"Path escapes locked root: {relative_path}") from exc
    return candidate


def _validate_fixed_inputs(
    *,
    lock: Mapping[str, object],
    generation_root: Path,
    checkpoint: Path,
) -> dict[str, object]:
    failures: list[str] = []
    fixed_inputs = _as_mapping(lock.get("fixed_inputs"), "fixed_inputs")
    observed_inputs: dict[str, str] = {}
    for relative_path, expected_hash in fixed_inputs.items():
        path = _resolve_inside(generation_root, str(relative_path))
        if not path.is_file():
            failures.append(f"missing_fixed_input:{relative_path}")
            continue
        observed = _sha256(path)
        observed_inputs[str(relative_path)] = observed
        if observed != str(expected_hash):
            failures.append(f"fixed_input_hash:{relative_path}")

    checkpoint_lock = _as_mapping(lock.get("checkpoint"), "checkpoint")
    if not checkpoint.is_file():
        failures.append("missing_checkpoint")
        checkpoint_hash = None
        checkpoint_bytes = None
    else:
        checkpoint_hash = _sha256(checkpoint)
        checkpoint_bytes = int(checkpoint.stat().st_size)
        if checkpoint_hash != str(checkpoint_lock.get("sha256")):
            failures.append("checkpoint_hash")
        if checkpoint_bytes != int(checkpoint_lock.get("bytes", -1)):
            failures.append("checkpoint_bytes")

    provenance_path = generation_root / "attempt_enable_model_cpu_offload" / "provenance.jsonl"
    provenance = (
        sorted(_read_jsonl(provenance_path), key=lambda row: str(row["output_id"]))
        if provenance_path.is_file()
        else []
    )
    expected_count = int(
        _as_mapping(lock.get("sample_policy"), "sample_policy").get("count", -1)
    )
    if len(provenance) != expected_count:
        failures.append("provenance_count")
    output_ids = [str(row.get("output_id", "")) for row in provenance]
    if len(set(output_ids)) != len(output_ids) or any(not item for item in output_ids):
        failures.append("provenance_output_ids")

    output_rows: list[dict[str, object]] = []
    attempt_root = generation_root / "attempt_enable_model_cpu_offload"
    for row in provenance:
        relative_path = str(row.get("output_relative_path", ""))
        path = _resolve_inside(attempt_root, relative_path)
        if not path.is_file():
            failures.append(f"missing_output:{row.get('output_id')}")
            continue
        observed = _sha256(path)
        if observed != str(row.get("output_sha256", "")):
            failures.append(f"output_hash:{row.get('output_id')}")
        bbox = row.get("bbox_xywh_normalized")
        if (
            not isinstance(bbox, list)
            or len(bbox) != 4
            or not all(math.isfinite(float(item)) for item in bbox)
        ):
            failures.append(f"bbox:{row.get('output_id')}")
        target = int(row.get("target_class_index", -1))
        if target < 0 or target > 4:
            failures.append(f"target:{row.get('output_id')}")
        output_rows.append(
            {
                "output_id": str(row.get("output_id")),
                "path": str(path),
                "sha256": observed,
                "bbox_xywh_normalized": [float(item) for item in bbox],
                "target_class_index": target,
            }
        )

    generation_summary = _read_json(generation_root / "generation_summary.json")
    fidelity = _read_json(generation_root / "generated_fidelity.json")
    blind = _read_json(generation_root / "blind_review_unblinded.json")
    duplicate = _read_json(
        generation_root / "duplicate_audit" / "duplicate_report.json"
    )
    generation_summary = _as_mapping(generation_summary, "generation_summary")
    fidelity = _as_mapping(fidelity, "generated_fidelity")
    blind = _as_mapping(blind, "blind_review_unblinded")
    duplicate = _as_mapping(duplicate, "duplicate_report")
    if bool(generation_summary.get("passed_train_only_fidelity")):
        failures.append("upstream_fidelity_unexpected_pass")
    if bool(fidelity.get("passed")):
        failures.append("fidelity_report_unexpected_pass")
    if bool(blind.get("passed")):
        failures.append("blind_review_unexpected_pass")
    if not bool(duplicate.get("passed")):
        failures.append("duplicate_audit_not_passed")

    return {
        "passed": not failures,
        "failures": failures,
        "checkpoint_sha256": checkpoint_hash,
        "checkpoint_bytes": checkpoint_bytes,
        "fixed_inputs": observed_inputs,
        "output_rows": output_rows,
        "upstream_decision": {
            "fidelity_passed": bool(fidelity.get("passed")),
            "blind_review_passed": bool(blind.get("passed")),
            "duplicate_audit_passed": bool(duplicate.get("passed")),
            "f1_already_rejected": (
                not bool(fidelity.get("passed"))
                and not bool(blind.get("passed"))
                and bool(duplicate.get("passed"))
            ),
        },
    }


def bbox_mask(
    image_size: tuple[int, int],
    bbox_xywh_normalized: Sequence[float],
) -> np.ndarray:
    width, height = (int(image_size[0]), int(image_size[1]))
    if width <= 0 or height <= 0 or len(bbox_xywh_normalized) != 4:
        raise ValueError("Invalid image size or bbox.")
    cx, cy, box_width, box_height = (
        float(item) for item in bbox_xywh_normalized
    )
    left = max(0, min(width - 1, int(math.floor((cx - box_width / 2.0) * width))))
    top = max(0, min(height - 1, int(math.floor((cy - box_height / 2.0) * height))))
    right = max(left + 1, min(width, int(math.ceil((cx + box_width / 2.0) * width))))
    bottom = max(top + 1, min(height, int(math.ceil((cy + box_height / 2.0) * height))))
    mask = np.zeros((height, width), dtype=bool)
    mask[top:bottom, left:right] = True
    return mask


def perturb_image(
    image: Image.Image,
    mask: np.ndarray,
    mode: str,
) -> Image.Image:
    rgb = image.convert("RGB")
    base = np.asarray(rgb, dtype=np.uint8)
    if mask.shape != base.shape[:2]:
        raise ValueError("Mask and image shapes differ.")
    gray = np.asarray(rgb.convert("L").convert("RGB"), dtype=np.uint8)
    if mode == "background_gray":
        result = np.where(mask[:, :, None], base, gray)
    elif mode == "background_blur":
        blurred = np.asarray(
            rgb.filter(ImageFilter.GaussianBlur(radius=8)),
            dtype=np.uint8,
        )
        result = np.where(mask[:, :, None], base, blurred)
    elif mode == "object_desaturate":
        result = np.where(mask[:, :, None], gray, base)
    else:
        raise ValueError(f"Unknown perturbation mode: {mode}")
    return Image.fromarray(result)


def normalize_input_gradient(gradient: torch.Tensor) -> np.ndarray:
    if gradient.ndim != 4 or int(gradient.shape[0]) != 1 or int(gradient.shape[1]) != 3:
        raise ValueError(f"Expected 1x3xHxW input gradient, got {tuple(gradient.shape)}.")
    values = torch.linalg.vector_norm(
        gradient.detach().to(dtype=torch.float32),
        ord=2,
        dim=1,
    )[0]
    if not bool(torch.isfinite(values).all()):
        raise ValueError("Input gradient contains non-finite values.")
    maximum = float(values.max().item())
    if maximum > 0.0:
        values = values / maximum
    else:
        values = torch.zeros_like(values)
    return values.cpu().numpy().astype(np.float32, copy=False)


def heatmap_bbox_focus(heatmap: np.ndarray, mask: np.ndarray) -> dict[str, float]:
    heat = np.asarray(heatmap, dtype=np.float64)
    if heat.ndim != 2:
        raise ValueError("Heatmap must be two-dimensional.")
    if heat.shape != mask.shape:
        heat_image = Image.fromarray(heat.astype(np.float32), mode="F")
        heat = np.asarray(
            heat_image.resize(
                (int(mask.shape[1]), int(mask.shape[0])),
                resample=Image.Resampling.BILINEAR,
            ),
            dtype=np.float64,
        )
    heat = np.maximum(heat, 0.0)
    total = float(heat.sum())
    if total <= 0.0:
        return {"bbox_mass": 0.0, "outside_bbox_mass": 0.0}
    bbox_mass = float(heat[mask].sum() / total)
    return {
        "bbox_mass": bbox_mass,
        "outside_bbox_mass": float(1.0 - bbox_mass),
    }


def _run_input_gradient(
    *,
    model: torch.nn.Module,
    tensor: torch.Tensor,
    bbox: torch.Tensor,
    prediction_index: int,
) -> tuple[np.ndarray, list[float]]:
    model.zero_grad(set_to_none=True)
    input_tensor = tensor.detach().clone().requires_grad_(True)
    with torch.enable_grad():
        logits = _forward_logits_with_optional_bbox(
            model,
            input_tensor,
            bbox,
            bbox_token_prior=bbox,
        )
        probabilities = F.softmax(logits.float(), dim=1)[0]
        logits[:, int(prediction_index)].sum().backward()
    if input_tensor.grad is None:
        raise RuntimeError("Input gradient was not captured.")
    return (
        normalize_input_gradient(input_tensor.grad),
        [float(item) for item in probabilities.detach().cpu().tolist()],
    )


def _run_perturbations(
    *,
    model: torch.nn.Module,
    checkpoint: Mapping[str, object],
    clean_tensor: torch.Tensor,
    clean_probabilities: Sequence[float],
    bbox: torch.Tensor,
    crop_image: Image.Image,
    mask: np.ndarray,
    case_dir: Path,
    prediction_index: int,
    target_index: int,
) -> dict[str, object]:
    variants = {
        "background_gray": perturb_image(crop_image, mask, "background_gray"),
        "background_blur": perturb_image(crop_image, mask, "background_blur"),
        "object_desaturate": perturb_image(crop_image, mask, "object_desaturate"),
    }
    for name, image in variants.items():
        image.save(case_dir / f"{name}.png")
    Image.fromarray(mask.astype(np.uint8) * 255, mode="L").save(
        case_dir / "bbox_mask.png"
    )

    clean = np.asarray(clean_probabilities, dtype=np.float64)
    results: dict[str, object] = {}
    model.eval()
    with torch.no_grad():
        clean_logits = _forward_logits_with_optional_bbox(
            model,
            clean_tensor,
            bbox,
            bbox_token_prior=bbox,
        )
        clean_recheck = F.softmax(clean_logits.float(), dim=1)[0].cpu().numpy()
        results["clean_recheck_max_probability_error"] = float(
            np.max(np.abs(clean_recheck - clean))
        )
        for name, image in variants.items():
            variant_tensor = _tensor_from_crop(
                dict(checkpoint),
                image,
                clean_tensor.device,
            )
            logits = _forward_logits_with_optional_bbox(
                model,
                variant_tensor,
                bbox,
                bbox_token_prior=bbox,
            )
            probabilities = F.softmax(logits.float(), dim=1)[0].cpu().numpy()
            results[name] = {
                "prediction_index": int(probabilities.argmax()),
                "prediction_probability": float(probabilities.max()),
                "clean_prediction_probability_drop": float(
                    clean[int(prediction_index)] - probabilities[int(prediction_index)]
                ),
                "target_probability_drop": float(
                    clean[int(target_index)] - probabilities[int(target_index)]
                ),
                "probabilities": [float(item) for item in probabilities.tolist()],
            }
    return results


def _fit_image(image: Image.Image, size: tuple[int, int]) -> Image.Image:
    result = image.convert("RGB").copy()
    result.thumbnail(size, Image.Resampling.LANCZOS)
    canvas = Image.new("RGB", size, (245, 247, 249))
    left = (size[0] - result.width) // 2
    top = (size[1] - result.height) // 2
    canvas.paste(result, (left, top))
    return canvas


def _contact_sheet(
    *,
    case_rows: Sequence[Mapping[str, object]],
    output_path: Path,
) -> None:
    columns = (
        ("synthetic_path", "Synthetic"),
        ("gradcam_overlay", "Stem Grad-CAM"),
        ("input_gradient_overlay", "Input gradient"),
        ("background_blur_path", "Background blur"),
        ("object_desaturate_path", "Object desaturation"),
    )
    cell = (240, 240)
    header_height = 34
    row_label_height = 32
    sheet = Image.new(
        "RGB",
        (
            len(columns) * cell[0],
            header_height + len(case_rows) * (cell[1] + row_label_height),
        ),
        (255, 255, 255),
    )
    draw = ImageDraw.Draw(sheet)
    for column_index, (_, title) in enumerate(columns):
        draw.text(
            (column_index * cell[0] + 8, 10),
            title,
            fill=(20, 25, 30),
        )
    for row_index, row in enumerate(case_rows):
        y = header_height + row_index * (cell[1] + row_label_height)
        label = (
            f"{row['output_id']}  target={row['target_class_index']} "
            f"pred={row['prediction_index']} p={float(row['confidence']):.3f}"
        )
        draw.text((8, y + 7), label, fill=(20, 25, 30))
        for column_index, (path_key, _) in enumerate(columns):
            with Image.open(str(row[path_key])) as handle:
                fitted = _fit_image(handle, cell)
            sheet.paste(
                fitted,
                (
                    column_index * cell[0],
                    y + row_label_height,
                ),
            )
    output_path.parent.mkdir(parents=True, exist_ok=True)
    sheet.save(output_path)


def _repo_state() -> dict[str, object]:
    def run(*args: str) -> str:
        return subprocess.check_output(
            ["git", "-C", str(REPO_ROOT), *args],
            text=True,
            encoding="utf-8",
        ).strip()

    return {
        "head": run("rev-parse", "HEAD"),
        "upstream": run("rev-parse", "@{upstream}"),
        "tracked_status": run("status", "--short", "--untracked-files=no"),
    }


def _artifact_manifest(root: Path, output_name: str) -> dict[str, object]:
    files: list[dict[str, object]] = []
    for path in sorted(item for item in root.rglob("*") if item.is_file()):
        relative = path.relative_to(root).as_posix()
        if relative in MANIFEST_EXCLUSIONS or relative == output_name:
            continue
        files.append(
            {
                "relative_path": relative,
                "bytes": int(path.stat().st_size),
                "sha256": _sha256(path),
            }
        )
    digest = hashlib.sha256(
        json.dumps(files, sort_keys=True, separators=(",", ":")).encode("utf-8")
    ).hexdigest()
    return {
        "schema": "trkh_saspa_f1_xai_artifact_manifest_v1",
        "root": str(root.resolve()),
        "file_count": len(files),
        "files_sha256": digest,
        "files": files,
    }


def _check_manifest(root: Path, manifest: Mapping[str, object]) -> dict[str, object]:
    failures: list[str] = []
    expected = {
        str(row["relative_path"]): row
        for row in manifest.get("files", [])
        if isinstance(row, Mapping)
    }
    observed = _artifact_manifest(root, "xai_manifest.json")
    observed_rows = {
        str(row["relative_path"]): row
        for row in observed["files"]
        if isinstance(row, Mapping)
    }
    if expected != observed_rows:
        missing = sorted(set(expected) - set(observed_rows))
        extra = sorted(set(observed_rows) - set(expected))
        changed = sorted(
            path
            for path in set(expected) & set(observed_rows)
            if expected[path] != observed_rows[path]
        )
        failures.extend(f"missing:{item}" for item in missing)
        failures.extend(f"extra:{item}" for item in extra)
        failures.extend(f"changed:{item}" for item in changed)
    if str(manifest.get("files_sha256")) != str(observed.get("files_sha256")):
        failures.append("files_sha256")
    return {
        "passed": not failures,
        "failures": failures,
        "observed_file_count": observed["file_count"],
        "observed_files_sha256": observed["files_sha256"],
    }


def run_preflight(args: argparse.Namespace) -> dict[str, object]:
    lock = _as_mapping(_read_json(args.lock), "lock")
    checks = _validate_fixed_inputs(
        lock=lock,
        generation_root=args.generation_root,
        checkpoint=args.checkpoint,
    )
    result = {
        "schema": "trkh_saspa_f1_xai_preflight_v1",
        "lock_sha256": _sha256(args.lock),
        "generation_root": str(args.generation_root.resolve()),
        "checkpoint": str(args.checkpoint.resolve()),
        "checks": checks,
        "passed": bool(checks["passed"]),
        "synthetic_pixels_opened": False,
        "source_train_pixels_opened": False,
        "validation_test_pixels_opened": False,
        "model_inference_performed": False,
        "a1_authorized": False,
        "training_authorized": False,
    }
    print(json.dumps(result, indent=2, sort_keys=True))
    if not result["passed"]:
        raise RuntimeError(f"SaSPA F1 XAI preflight failed: {checks['failures']}")
    return result


def run_formal(args: argparse.Namespace) -> dict[str, object]:
    if args.output_dir.exists() and any(args.output_dir.iterdir()):
        raise FileExistsError(
            f"XAI output directory is not empty: {args.output_dir}"
        )
    args.output_dir.mkdir(parents=True, exist_ok=True)
    started = time.time()
    lock = _as_mapping(_read_json(args.lock), "lock")
    checks = _validate_fixed_inputs(
        lock=lock,
        generation_root=args.generation_root,
        checkpoint=args.checkpoint,
    )
    if not checks["passed"]:
        raise RuntimeError(f"SaSPA F1 XAI fixed-input checks failed: {checks['failures']}")
    repo_state = _repo_state()
    if repo_state["tracked_status"]:
        raise RuntimeError("Formal SaSPA F1 XAI requires a clean tracked worktree.")
    if repo_state["head"] != repo_state["upstream"]:
        raise RuntimeError("Formal SaSPA F1 XAI requires HEAD to equal upstream.")

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model, checkpoint, class_names = load_model(args.checkpoint, device)
    model.to(device).eval()
    if len(class_names) != 5:
        raise ValueError(f"Expected five keeper classes, got {len(class_names)}.")
    layer_index = max(0, count_attention_layers(model) - 1)
    case_rows: list[dict[str, object]] = []
    probability_error_limit = float(
        _as_mapping(lock.get("failure_policy"), "failure_policy").get(
            "require_clean_forward_recheck_max_probability_error_at_most",
            1e-6,
        )
    )

    for order, source_row in enumerate(checks["output_rows"], start=1):
        output_id = str(source_row["output_id"])
        target_index = int(source_row["target_class_index"])
        bbox_values = [float(item) for item in source_row["bbox_xywh_normalized"]]
        case_dir = args.output_dir / f"case_{order:02d}_{output_id}"
        case_dir.mkdir(parents=True, exist_ok=False)
        crop_image, tensor = prepare_image_and_tensor(
            checkpoint=checkpoint,
            image_path=Path(str(source_row["path"])),
            device=device,
        )
        bbox = torch.tensor(
            [bbox_values],
            dtype=torch.float32,
            device=device,
        )
        capture = _capture_forward(
            model=model,
            tensor=tensor,
            crop_image=crop_image,
            layer_index=layer_index,
            head_reduction="mean",
            query_tokens="cls_register_mean",
            method="gradcam",
            target_class=None,
            feature_source="stem_last",
            rollout_start_layer=0,
            bbox_metadata=bbox,
            bbox_token_prior=bbox,
        )
        probabilities_tensor = capture["probabilities"]
        if not torch.is_tensor(probabilities_tensor):
            raise TypeError("Grad-CAM capture did not return probabilities.")
        probabilities = [float(item) for item in probabilities_tensor.tolist()]
        if not all(math.isfinite(item) for item in probabilities):
            raise ValueError(f"Non-finite probabilities for {output_id}.")
        prediction_index = int(capture["predicted_class"])
        if prediction_index != int(np.argmax(probabilities)):
            raise RuntimeError(f"Prediction mismatch for {output_id}.")
        gradcam = np.asarray(capture.get("gradcam_heatmap"), dtype=np.float32)
        if gradcam.ndim != 2 or not np.isfinite(gradcam).all():
            raise ValueError(f"Invalid Grad-CAM heatmap for {output_id}.")
        gradcam_files = save_heatmap_visualizations(
            crop_image=crop_image,
            heatmap=gradcam,
            output_dir=case_dir,
            alpha=0.45,
            prefix="gradcam",
        )

        input_gradient, gradient_probabilities = _run_input_gradient(
            model=model,
            tensor=tensor,
            bbox=bbox,
            prediction_index=prediction_index,
        )
        gradient_probability_error = float(
            np.max(
                np.abs(
                    np.asarray(gradient_probabilities)
                    - np.asarray(probabilities)
                )
            )
        )
        input_gradient_files = save_heatmap_visualizations(
            crop_image=crop_image,
            heatmap=input_gradient,
            output_dir=case_dir,
            alpha=0.45,
            prefix="input_gradient",
        )

        mask = bbox_mask(crop_image.size, bbox_values)
        perturbations = _run_perturbations(
            model=model,
            checkpoint=checkpoint,
            clean_tensor=tensor,
            clean_probabilities=probabilities,
            bbox=bbox,
            crop_image=crop_image,
            mask=mask,
            case_dir=case_dir,
            prediction_index=prediction_index,
            target_index=target_index,
        )
        maximum_recheck_error = max(
            float(perturbations["clean_recheck_max_probability_error"]),
            gradient_probability_error,
        )
        if maximum_recheck_error > probability_error_limit:
            raise RuntimeError(
                f"Clean forward drift for {output_id}: {maximum_recheck_error}"
            )
        case = {
            "schema": "trkh_saspa_f1_xai_case_v1",
            "order": order,
            "output_id": output_id,
            "synthetic_path": str(Path(str(source_row["path"])).resolve()),
            "synthetic_sha256": str(source_row["sha256"]),
            "target_class_index": target_index,
            "target_class_name": class_names[target_index],
            "bbox_xywh_normalized": bbox_values,
            "bbox_area_ratio": float(mask.mean()),
            "prediction_index": prediction_index,
            "prediction_name": class_names[prediction_index],
            "confidence": float(max(probabilities)),
            "correct": prediction_index == target_index,
            "probabilities": probabilities,
            "objective": "clean_top1_logit",
            "gradcam": {
                "feature_source": str(capture["feature_source"]),
                "files": gradcam_files,
                "focus": heatmap_bbox_focus(gradcam, mask),
            },
            "input_gradient": {
                "reduction": "channelwise_l2_normalized_by_finite_maximum",
                "files": input_gradient_files,
                "focus": heatmap_bbox_focus(input_gradient, mask),
                "clean_probability_maximum_error": gradient_probability_error,
            },
            "perturbations": perturbations,
            "diagnostic_only": True,
            "a1_authorized": False,
            "training_authorized": False,
        }
        _write_json(case_dir / "case.json", case)
        case_rows.append(
            {
                **case,
                "gradcam_overlay": gradcam_files["overlay"],
                "input_gradient_overlay": input_gradient_files["overlay"],
                "background_blur_path": str(
                    (case_dir / "background_blur.png").resolve()
                ),
                "object_desaturate_path": str(
                    (case_dir / "object_desaturate.png").resolve()
                ),
            }
        )
        print(
            json.dumps(
                {
                    "output_id": output_id,
                    "target": target_index,
                    "prediction": prediction_index,
                    "confidence": max(probabilities),
                    "correct": prediction_index == target_index,
                },
                sort_keys=True,
            ),
            flush=True,
        )

    contact_sheet_path = args.output_dir / "xai_contact_sheet.png"
    _contact_sheet(case_rows=case_rows, output_path=contact_sheet_path)
    per_class: dict[int, dict[str, int]] = defaultdict(
        lambda: {"count": 0, "correct": 0}
    )
    for row in case_rows:
        target = int(row["target_class_index"])
        per_class[target]["count"] += 1
        per_class[target]["correct"] += int(bool(row["correct"]))

    def mean_drop(probe: str, metric: str) -> float:
        return float(
            np.mean(
                [
                    float(row["perturbations"][probe][metric])
                    for row in case_rows
                ]
            )
        )

    summary = {
        "schema": "trkh_saspa_f1_xai_summary_v1",
        "status": "diagnostic_complete_upstream_rejected",
        "started_unix": started,
        "finished_unix": time.time(),
        "execution_lock": str(args.lock.resolve()),
        "execution_lock_sha256": _sha256(args.lock),
        "generation_root": str(args.generation_root.resolve()),
        "checkpoint": str(args.checkpoint.resolve()),
        "checkpoint_sha256": checks["checkpoint_sha256"],
        "repo_state": repo_state,
        "device": str(device),
        "precision": "fp32",
        "class_names": class_names,
        "source_output_count": len(case_rows),
        "case_count": len(case_rows),
        "correct_count": sum(int(bool(row["correct"])) for row in case_rows),
        "accuracy": float(np.mean([bool(row["correct"]) for row in case_rows])),
        "per_class": {
            str(key): value for key, value in sorted(per_class.items())
        },
        "mean_gradcam_bbox_mass": float(
            np.mean([row["gradcam"]["focus"]["bbox_mass"] for row in case_rows])
        ),
        "mean_input_gradient_bbox_mass": float(
            np.mean(
                [row["input_gradient"]["focus"]["bbox_mass"] for row in case_rows]
            )
        ),
        "robustness_mean": {
            "background_gray_clean_prediction_probability_drop": mean_drop(
                "background_gray",
                "clean_prediction_probability_drop",
            ),
            "background_blur_clean_prediction_probability_drop": mean_drop(
                "background_blur",
                "clean_prediction_probability_drop",
            ),
            "object_desaturate_clean_prediction_probability_drop": mean_drop(
                "object_desaturate",
                "clean_prediction_probability_drop",
            ),
            "background_gray_target_probability_drop": mean_drop(
                "background_gray",
                "target_probability_drop",
            ),
            "background_blur_target_probability_drop": mean_drop(
                "background_blur",
                "target_probability_drop",
            ),
            "object_desaturate_target_probability_drop": mean_drop(
                "object_desaturate",
                "target_probability_drop",
            ),
        },
        "maximum_clean_forward_recheck_probability_error": max(
            max(
                float(row["perturbations"]["clean_recheck_max_probability_error"]),
                float(
                    row["input_gradient"][
                        "clean_probability_maximum_error"
                    ]
                ),
            )
            for row in case_rows
        ),
        "contact_sheet": str(contact_sheet_path.resolve()),
        "case_rows": [
            {
                key: value
                for key, value in row.items()
                if key
                not in {
                    "gradcam_overlay",
                    "input_gradient_overlay",
                    "background_blur_path",
                    "object_desaturate_path",
                }
            }
            for row in case_rows
        ],
        "upstream_decision": checks["upstream_decision"],
        "xai_may_change_upstream_decision": False,
        "raw_dataset_modified": False,
        "source_train_pixels_opened_by_xai": False,
        "validation_test_pixels_opened_by_xai": False,
        "validation_test_labels_opened_by_xai": False,
        "output_filtered": False,
        "output_regenerated": False,
        "a1_authorized": False,
        "training_authorized": False,
        "full_train_authorized": False,
        "current_best_command_updated": False,
    }
    _write_json(args.output_dir / "xai_summary.json", summary)
    manifest = _artifact_manifest(args.output_dir, "xai_manifest.json")
    _write_json(args.output_dir / "xai_manifest.json", manifest)
    printed = {
        "summary_sha256": _sha256(args.output_dir / "xai_summary.json"),
        "manifest_sha256": _sha256(args.output_dir / "xai_manifest.json"),
        "case_count": summary["case_count"],
        "accuracy": summary["accuracy"],
        "passed_execution": (
            summary["case_count"] == 10
            and summary["maximum_clean_forward_recheck_probability_error"]
            <= probability_error_limit
        ),
        "a1_authorized": False,
    }
    print(json.dumps(printed, indent=2, sort_keys=True))
    return summary


def replay(summary_path: Path) -> dict[str, object]:
    summary_path = summary_path.resolve()
    root = summary_path.parent
    summary = _as_mapping(_read_json(summary_path), "xai_summary")
    lock_path = Path(str(summary["execution_lock"]))
    generation_root = Path(str(summary["generation_root"]))
    checkpoint = Path(str(summary["checkpoint"]))
    lock = _as_mapping(_read_json(lock_path), "lock")
    fixed = _validate_fixed_inputs(
        lock=lock,
        generation_root=generation_root,
        checkpoint=checkpoint,
    )
    manifest_path = root / "xai_manifest.json"
    manifest = _as_mapping(_read_json(manifest_path), "xai_manifest")
    manifest_check = _check_manifest(root, manifest)
    failures: list[str] = []
    if not fixed["passed"]:
        failures.extend(f"fixed:{item}" for item in fixed["failures"])
    if not manifest_check["passed"]:
        failures.extend(
            f"manifest:{item}" for item in manifest_check["failures"]
        )
    if int(summary.get("case_count", -1)) != 10:
        failures.append("case_count")
    if int(summary.get("source_output_count", -1)) != 10:
        failures.append("source_output_count")
    if bool(summary.get("a1_authorized")) or bool(
        summary.get("training_authorized")
    ):
        failures.append("premature_authorization")
    if bool(summary.get("output_filtered")) or bool(
        summary.get("output_regenerated")
    ):
        failures.append("xai_changed_outputs")
    if bool(summary.get("validation_test_pixels_opened_by_xai")):
        failures.append("split_boundary")
    result = {
        "schema": "trkh_saspa_f1_xai_replay_v1",
        "summary_sha256": _sha256(summary_path),
        "manifest_sha256": _sha256(manifest_path),
        "fixed_input_checks": fixed,
        "manifest_check": manifest_check,
        "failures": failures,
        "passed": not failures,
        "model_inference_replayed": False,
        "synthetic_pixels_decoded": False,
        "source_train_pixels_opened": False,
        "validation_test_pixels_opened": False,
        "output_filtered": False,
        "output_regenerated": False,
        "a1_authorized": False,
        "training_authorized": False,
    }
    _write_json(root / "xai_replay.json", result)
    final_manifest = _artifact_manifest(root, "xai_final_manifest.json")
    _write_json(root / "xai_final_manifest.json", final_manifest)
    printed = dict(result)
    printed["final_manifest_sha256"] = _sha256(
        root / "xai_final_manifest.json"
    )
    print(json.dumps(printed, indent=2, sort_keys=True))
    if failures:
        raise RuntimeError(f"SaSPA F1 XAI replay failed: {failures}")
    return result


def main(argv: Optional[Sequence[str]] = None) -> None:
    args = parse_args(argv)
    if args.preflight_only:
        run_preflight(args)
    elif args.formal_audit:
        run_formal(args)
    else:
        replay(args.replay_summary)


if __name__ == "__main__":
    main()
