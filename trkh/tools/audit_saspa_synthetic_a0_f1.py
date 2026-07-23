from __future__ import annotations

"""Locked SaSPA A0 F1 tiny-output generation and train-only fidelity audit."""

import argparse
import hashlib
import json
import math
import os
import shutil
import subprocess
import sys
import time
import traceback
from pathlib import Path
from typing import Dict, Iterable, List, Mapping, Optional, Sequence, Set, Tuple

from trkh.tools import audit_saspa_synthetic_a0_f0 as f0


METHOD = "saspa_dual_view_synthetic_a0_f1"
SCHEMA = "trkh_saspa_dual_view_synthetic_a0_f1_generation_v1"
WORKER_SCHEMA = "trkh_saspa_dual_view_synthetic_a0_f1_worker_v1"
REPO_ROOT = Path(__file__).resolve().parents[2]
PROTOCOL_PATH = f0.PROTOCOL_PATH
REQUIREMENTS_PATH = f0.REQUIREMENTS_PATH
IMPLEMENTATION_LOCK_PATH = (
    REPO_ROOT
    / "docs"
    / "TRKH_5CLASS_SASPA_F1_IMPLEMENTATION_LOCK_20260724.json"
)
IMPLEMENTATION_LOCK_MD_PATH = (
    REPO_ROOT
    / "docs"
    / "TRKH_5CLASS_SASPA_F1_IMPLEMENTATION_LOCK_20260724.md"
)
IMPLEMENTATION_LOCK_SHA_PATH = (
    REPO_ROOT
    / "docs"
    / "TRKH_5CLASS_SASPA_F1_IMPLEMENTATION_LOCK_20260724.sha256"
)
F0_ROOT = f0.DEFAULT_OUTPUT_DIR
DEFAULT_OUTPUT_DIR = F0_ROOT / "f1_tiny_output"

LOCKED_IMPLEMENTATION_JSON_SHA256 = (
    "e7d9f868bdca078a08502d525b115704d7909db4018e3f297728acf1196c4cf3"
)
LOCKED_IMPLEMENTATION_MD_SHA256 = (
    "3498bd64c972f9286f030ef23950814d5b7d0d5c75e1f635f2f0bf1bcb5bc9b4"
)
LOCKED_IMPLEMENTATION_SHA_FILE_SHA256 = (
    "7c9741f44fd93ff5a4bbb52071a05ae9405d8dec077e6f730835ac80db230adb"
)
LOCKED_F0_SUMMARY_SHA256 = (
    "cf9046c929fda8bb0698c1a9a55e4aa0484856adf3664980371a0e8528fe257a"
)
LOCKED_F0_MANIFEST_SHA256 = (
    "682f2c8b205104b72a8581be4d8ef85fea0177dbe60f23f9672d20e0859aa712"
)
LOCKED_F0_REPLAY_SHA256 = (
    "27beea013ca9434968a89f2b45bfd9d456bf73558d870fb93775c8b76e2836ba"
)
LOCKED_F0_FINAL_MANIFEST_SHA256 = (
    "57003d4a9357b6e34d408fac164886c24a228f078891dbc51b0153521585dd63"
)
LOCKED_SOURCE_SELECTION_SHA256 = (
    "bca9852568d372f9589a278ca09f9858d6cc3988ca0f5700ceb063cda94744e9"
)
LOCKED_SELECTION_CONTENT_SHA256 = (
    "206ff06b64abb289c451816ca07f28e78f592cba272df67f6a84694ae87fa28b"
)
LOCKED_SASPA_UTILS_SHA256 = (
    "5462c1df96f375cb475f49a5588836598e5267887b5cde349d640de17529ca6c"
)
LOCKED_PIPELINE_SOURCE_SHA256 = (
    "47bdc9e5073233bbb98e34fcdc54c50ef0d34dbd16b15d387538c144ffc11baf"
)
EXPECTED_OUTPUTS = 10
REFERENCE_PAIRS_PER_CLASS = 256
METRIC_SIZE = 256
PROVENANCE_FIELDS = (
    "protocol_id",
    "output_id",
    "target_class_index",
    "target_class_folder",
    "edge_relative_path",
    "edge_leakage_group",
    "edge_file_sha256",
    "edge_label_file_sha256",
    "bbox_xywh_normalized",
    "subject_relative_path",
    "subject_leakage_group",
    "subject_file_sha256",
    "prompt_template_index",
    "prompt_sha256",
    "seed",
    "generator_repository",
    "generator_revision",
    "runtime_versions",
    "offload_mode",
    "generation_seconds",
    "torch_peak_allocated_bytes",
    "torch_peak_reserved_bytes",
    "nvml_peak_used_bytes",
    "output_sha256",
    "decoded_rgb_sha256",
    "preprocessing_sha256",
    "license_provenance",
)


def parse_args(argv: Optional[Sequence[str]] = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Locked SaSPA A0 F1 tiny-output generation audit."
    )
    parser.add_argument("--protocol", type=Path, default=PROTOCOL_PATH)
    parser.add_argument(
        "--implementation-lock",
        type=Path,
        default=IMPLEMENTATION_LOCK_PATH,
    )
    parser.add_argument("--requirements", type=Path, default=REQUIREMENTS_PATH)
    parser.add_argument("--f0-root", type=Path, default=F0_ROOT)
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT_DIR)
    mode = parser.add_mutually_exclusive_group(required=True)
    mode.add_argument("--preflight-only", action="store_true")
    mode.add_argument("--formal-generate", action="store_true")
    mode.add_argument("--replay-summary", type=Path)
    mode.add_argument(
        "--generation-worker",
        action="store_true",
        help=argparse.SUPPRESS,
    )
    parser.add_argument("--worker-input", type=Path, help=argparse.SUPPRESS)
    parser.add_argument("--worker-result", type=Path, help=argparse.SUPPRESS)
    parser.add_argument(
        "--worker-allowed-pid",
        action="append",
        type=int,
        default=[],
        help=argparse.SUPPRESS,
    )
    return parser.parse_args(argv)


def _canonical_sha256(value: object) -> str:
    return f0._canonical_sha256(value)


def _sha256(path: Path) -> str:
    return f0._sha256(path)


def _read_json(path: Path) -> object:
    return f0._read_json(path)


def _write_json(path: Path, value: object) -> None:
    f0._write_json(path, value)


def _write_jsonl(path: Path, rows: Iterable[Mapping[str, object]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(path.name + ".tmp")
    with temporary.open("w", encoding="utf-8", newline="\n") as handle:
        for row in rows:
            handle.write(
                json.dumps(
                    dict(row),
                    sort_keys=True,
                    separators=(",", ":"),
                    ensure_ascii=True,
                )
            )
            handle.write("\n")
    temporary.replace(path)


def _read_jsonl(path: Path) -> List[Dict[str, object]]:
    rows: List[Dict[str, object]] = []
    with path.open("r", encoding="utf-8") as handle:
        for line in handle:
            if line.strip():
                value = json.loads(line)
                if not isinstance(value, dict):
                    raise TypeError("JSONL row must be an object")
                rows.append(value)
    return rows


def _verify_file(path: Path, expected: str, label: str) -> Dict[str, object]:
    return f0._verify_hash(path, expected, label)


def _pipeline_source_path() -> Path:
    return (
        Path(sys.prefix)
        / "Lib"
        / "site-packages"
        / "diffusers"
        / "pipelines"
        / "controlnet"
        / "pipeline_controlnet_blip_diffusion.py"
    )


def _verify_manifest_rows(
    root: Path,
    manifest_path: Path,
    *,
    label: str,
) -> Dict[str, object]:
    manifest = _read_json(manifest_path)
    if not isinstance(manifest, dict) or not isinstance(
        manifest.get("files"), list
    ):
        raise TypeError("{} must contain a files list".format(label))
    checked = []
    for row in manifest["files"]:
        if not isinstance(row, dict):
            raise TypeError("{} contains a non-object row".format(label))
        relative_path = Path(str(row["path"]))
        path = (root / relative_path).resolve()
        if not f0._path_within(path, root):
            raise RuntimeError("{} path escaped its root".format(label))
        if not path.is_file():
            raise FileNotFoundError(
                "{} file is missing: {}".format(label, relative_path)
            )
        size_bytes = path.stat().st_size
        digest = _sha256(path)
        checks = {
            "size_bytes": size_bytes == int(row["size_bytes"]),
            "sha256": digest == str(row["sha256"]),
        }
        if not all(checks.values()):
            raise RuntimeError(
                "{} file differs: {} {}".format(
                    label,
                    relative_path,
                    checks,
                )
            )
        checked.append(
            {
                "path": relative_path.as_posix(),
                "size_bytes": size_bytes,
                "sha256": digest,
            }
        )
    return {
        "schema": "trkh_saspa_f1_manifest_row_check_v1",
        "manifest_path": str(manifest_path.resolve()),
        "manifest_sha256": _sha256(manifest_path),
        "checked_file_count": len(checked),
        "checked_files_sha256": _canonical_sha256(checked),
        "passed": True,
    }


def verify_model_snapshot(
    snapshot_manifest: Mapping[str, object],
) -> Dict[str, object]:
    snapshot_root = Path(str(snapshot_manifest["snapshot_path"])).resolve()
    rows = snapshot_manifest.get("files")
    if not isinstance(rows, list):
        raise TypeError("Model snapshot manifest must contain a files list")
    expected_manifest_sha = str(snapshot_manifest["file_manifest_sha256"])
    if _canonical_sha256(rows) != expected_manifest_sha:
        raise RuntimeError("Model snapshot manifest content hash differs")
    checked = []
    total_bytes = 0
    for row in rows:
        if not isinstance(row, dict):
            raise TypeError("Model snapshot contains a non-object row")
        relative_path = Path(str(row["path"]))
        path = (snapshot_root / relative_path).resolve()
        if not f0._path_within(path, snapshot_root):
            raise RuntimeError("Model snapshot path escaped its root")
        if not path.is_file():
            raise FileNotFoundError(
                "Model snapshot file is missing: {}".format(relative_path)
            )
        size_bytes = path.stat().st_size
        digest = _sha256(path)
        checks = {
            "size_bytes": size_bytes == int(row["size_bytes"]),
            "sha256": digest == str(row["sha256"]),
        }
        if not all(checks.values()):
            raise RuntimeError(
                "Model snapshot file differs: {} {}".format(
                    relative_path,
                    checks,
                )
            )
        total_bytes += size_bytes
        checked.append(
            {
                "path": relative_path.as_posix(),
                "size_bytes": size_bytes,
                "sha256": digest,
            }
        )
    if total_bytes != int(snapshot_manifest["total_bytes"]):
        raise RuntimeError("Model snapshot total byte count differs")
    if total_bytes != f0.LOCKED_MODEL_REMOTE_BYTES:
        raise RuntimeError("Model snapshot differs from the parent lock")
    return {
        "schema": "trkh_saspa_f1_model_snapshot_check_v1",
        "snapshot_path": str(snapshot_root),
        "file_count": len(checked),
        "total_bytes": total_bytes,
        "files_sha256": _canonical_sha256(checked),
        "expected_files_sha256": expected_manifest_sha,
        "passed": True,
    }


def verify_static_locks(
    protocol_path: Path,
    implementation_lock_path: Path,
    requirements_path: Path,
    f0_root: Path,
) -> Dict[str, object]:
    files = {
        "protocol": _verify_file(
            protocol_path,
            f0.LOCKED_PROTOCOL_SHA256,
            "parent protocol",
        ),
        "requirements": _verify_file(
            requirements_path,
            f0.LOCKED_REQUIREMENTS_SHA256,
            "SaSPA requirements",
        ),
        "implementation_json": _verify_file(
            implementation_lock_path,
            LOCKED_IMPLEMENTATION_JSON_SHA256,
            "F1 implementation lock",
        ),
        "implementation_md": _verify_file(
            IMPLEMENTATION_LOCK_MD_PATH,
            LOCKED_IMPLEMENTATION_MD_SHA256,
            "F1 implementation Markdown",
        ),
        "implementation_sha": _verify_file(
            IMPLEMENTATION_LOCK_SHA_PATH,
            LOCKED_IMPLEMENTATION_SHA_FILE_SHA256,
            "F1 implementation SHA lock",
        ),
        "f0_summary": _verify_file(
            f0_root / "summary.json",
            LOCKED_F0_SUMMARY_SHA256,
            "F0 summary",
        ),
        "f0_manifest": _verify_file(
            f0_root / "manifest.json",
            LOCKED_F0_MANIFEST_SHA256,
            "F0 manifest",
        ),
        "f0_replay": _verify_file(
            f0_root / "replay.json",
            LOCKED_F0_REPLAY_SHA256,
            "F0 replay",
        ),
        "f0_final_manifest": _verify_file(
            f0_root / "final_manifest.json",
            LOCKED_F0_FINAL_MANIFEST_SHA256,
            "F0 final manifest",
        ),
        "source_selection": _verify_file(
            f0_root / "source_selection.json",
            LOCKED_SOURCE_SELECTION_SHA256,
            "F0 source selection",
        ),
        "saspa_utils": _verify_file(
            f0.SASPA_ROOT / "all_utils" / "utils.py",
            LOCKED_SASPA_UTILS_SHA256,
            "SaSPA Canny source",
        ),
        "pipeline_source": _verify_file(
            _pipeline_source_path(),
            LOCKED_PIPELINE_SOURCE_SHA256,
            "pinned Diffusers pipeline source",
        ),
    }
    protocol = _read_json(protocol_path)
    implementation = _read_json(implementation_lock_path)
    summary = _read_json(f0_root / "summary.json")
    replay = _read_json(f0_root / "replay.json")
    selection = _read_json(f0_root / "source_selection.json")
    if not all(
        isinstance(value, dict)
        for value in (protocol, implementation, summary, replay, selection)
    ):
        raise TypeError("Locked F1 inputs must all be JSON objects")
    checks = {
        "protocol_revision": protocol["protocol_revision"] == 2,
        "implementation_state": implementation["lock_state"]
        == "prospective_before_any_f1_output",
        "f0_passed": bool(summary["passed"]),
        "f0_f1_authorized": bool(summary["f1_authorized"]),
        "f0_no_pipeline_invocation": not bool(summary["pipeline_invoked"]),
        "f0_no_synthetic_pixels": not bool(summary["synthetic_pixels_generated"]),
        "f0_replay_passed": bool(replay["passed"]),
        "selection_content": selection["selection_sha256"]
        == LOCKED_SELECTION_CONTENT_SHA256,
        "selection_rows": int(selection["row_count"]) == EXPECTED_OUTPUTS,
        "selection_train_only": bool(selection["train_only"]),
        "selection_no_validation_test": not bool(
            selection["validation_test_pixels_opened"]
        ),
    }
    if not all(checks.values()):
        raise RuntimeError("F1 static lock checks failed: {}".format(checks))
    parent_static = f0.verify_static_inputs(
        protocol_path,
        requirements_path,
    )
    f0_artifacts = _verify_manifest_rows(
        f0_root,
        f0_root / "final_manifest.json",
        label="F0 final manifest",
    )
    return {
        "schema": "trkh_saspa_f1_static_locks_v1",
        "files": files,
        "checks": checks,
        "parent_static_inputs": parent_static,
        "f0_artifacts": f0_artifacts,
        "passed": True,
    }


def _hwc3(array):
    import numpy as np

    value = np.asarray(array)
    if value.dtype != np.uint8:
        raise TypeError("Canny input must be uint8")
    if value.ndim == 2:
        value = value[:, :, None]
    if value.ndim != 3 or value.shape[2] not in (1, 3, 4):
        raise ValueError("Canny input must have 1, 3, or 4 channels")
    if value.shape[2] == 3:
        return value
    if value.shape[2] == 1:
        return np.concatenate([value, value, value], axis=2)
    color = value[:, :, :3].astype(np.float32)
    alpha = value[:, :, 3:4].astype(np.float32) / 255.0
    return (color * alpha + 255.0 * (1.0 - alpha)).clip(0, 255).astype(
        np.uint8
    )


def saspa_resize_image(array, smaller_side_resolution: int = 512):
    import cv2
    import numpy as np

    image = _hwc3(array)
    height, width, _ = image.shape
    scaled_height = float(height)
    scaled_width = float(width)
    scale = float(smaller_side_resolution) / min(scaled_height, scaled_width)
    scaled_height *= scale
    scaled_width *= scale
    if scaled_height * scaled_width > 1_200_000:
        scale = math.sqrt(1_200_000 / (scaled_height * scaled_width))
        scaled_height *= scale
        scaled_width *= scale
    output_height = int(np.round(scaled_height / 64.0)) * 64
    output_width = int(np.round(scaled_width / 64.0)) * 64
    interpolation = cv2.INTER_LANCZOS4 if scale > 1 else cv2.INTER_AREA
    return cv2.resize(
        image,
        (output_width, output_height),
        interpolation=interpolation,
    )


def make_locked_canny(array):
    import cv2
    import numpy as np

    resized = saspa_resize_image(array, smaller_side_resolution=512)
    canny = cv2.Canny(resized, 120, 200)
    return np.concatenate([canny[:, :, None]] * 3, axis=2)


def _decoded_rgb_sha256(array) -> str:
    import numpy as np

    value = np.ascontiguousarray(array, dtype=np.uint8)
    return hashlib.sha256(value.tobytes()).hexdigest()


def _bbox_crop(array, bbox_xywh: Sequence[float]):
    import numpy as np

    value = np.asarray(array)
    height, width = value.shape[:2]
    x_center, y_center, box_width, box_height = [
        float(item) for item in bbox_xywh
    ]
    left = max(0, int(math.floor((x_center - box_width / 2.0) * width)))
    top = max(0, int(math.floor((y_center - box_height / 2.0) * height)))
    right = min(width, int(math.ceil((x_center + box_width / 2.0) * width)))
    bottom = min(
        height,
        int(math.ceil((y_center + box_height / 2.0) * height)),
    )
    if right <= left or bottom <= top:
        raise ValueError("Normalized bbox produced an empty crop")
    return value[top:bottom, left:right].copy()


def _trim_subject_padding(array):
    import numpy as np

    value = _hwc3(array)
    height, width = value.shape[:2]
    corner_size = min(8, height, width)
    corners = np.concatenate(
        [
            value[:corner_size, :corner_size].reshape(-1, 3),
            value[:corner_size, -corner_size:].reshape(-1, 3),
            value[-corner_size:, :corner_size].reshape(-1, 3),
            value[-corner_size:, -corner_size:].reshape(-1, 3),
        ],
        axis=0,
    )
    fill = np.median(corners.astype(np.float32), axis=0)
    near_fill = (
        np.max(np.abs(value.astype(np.float32) - fill[None, None, :]), axis=2)
        <= 4.0
    )
    row_fill = near_fill.mean(axis=1) >= 0.98
    column_fill = near_fill.mean(axis=0) >= 0.98
    top = 0
    while top < height and bool(row_fill[top]):
        top += 1
    bottom = height
    while bottom > top and bool(row_fill[bottom - 1]):
        bottom -= 1
    left = 0
    while left < width and bool(column_fill[left]):
        left += 1
    right = width
    while right > left and bool(column_fill[right - 1]):
        right -= 1
    candidate = value[top:bottom, left:right]
    if (
        candidate.shape[0] < 32
        or candidate.shape[1] < 32
        or candidate.shape[0] * candidate.shape[1] < 0.25 * height * width
    ):
        return value.copy()
    return candidate.copy()


def _resize_metric(array, size: int = METRIC_SIZE):
    import cv2

    value = _hwc3(array)
    interpolation = (
        cv2.INTER_LANCZOS4
        if value.shape[0] < size or value.shape[1] < size
        else cv2.INTER_AREA
    )
    return cv2.resize(value, (size, size), interpolation=interpolation)


def _normalized_histogram(values, bins: int, value_range: Tuple[float, float]):
    import numpy as np

    histogram, _ = np.histogram(values, bins=bins, range=value_range)
    result = histogram.astype(np.float64)
    total = float(result.sum())
    if total <= 0:
        raise ValueError("Histogram has zero mass")
    return result / total


def _jensen_shannon_distance(first, second) -> float:
    import numpy as np

    p = np.asarray(first, dtype=np.float64)
    q = np.asarray(second, dtype=np.float64)
    midpoint = 0.5 * (p + q)
    epsilon = 1e-12
    divergence = 0.5 * float(
        np.sum(p * np.log((p + epsilon) / (midpoint + epsilon)))
        + np.sum(q * np.log((q + epsilon) / (midpoint + epsilon)))
    )
    return math.sqrt(max(0.0, divergence))


def _lab_histogram_js(first, second) -> float:
    import cv2
    import numpy as np

    def descriptor(image):
        lab = cv2.cvtColor(image, cv2.COLOR_RGB2LAB)
        channels = [
            _normalized_histogram(lab[:, :, index], 16, (0.0, 256.0))
            for index in range(3)
        ]
        return np.concatenate(channels) / 3.0

    return _jensen_shannon_distance(descriptor(first), descriptor(second))


def _hue_histogram(image):
    import cv2

    hsv = cv2.cvtColor(image, cv2.COLOR_RGB2HSV)
    mask = (hsv[:, :, 1] >= 32) & (hsv[:, :, 2] >= 32)
    if float(mask.mean()) < 0.05:
        values = hsv[:, :, 0].reshape(-1)
    else:
        values = hsv[:, :, 0][mask]
    return _normalized_histogram(values, 36, (0.0, 180.0))


def _circular_wasserstein(first, second) -> float:
    import numpy as np

    difference_cdf = np.cumsum(
        np.asarray(first, dtype=np.float64)
        - np.asarray(second, dtype=np.float64)
    )
    center = float(np.median(difference_cdf))
    return float(np.abs(difference_cdf - center).sum() / len(difference_cdf))


def _lbp_histogram(image):
    import cv2
    import numpy as np

    gray = cv2.cvtColor(image, cv2.COLOR_RGB2GRAY)
    center = gray[1:-1, 1:-1]
    neighbors = (
        gray[:-2, :-2],
        gray[:-2, 1:-1],
        gray[:-2, 2:],
        gray[1:-1, 2:],
        gray[2:, 2:],
        gray[2:, 1:-1],
        gray[2:, :-2],
        gray[1:-1, :-2],
    )
    codes = np.zeros_like(center, dtype=np.uint8)
    for bit, neighbor in enumerate(neighbors):
        codes |= ((neighbor >= center).astype(np.uint8) << bit)
    return _normalized_histogram(codes, 256, (0.0, 256.0))


def _chi_square_distance(first, second) -> float:
    import numpy as np

    p = np.asarray(first, dtype=np.float64)
    q = np.asarray(second, dtype=np.float64)
    return float(0.5 * np.sum((p - q) ** 2 / (p + q + 1e-12)))


def _edge_density(image) -> float:
    import cv2

    gray = cv2.cvtColor(image, cv2.COLOR_RGB2GRAY)
    return float((cv2.Canny(gray, 120, 200) > 0).mean())


def fidelity_statistics(first, second) -> Dict[str, float]:
    left = _resize_metric(first)
    right = _resize_metric(second)
    first_density = _edge_density(left)
    second_density = _edge_density(right)
    return {
        "lab_histogram_js": _lab_histogram_js(left, right),
        "hsv_hue_circular_w1": _circular_wasserstein(
            _hue_histogram(left),
            _hue_histogram(right),
        ),
        "lbp_texture_chi_square": _chi_square_distance(
            _lbp_histogram(left),
            _lbp_histogram(right),
        ),
        "edge_density_ratio": (
            max(first_density, second_density) + 1e-6
        )
        / (min(first_density, second_density) + 1e-6),
    }


def _load_rgb(path: Path):
    import numpy as np
    from PIL import Image

    with Image.open(path) as image:
        return np.asarray(image.convert("RGB"), dtype=np.uint8)


def _fidelity_rank(protocol_id: str, candidate: Mapping[str, object]) -> str:
    return f0._selection_hash(
        "fidelity",
        protocol_id,
        candidate["class_index"],
        candidate["leakage_group"],
        candidate["edge_relative_path"],
        candidate["subject_relative_path"],
    )


def select_fidelity_reference_pairs(
    protocol: Mapping[str, object],
    selected_rows: Sequence[Mapping[str, object]],
) -> Dict[str, object]:
    candidates, population = f0._eligible_candidates(protocol)
    excluded_groups = {
        str(row[field])
        for row in selected_rows
        for field in ("edge_leakage_group", "subject_leakage_group")
    }
    protocol_id = str(protocol["protocol_id"])
    selected: List[Dict[str, object]] = []
    counts: List[int] = []
    for class_index in range(5):
        used_groups: Set[str] = set()
        ranked = sorted(
            (
                candidate
                for candidate in candidates
                if int(candidate["class_index"]) == class_index
                and str(candidate["leakage_group"]) not in excluded_groups
            ),
            key=lambda candidate: _fidelity_rank(protocol_id, candidate),
        )
        class_rows = []
        for candidate in ranked:
            leakage_group = str(candidate["leakage_group"])
            if leakage_group in used_groups:
                continue
            used_groups.add(leakage_group)
            class_rows.append(
                {
                    "class_index": class_index,
                    "leakage_group": leakage_group,
                    "edge_path": str(Path(candidate["edge_path"]).resolve()),
                    "edge_relative_path": candidate["edge_relative_path"],
                    "subject_path": str(
                        Path(candidate["subject_path"]).resolve()
                    ),
                    "subject_relative_path": candidate[
                        "subject_relative_path"
                    ],
                    "bbox_xywh_normalized": candidate[
                        "bbox_xywh_normalized"
                    ],
                    "rank_sha256": _fidelity_rank(protocol_id, candidate),
                }
            )
            if len(class_rows) == REFERENCE_PAIRS_PER_CLASS:
                break
        if len(class_rows) != REFERENCE_PAIRS_PER_CLASS:
            raise RuntimeError(
                "Fidelity reference class {} has only {} pairs".format(
                    class_index, len(class_rows)
                )
            )
        counts.append(len(class_rows))
        selected.extend(class_rows)
    identity = [
        {
            key: row[key]
            for key in (
                "class_index",
                "leakage_group",
                "edge_relative_path",
                "subject_relative_path",
                "bbox_xywh_normalized",
                "rank_sha256",
            )
        }
        for row in selected
    ]
    return {
        "schema": "trkh_saspa_f1_fidelity_reference_selection_v1",
        "rows": selected,
        "counts": counts,
        "row_count": len(selected),
        "selection_sha256": _canonical_sha256(identity),
        "eligible_population": population,
        "excluded_groups": sorted(excluded_groups),
        "train_only": True,
        "validation_test_pixels_opened": False,
    }


def compute_fidelity_reference(
    selection: Mapping[str, object],
) -> Dict[str, object]:
    import numpy as np

    metrics = (
        "lab_histogram_js",
        "hsv_hue_circular_w1",
        "lbp_texture_chi_square",
        "edge_density_ratio",
    )
    rows: List[Dict[str, object]] = []
    values_by_class = {
        class_index: {metric: [] for metric in metrics}
        for class_index in range(5)
    }
    for row in selection["rows"]:  # type: ignore[index]
        edge = _load_rgb(Path(str(row["edge_path"])))
        subject = _load_rgb(Path(str(row["subject_path"])))
        edge_crop = _bbox_crop(edge, row["bbox_xywh_normalized"])
        subject_crop = _trim_subject_padding(subject)
        statistics = fidelity_statistics(edge_crop, subject_crop)
        class_index = int(row["class_index"])
        for metric in metrics:
            values_by_class[class_index][metric].append(statistics[metric])
        rows.append(
            {
                "class_index": class_index,
                "leakage_group": row["leakage_group"],
                "edge_relative_path": row["edge_relative_path"],
                "subject_relative_path": row["subject_relative_path"],
                "rank_sha256": row["rank_sha256"],
                "statistics": statistics,
            }
        )
    thresholds: Dict[str, Dict[str, float]] = {}
    summaries: Dict[str, Dict[str, Dict[str, float]]] = {}
    for class_index in range(5):
        class_key = str(class_index)
        thresholds[class_key] = {}
        summaries[class_key] = {}
        for metric in metrics:
            values = np.asarray(
                values_by_class[class_index][metric],
                dtype=np.float64,
            )
            thresholds[class_key][metric] = float(
                np.quantile(values, 0.95, method="higher")
            )
            summaries[class_key][metric] = {
                "minimum": float(values.min()),
                "median": float(np.median(values)),
                "mean": float(values.mean()),
                "maximum": float(values.max()),
            }
    return {
        "schema": "trkh_saspa_f1_fidelity_reference_v1",
        "selection_sha256": selection["selection_sha256"],
        "pair_count_per_class": REFERENCE_PAIRS_PER_CLASS,
        "threshold_quantile": 0.95,
        "quantile_method": "higher",
        "thresholds": thresholds,
        "summaries": summaries,
        "rows": rows,
        "rows_sha256": _canonical_sha256(rows),
        "train_only": True,
        "validation_test_pixels_opened": False,
        "passed": True,
    }


def _source_paths(
    protocol: Mapping[str, object],
    row: Mapping[str, object],
) -> Tuple[Path, Path, Path]:
    immutable = protocol["immutable_inputs"]  # type: ignore[index]
    yolo_root = Path(str(immutable["yolo_train_images"])).parents[1]
    class_root = Path(str(immutable["classification_train_root"])).parent
    edge_path = yolo_root / str(row["edge_relative_path"])
    label_path = yolo_root / str(row["edge_label_relative_path"])
    subject_path = class_root / str(row["subject_relative_path"])
    roots = (
        (edge_path, Path(str(immutable["yolo_train_images"]))),
        (label_path, Path(str(immutable["yolo_train_labels"]))),
        (subject_path, Path(str(immutable["classification_train_root"]))),
    )
    if not all(f0._path_within(path, root) for path, root in roots):
        raise RuntimeError("Selected F1 source escaped a locked train root")
    return edge_path, label_path, subject_path


def verify_selected_sources(
    protocol: Mapping[str, object],
    rows: Sequence[Mapping[str, object]],
) -> Dict[str, object]:
    verified = []
    for row in rows:
        edge_path, label_path, subject_path = _source_paths(protocol, row)
        checks = {
            "edge": _sha256(edge_path) == str(row["edge_file_sha256"]),
            "label": _sha256(label_path)
            == str(row["edge_label_file_sha256"]),
            "subject": _sha256(subject_path)
            == str(row["subject_file_sha256"]),
        }
        if not all(checks.values()):
            raise RuntimeError(
                "Selected source hash mismatch for {}".format(row["output_id"])
            )
        verified.append(
            {
                "output_id": row["output_id"],
                "edge_path": str(edge_path.resolve()),
                "label_path": str(label_path.resolve()),
                "subject_path": str(subject_path.resolve()),
                "checks": checks,
            }
        )
    return {
        "schema": "trkh_saspa_f1_selected_source_check_v1",
        "rows": verified,
        "rows_sha256": _canonical_sha256(verified),
        "passed": True,
        "raw_dataset_modified": False,
        "validation_test_pixels_opened": False,
    }


def _runtime_versions() -> Dict[str, str]:
    import importlib.metadata
    import platform

    return {
        "python": platform.python_version(),
        "torch": importlib.metadata.version("torch"),
        "torchvision": importlib.metadata.version("torchvision"),
        "diffusers": importlib.metadata.version("diffusers"),
        "transformers": importlib.metadata.version("transformers"),
        "accelerate": importlib.metadata.version("accelerate"),
    }


def _enable_offload(pipe: object, mode: str) -> None:
    if mode == "enable_model_cpu_offload":
        pipe.enable_model_cpu_offload()  # type: ignore[attr-defined]
    elif mode == "enable_sequential_cpu_offload":
        pipe.enable_sequential_cpu_offload()  # type: ignore[attr-defined]
    else:
        raise ValueError("Unsupported F1 offload mode: {}".format(mode))


def _install_final_decode_finiteness_hook(pipe: object, torch_module):
    decoder = pipe.vae.decoder  # type: ignore[attr-defined]
    state = {"detected": False}

    def finite_hook(unused_module, unused_inputs, output):
        tensors = []
        if torch_module.is_tensor(output):
            tensors.append(output)
        elif isinstance(output, (tuple, list)):
            tensors.extend(
                item for item in output if torch_module.is_tensor(item)
            )
        elif hasattr(output, "sample") and torch_module.is_tensor(
            output.sample
        ):
            tensors.append(output.sample)
        if any(
            not bool(torch_module.isfinite(tensor).all().item())
            for tensor in tensors
        ):
            state["detected"] = True

    handle = decoder.register_forward_hook(finite_hook)
    return state, handle, "pipeline.vae.decoder.register_forward_hook"


def _generation_resource_checks(
    limits: Mapping[str, object],
    *,
    generation_seconds: float,
    output_index: int,
    torch_peak_allocated_bytes: int,
    torch_peak_reserved_bytes: int,
    telemetry_summary: Mapping[str, object],
    nonfinite_detected: bool,
) -> Dict[str, object]:
    gib = float(1024 ** 3)
    nvml_peak_mib = telemetry_summary.get("max_nvml_used_mib")
    virtual_fraction = telemetry_summary.get("max_virtual_used_fraction")
    external_pids = telemetry_summary.get(
        "external_python_or_trtexec_pids", []
    )
    checks = {
        "torch_peak_allocated": torch_peak_allocated_bytes / gib
        <= float(limits["torch_peak_allocated_gib_max"]),
        "torch_peak_reserved": torch_peak_reserved_bytes / gib
        <= float(limits["torch_peak_reserved_gib_max"]),
        "nvml_total_used": nvml_peak_mib is not None
        and float(nvml_peak_mib) / 1024.0
        <= float(limits["nvml_total_used_gib_max"]),
        "host_virtual_memory_fraction": virtual_fraction is not None
        and float(virtual_fraction)
        < float(limits["host_virtual_memory_fraction_max"]),
        "nonfinite_tensor": not nonfinite_detected,
        "no_unknown_python_or_trtexec": not bool(external_pids),
    }
    if output_index == 0:
        checks["first_output_seconds"] = generation_seconds <= float(
            limits["first_output_seconds_max"]
        )
        checks["ten_output_estimate"] = (
            generation_seconds * EXPECTED_OUTPUTS / 60.0
            <= float(limits["ten_output_estimated_minutes_max"])
        )
    return {
        "limits": dict(limits),
        "checks": checks,
        "passed": all(checks.values()),
    }


def _preprocessing_record(
    row: Mapping[str, object],
    canny,
) -> Dict[str, object]:
    return {
        "algorithm": "locked_saspa_resize_hwc3_canny_v1",
        "source_code_sha256": LOCKED_SASPA_UTILS_SHA256,
        "edge_file_sha256": row["edge_file_sha256"],
        "smaller_side_resolution": 512,
        "maximum_resized_area_pixels": 1_200_000,
        "dimension_rounding": "nearest_multiple_of_64",
        "canny_low_threshold": 120,
        "canny_high_threshold": 200,
        "canny_shape": list(canny.shape),
        "canny_decoded_rgb_sha256": _decoded_rgb_sha256(canny),
        "pipeline_target_size": [512, 512],
    }


def _generation_worker(args: argparse.Namespace) -> Dict[str, object]:
    if args.worker_input is None or args.worker_result is None:
        raise ValueError("Generation worker requires input and result paths")
    started_unix = time.time()
    input_payload = _read_json(args.worker_input)
    if not isinstance(input_payload, dict):
        raise TypeError("Generation worker input must be a JSON object")
    attempt_dir = Path(str(input_payload["attempt_dir"])).resolve()
    attempt_dir.mkdir(parents=True, exist_ok=False)
    controls_dir = attempt_dir / "controls"
    outputs_dir = attempt_dir / "outputs"
    controls_dir.mkdir()
    outputs_dir.mkdir()
    result_path = Path(args.worker_result).resolve()
    saved_outputs: List[str] = []
    pipeline_invoked = False
    synthetic_pixels_generated = False
    hook_handle = None
    try:
        os.environ["HF_HUB_OFFLINE"] = "1"
        os.environ["TRANSFORMERS_OFFLINE"] = "1"
        os.environ["DIFFUSERS_OFFLINE"] = "1"
        os.environ["CUBLAS_WORKSPACE_CONFIG"] = ":4096:8"
        import numpy as np
        import torch
        from diffusers import BlipDiffusionControlNetPipeline
        from PIL import Image

        if not torch.cuda.is_available():
            raise RuntimeError("CUDA is unavailable in the F1 worker")
        torch.use_deterministic_algorithms(True)
        torch.backends.cuda.matmul.allow_tf32 = False
        torch.backends.cudnn.allow_tf32 = False
        torch.backends.cudnn.benchmark = False
        protocol = input_payload["protocol"]
        rows = input_payload["rows"]
        snapshot_path = str(input_payload["snapshot_path"])
        offload_mode = str(input_payload["offload_mode"])
        if len(rows) != EXPECTED_OUTPUTS:
            raise RuntimeError("Worker did not receive ten locked rows")
        pipe = BlipDiffusionControlNetPipeline.from_pretrained(
            snapshot_path,
            torch_dtype=torch.float16,
            local_files_only=True,
            low_cpu_mem_usage=True,
            use_safetensors=False,
        )
        _enable_offload(pipe, offload_mode)
        pipe.enable_attention_slicing("auto")
        vae_slicing_api = f0._enable_locked_vae_slicing(pipe)
        (
            nonfinite_state,
            hook_handle,
            finiteness_hook_api,
        ) = _install_final_decode_finiteness_hook(pipe, torch)
        provenance: List[Dict[str, object]] = []
        per_output_telemetry: List[Dict[str, object]] = []
        generation_limits = protocol["phase_f1_tiny_output"]["resource_gates"]
        allowed_process_ids = set(args.worker_allowed_pid)
        runtime_versions = _runtime_versions()
        inference = protocol["generator"]["inference"]
        immutable = protocol["immutable_inputs"]
        yolo_root = Path(str(immutable["yolo_train_images"])).parents[1]
        class_root = Path(str(immutable["classification_train_root"])).parent
        for output_index, row in enumerate(rows):
            edge_path = yolo_root / str(row["edge_relative_path"])
            subject_path = class_root / str(row["subject_relative_path"])
            edge = _load_rgb(edge_path)
            subject = _load_rgb(subject_path)
            canny = make_locked_canny(edge)
            preprocessing = _preprocessing_record(row, canny)
            preprocessing_sha256 = _canonical_sha256(preprocessing)
            canny_path = controls_dir / "{}_canny.png".format(row["output_id"])
            Image.fromarray(canny, mode="RGB").save(
                canny_path,
                format="PNG",
                optimize=False,
                compress_level=6,
            )
            torch.cuda.empty_cache()
            torch.cuda.reset_peak_memory_stats()
            nonfinite_state["detected"] = False
            generator = torch.Generator(device="cpu").manual_seed(
                int(row["seed"])
            )
            started = time.perf_counter()
            with f0.ResourceSampler(
                interval_seconds=0.5,
                allowed_process_ids=allowed_process_ids,
            ) as sampler:
                pipeline_invoked = True
                generated = pipe(
                    prompt=str(row["prompt"]),
                    reference_image=Image.fromarray(subject, mode="RGB"),
                    condtioning_image=Image.fromarray(canny, mode="RGB"),
                    source_subject_category=str(
                        inference["source_subject_category"]
                    ),
                    target_subject_category=str(
                        inference["target_subject_category"]
                    ),
                    guidance_scale=float(inference["guidance_scale"]),
                    height=int(inference["height"]),
                    width=int(inference["width"]),
                    num_inference_steps=int(inference["num_inference_steps"]),
                    generator=generator,
                    neg_prompt=str(inference["negative_prompt"]),
                    prompt_strength=1.0,
                    prompt_reps=20,
                    output_type="pil",
                    return_dict=True,
                ).images[0]
                synthetic_pixels_generated = True
            generation_seconds = time.perf_counter() - started
            torch_peak_allocated = int(torch.cuda.max_memory_allocated())
            torch_peak_reserved = int(torch.cuda.max_memory_reserved())
            telemetry = {
                "samples": sampler.samples,
                "summary": sampler.summary(),
            }
            checks = _generation_resource_checks(
                generation_limits,
                generation_seconds=generation_seconds,
                output_index=output_index,
                torch_peak_allocated_bytes=torch_peak_allocated,
                torch_peak_reserved_bytes=torch_peak_reserved,
                telemetry_summary=telemetry["summary"],
                nonfinite_detected=bool(nonfinite_state["detected"]),
            )
            per_output_telemetry.append(
                {
                    "output_id": row["output_id"],
                    "output_index": output_index,
                    "generation_seconds": generation_seconds,
                    "torch_peak_allocated_bytes": torch_peak_allocated,
                    "torch_peak_reserved_bytes": torch_peak_reserved,
                    "nonfinite_detected": bool(
                        nonfinite_state["detected"]
                    ),
                    "resource_gates": checks,
                    "telemetry": telemetry,
                }
            )
            if not checks["passed"]:
                raise RuntimeError(
                    "Generation resource gates failed for {}: {}".format(
                        row["output_id"], checks["checks"]
                    )
                )
            output = generated.convert("RGB")
            if output.size != (512, 512):
                raise RuntimeError("Generated output size differs from 512x512")
            output_array = np.asarray(output, dtype=np.uint8)
            output_path = outputs_dir / "{}.png".format(row["output_id"])
            output.save(
                output_path,
                format="PNG",
                optimize=False,
                compress_level=6,
            )
            saved_outputs.append(str(output_path.resolve()))
            nvml_peak = telemetry["summary"].get("max_nvml_used_mib")
            provenance_row = {
                "protocol_id": protocol["protocol_id"],
                "output_id": row["output_id"],
                "target_class_index": row["class_index"],
                "target_class_folder": row["class_folder"],
                "edge_relative_path": row["edge_relative_path"],
                "edge_leakage_group": row["edge_leakage_group"],
                "edge_file_sha256": row["edge_file_sha256"],
                "edge_label_file_sha256": row["edge_label_file_sha256"],
                "bbox_xywh_normalized": row["bbox_xywh_normalized"],
                "subject_relative_path": row["subject_relative_path"],
                "subject_leakage_group": row["subject_leakage_group"],
                "subject_file_sha256": row["subject_file_sha256"],
                "prompt_template_index": row["prompt_template_index"],
                "prompt_sha256": row["prompt_sha256"],
                "seed": row["seed"],
                "generator_repository": protocol["generator"][
                    "model_repository"
                ],
                "generator_revision": protocol["generator"]["model_revision"],
                "runtime_versions": runtime_versions,
                "offload_mode": offload_mode,
                "generation_seconds": generation_seconds,
                "torch_peak_allocated_bytes": torch_peak_allocated,
                "torch_peak_reserved_bytes": torch_peak_reserved,
                "nvml_peak_used_bytes": (
                    int(float(nvml_peak) * 1024 ** 2)
                    if nvml_peak is not None
                    else None
                ),
                "output_sha256": _sha256(output_path),
                "decoded_rgb_sha256": _decoded_rgb_sha256(output_array),
                "preprocessing_sha256": preprocessing_sha256,
                "license_provenance": {
                    "generator_model": "apache-2.0",
                    "saspa_code": "MIT",
                    "external_prior": True,
                },
                "output_relative_path": output_path.relative_to(
                    attempt_dir
                ).as_posix(),
                "canny_relative_path": canny_path.relative_to(
                    attempt_dir
                ).as_posix(),
                "preprocessing": preprocessing,
                "resource_gates": checks,
            }
            missing = [
                field for field in PROVENANCE_FIELDS if field not in provenance_row
            ]
            if missing:
                raise RuntimeError(
                    "Provenance fields missing for {}: {}".format(
                        row["output_id"], missing
                    )
                )
            provenance.append(provenance_row)
        hook_handle.remove()
        hook_handle = None
        _write_jsonl(attempt_dir / "provenance.jsonl", provenance)
        _write_json(
            attempt_dir / "resource_telemetry.json",
            {
                "schema": "trkh_saspa_f1_generation_telemetry_v1",
                "rows": per_output_telemetry,
            },
        )
        result = {
            "schema": WORKER_SCHEMA,
            "status": "Generated",
            "passed": True,
            "started_unix": started_unix,
            "finished_unix": time.time(),
            "offload_mode": offload_mode,
            "vae_slicing_api": vae_slicing_api,
            "finiteness_hook_api": finiteness_hook_api,
            "runtime_versions": runtime_versions,
            "saved_output_count": len(saved_outputs),
            "saved_outputs": saved_outputs,
            "provenance_sha256": _sha256(attempt_dir / "provenance.jsonl"),
            "pipeline_invoked": pipeline_invoked,
            "synthetic_pixels_generated": synthetic_pixels_generated,
            "native_full_gpu_placement": False,
        }
        _write_json(result_path, result)
        return result
    except Exception as exc:
        failure = {
            "schema": WORKER_SCHEMA,
            "status": "FailClosed",
            "passed": False,
            "started_unix": started_unix,
            "finished_unix": time.time(),
            "error_type": type(exc).__name__,
            "error": str(exc),
            "traceback": traceback.format_exc(),
            "saved_output_count": len(saved_outputs),
            "saved_outputs": saved_outputs,
            "pipeline_invoked": pipeline_invoked,
            "synthetic_pixels_generated": synthetic_pixels_generated,
            "native_full_gpu_placement": False,
        }
        _write_json(result_path, failure)
        raise
    finally:
        if hook_handle is not None:
            hook_handle.remove()


def _run_generation_attempt(
    *,
    protocol: Mapping[str, object],
    rows: Sequence[Mapping[str, object]],
    snapshot_path: Path,
    output_dir: Path,
    offload_mode: str,
    timeout_seconds: float,
) -> Dict[str, object]:
    attempt_name = "attempt_{}".format(offload_mode)
    attempt_dir = output_dir / attempt_name
    worker_input = output_dir / "{}_worker_input.json".format(attempt_name)
    worker_result = output_dir / "{}_worker_result.json".format(attempt_name)
    payload = {
        "schema": "trkh_saspa_f1_worker_input_v1",
        "protocol": dict(protocol),
        "rows": [dict(row) for row in rows],
        "snapshot_path": str(snapshot_path.resolve()),
        "attempt_dir": str(attempt_dir.resolve()),
        "offload_mode": offload_mode,
    }
    _write_json(worker_input, payload)
    allowed_parent_pids = sorted({os.getpid(), os.getppid()})
    command = [
        sys.executable,
        "-m",
        "trkh.tools.audit_saspa_synthetic_a0_f1",
        "--generation-worker",
        "--worker-input",
        str(worker_input),
        "--worker-result",
        str(worker_result),
    ]
    for process_id in allowed_parent_pids:
        command.extend(["--worker-allowed-pid", str(process_id)])
    environment = os.environ.copy()
    existing_python_path = environment.get("PYTHONPATH")
    environment["PYTHONPATH"] = str(REPO_ROOT)
    if existing_python_path:
        environment["PYTHONPATH"] += os.pathsep + existing_python_path
    environment["CUBLAS_WORKSPACE_CONFIG"] = ":4096:8"
    timed_out = False
    timeout_stdout = ""
    timeout_stderr = ""
    try:
        completed = subprocess.run(
            command,
            cwd=str(REPO_ROOT),
            env=environment,
            capture_output=True,
            text=True,
            encoding="utf-8",
            errors="replace",
            timeout=timeout_seconds,
            check=False,
        )
    except subprocess.TimeoutExpired as exc:
        completed = None
        timed_out = True
        timeout_stdout = (
            exc.stdout.decode("utf-8", errors="replace")
            if isinstance(exc.stdout, bytes)
            else str(exc.stdout or "")
        )
        timeout_stderr = (
            exc.stderr.decode("utf-8", errors="replace")
            if isinstance(exc.stderr, bytes)
            else str(exc.stderr or "")
        )
    result = _read_json(worker_result) if worker_result.is_file() else {}
    if not isinstance(result, dict):
        result = {}
    observed_outputs = sorted(
        (attempt_dir / "outputs").glob("*.png")
        if (attempt_dir / "outputs").is_dir()
        else ()
    )
    reported_count = int(result.get("saved_output_count", 0))
    observed_count = len(observed_outputs)
    result["reported_saved_output_count"] = reported_count
    result["observed_saved_output_count"] = observed_count
    result["saved_output_count"] = max(reported_count, observed_count)
    if observed_count:
        result["synthetic_pixels_generated"] = True
    result["worker_process"] = {
        "mode": "isolated_subprocess",
        "python_executable": sys.executable,
        "allowed_parent_pids": allowed_parent_pids,
        "return_code": completed.returncode if completed is not None else None,
        "timeout_seconds": timeout_seconds,
        "timed_out": timed_out,
    }
    if result.get("schema") != WORKER_SCHEMA:
        result["passed"] = False
        result["error"] = "Generation worker returned an invalid schema"
    if reported_count != observed_count:
        result["passed"] = False
        result["error"] = (
            "Worker report/output count mismatch: reported {}, observed {}"
        ).format(reported_count, observed_count)
    if (
        completed is None
        or completed.returncode != 0
        or not result.get("passed")
    ):
        result["passed"] = False
        result.setdefault("error", "Generation worker failed without result")
        stdout = completed.stdout if completed is not None else timeout_stdout
        stderr = completed.stderr if completed is not None else timeout_stderr
        result["stdout_tail"] = stdout[-4000:]
        result["stderr_tail"] = stderr[-4000:]
    _write_json(worker_result, result)
    return result


def _evaluate_generated_fidelity(
    protocol: Mapping[str, object],
    selected_rows: Sequence[Mapping[str, object]],
    provenance: Sequence[Mapping[str, object]],
    attempt_dir: Path,
    reference: Mapping[str, object],
) -> Dict[str, object]:
    provenance_by_id = {
        str(row["output_id"]): row for row in provenance
    }
    thresholds = reference["thresholds"]  # type: ignore[index]
    results = []
    for selected in selected_rows:
        output_id = str(selected["output_id"])
        row = provenance_by_id[output_id]
        output_path = attempt_dir / str(row["output_relative_path"])
        _, _, subject_path = _source_paths(protocol, selected)
        output = _load_rgb(output_path)
        subject = _trim_subject_padding(_load_rgb(subject_path))
        generated_crop = _bbox_crop(
            output,
            selected["bbox_xywh_normalized"],
        )
        statistics = fidelity_statistics(generated_crop, subject)
        class_thresholds = thresholds[str(selected["class_index"])]
        checks = {
            metric: float(value) <= float(class_thresholds[metric])
            for metric, value in statistics.items()
        }
        results.append(
            {
                "output_id": output_id,
                "class_index": selected["class_index"],
                "statistics": statistics,
                "thresholds": class_thresholds,
                "checks": checks,
                "passed": all(checks.values()),
            }
        )
    return {
        "schema": "trkh_saspa_f1_generated_fidelity_v1",
        "reference_rows_sha256": reference["rows_sha256"],
        "rows": results,
        "row_count": len(results),
        "passed_count": sum(bool(row["passed"]) for row in results),
        "passed": all(bool(row["passed"]) for row in results),
        "validation_test_pixels_opened": False,
    }


def _fit_image(image, size: Tuple[int, int], background=(245, 247, 249)):
    from PIL import Image

    value = image.convert("RGB")
    value.thumbnail(size, Image.Resampling.LANCZOS)
    canvas = Image.new("RGB", size, background)
    left = (size[0] - value.width) // 2
    top = (size[1] - value.height) // 2
    canvas.paste(value, (left, top))
    return canvas


def _font(size: int, bold: bool = False):
    from PIL import ImageFont

    paths = [
        Path(r"C:\Windows\Fonts\arialbd.ttf" if bold else r"C:\Windows\Fonts\arial.ttf"),
        Path(r"C:\Windows\Fonts\segoeuib.ttf" if bold else r"C:\Windows\Fonts\segoeui.ttf"),
    ]
    for path in paths:
        if path.is_file():
            return ImageFont.truetype(str(path), size=size)
    return ImageFont.load_default()


def _locked_blind_mapping(
    protocol: Mapping[str, object],
    selected_rows: Sequence[Mapping[str, object]],
    provenance: Sequence[Mapping[str, object]],
) -> List[Dict[str, object]]:
    provenance_by_id = {
        str(row["output_id"]): row for row in provenance
    }
    blind_ranked = sorted(
        selected_rows,
        key=lambda row: f0._selection_hash(
            "blind",
            protocol["protocol_id"],
            row["output_id"],
        ),
    )
    return [
        {
            "blind_id": chr(ord("A") + position),
            "output_id": selected["output_id"],
            "output_sha256": provenance_by_id[str(selected["output_id"])][
                "output_sha256"
            ],
            "blind_rank_sha256": f0._selection_hash(
                "blind",
                protocol["protocol_id"],
                selected["output_id"],
            ),
        }
        for position, selected in enumerate(blind_ranked)
    ]


def _build_contact_sheets(
    protocol: Mapping[str, object],
    selected_rows: Sequence[Mapping[str, object]],
    provenance: Sequence[Mapping[str, object]],
    attempt_dir: Path,
    output_dir: Path,
) -> Dict[str, object]:
    from PIL import Image, ImageDraw

    provenance_by_id = {
        str(row["output_id"]): row for row in provenance
    }
    tile_width, tile_height = 220, 220
    header_height = 52
    columns = 5
    paired = Image.new(
        "RGB",
        (
            columns * tile_width,
            len(selected_rows) * (tile_height + header_height),
        ),
        "white",
    )
    draw = ImageDraw.Draw(paired)
    title_font = _font(18, bold=True)
    label_font = _font(15, bold=True)
    labels = ("Edge", "Canny", "Subject", "Output", "Output bbox")
    for row_index, selected in enumerate(selected_rows):
        output_id = str(selected["output_id"])
        provenance_row = provenance_by_id[output_id]
        edge_path, _, subject_path = _source_paths(protocol, selected)
        output_path = attempt_dir / str(
            provenance_row["output_relative_path"]
        )
        canny_path = attempt_dir / str(provenance_row["canny_relative_path"])
        edge = Image.open(edge_path).convert("RGB")
        canny = Image.open(canny_path).convert("RGB")
        subject = Image.open(subject_path).convert("RGB")
        output = Image.open(output_path).convert("RGB")
        output_bbox = Image.fromarray(
            _bbox_crop(
                _load_rgb(output_path),
                selected["bbox_xywh_normalized"],
            ),
            mode="RGB",
        )
        top = row_index * (tile_height + header_height)
        draw.text(
            (8, top + 4),
            "{} | class {} | rank {}".format(
                output_id,
                selected["class_index"],
                selected["edge_rank"],
            ),
            fill=(25, 45, 70),
            font=title_font,
        )
        for column, (label, image) in enumerate(
            zip(labels, (edge, canny, subject, output, output_bbox))
        ):
            left = column * tile_width
            draw.text(
                (left + 8, top + 29),
                label,
                fill=(70, 80, 90),
                font=label_font,
            )
            paired.paste(
                _fit_image(image, (tile_width, tile_height)),
                (left, top + header_height),
            )
    paired_path = output_dir / "paired_contact_sheet.png"
    paired.save(paired_path, format="PNG", optimize=False, compress_level=6)

    blind_mapping = _locked_blind_mapping(
        protocol,
        selected_rows,
        provenance,
    )
    blind_tile = 300
    blind_header = 42
    blind_columns = 2
    blind_rows = math.ceil(len(blind_mapping) / blind_columns)
    blind = Image.new(
        "RGB",
        (
            blind_columns * blind_tile,
            blind_rows * (blind_tile + blind_header),
        ),
        "white",
    )
    blind_draw = ImageDraw.Draw(blind)
    blind_font = _font(22, bold=True)
    selected_by_id = {
        str(row["output_id"]): row for row in selected_rows
    }
    for position, mapping_row in enumerate(blind_mapping):
        blind_id = str(mapping_row["blind_id"])
        selected = selected_by_id[str(mapping_row["output_id"])]
        provenance_row = provenance_by_id[str(mapping_row["output_id"])]
        output_path = attempt_dir / str(
            provenance_row["output_relative_path"]
        )
        output = Image.open(output_path).convert("RGB")
        column = position % blind_columns
        row_index = position // blind_columns
        left = column * blind_tile
        top = row_index * (blind_tile + blind_header)
        blind_draw.text(
            (left + 10, top + 8),
            "Review {}".format(blind_id),
            fill=(25, 45, 70),
            font=blind_font,
        )
        blind.paste(
            _fit_image(output, (blind_tile, blind_tile)),
            (left, top + blind_header),
        )
    blind_path = output_dir / "blind_review_contact_sheet.png"
    blind.save(blind_path, format="PNG", optimize=False, compress_level=6)
    commitment = [
        {
            "blind_id": row["blind_id"],
            "output_sha256": row["output_sha256"],
            "blind_rank_sha256": row["blind_rank_sha256"],
        }
        for row in blind_mapping
    ]
    _write_json(
        output_dir / "blind_review_commitment.json",
        {
            "schema": "trkh_saspa_f1_blind_commitment_v1",
            "rows": commitment,
            "mapping_sha256": _canonical_sha256(blind_mapping),
            "target_labels_exposed": False,
        },
    )
    return {
        "paired_contact_sheet": str(paired_path.resolve()),
        "paired_contact_sheet_sha256": _sha256(paired_path),
        "blind_contact_sheet": str(blind_path.resolve()),
        "blind_contact_sheet_sha256": _sha256(blind_path),
        "blind_commitment_sha256": _sha256(
            output_dir / "blind_review_commitment.json"
        ),
        "blind_mapping_content_sha256": _canonical_sha256(blind_mapping),
        "blind_mapping_artifact_written": False,
        "passed": True,
    }


def _artifact_manifest(root: Path, filename: str) -> Dict[str, object]:
    rows = []
    for path in sorted(root.rglob("*"), key=lambda item: item.as_posix().lower()):
        if not path.is_file() or path.name == filename:
            continue
        rows.append(
            {
                "path": path.relative_to(root).as_posix(),
                "size_bytes": path.stat().st_size,
                "sha256": _sha256(path),
            }
        )
    return {
        "schema": "trkh_saspa_f1_artifact_manifest_v1",
        "files": rows,
        "files_sha256": _canonical_sha256(rows),
    }


def run_preflight(args: argparse.Namespace) -> Dict[str, object]:
    static = verify_static_locks(
        args.protocol,
        args.implementation_lock,
        args.requirements,
        args.f0_root,
    )
    protocol = _read_json(args.protocol)
    selection = _read_json(args.f0_root / "source_selection.json")
    if not isinstance(protocol, dict) or not isinstance(selection, dict):
        raise TypeError("F1 preflight inputs must be JSON objects")
    source_checks = verify_selected_sources(protocol, selection["rows"])
    reference_selection = select_fidelity_reference_pairs(
        protocol,
        selection["rows"],
    )
    result = {
        "schema": "trkh_saspa_f1_preflight_v1",
        "static_locks": static,
        "selected_sources": source_checks,
        "fidelity_reference_selection": {
            key: reference_selection[key]
            for key in (
                "counts",
                "row_count",
                "selection_sha256",
                "excluded_groups",
                "train_only",
                "validation_test_pixels_opened",
            )
        },
        "synthetic_pixels_generated": False,
        "validation_test_pixels_opened": False,
        "passed": True,
    }
    print(json.dumps(result, indent=2, sort_keys=True))
    return result


def run_formal_generation(args: argparse.Namespace) -> Dict[str, object]:
    output_dir = Path(args.output_dir).resolve()
    if output_dir.exists():
        raise FileExistsError(
            "F1 output already exists; never overwrite: {}".format(output_dir)
        )
    output_dir.mkdir(parents=True)
    started_unix = time.time()
    failure_context: Dict[str, object] = {"stage": "repo_state"}
    try:
        repo = f0._repo_state()
        failure_context["repo_state"] = repo
        if not repo["passed"]:
            raise RuntimeError("Formal F1 requires clean pushed state")
        failure_context["stage"] = "static_locks"
        static = verify_static_locks(
            args.protocol,
            args.implementation_lock,
            args.requirements,
            args.f0_root,
        )
        protocol = _read_json(args.protocol)
        selection = _read_json(args.f0_root / "source_selection.json")
        snapshot = _read_json(args.f0_root / "model_snapshot_manifest.json")
        if not all(
            isinstance(value, dict)
            for value in (protocol, selection, snapshot)
        ):
            raise TypeError("Formal F1 locked inputs must be JSON objects")
        rows = sorted(
            selection["rows"],
            key=lambda row: (int(row["class_index"]), int(row["edge_rank"])),
        )
        expected_order = [
            (class_index, edge_rank)
            for class_index in range(5)
            for edge_rank in range(2)
        ]
        actual_order = [
            (int(row["class_index"]), int(row["edge_rank"])) for row in rows
        ]
        if actual_order != expected_order:
            raise RuntimeError("F1 source rows differ from locked order")
        failure_context["stage"] = "start_resource_gate"
        start_resources = f0.resource_snapshot()
        start_gates = f0.evaluate_f0_resource_gates(
            protocol,
            start_resources,
            model_snapshot_bytes=int(snapshot["total_bytes"]),
        )
        failure_context["start_resources"] = start_resources
        failure_context["start_gates"] = start_gates
        if not start_gates["passed"]:
            raise RuntimeError(
                "F1 start resource gates failed: {}".format(
                    start_gates["checks"]
                )
            )
        failure_context["stage"] = "model_snapshot_verify"
        snapshot_check = verify_model_snapshot(snapshot)
        failure_context["stage"] = "selected_source_check"
        source_checks_before = verify_selected_sources(protocol, rows)
        _write_json(output_dir / "static_locks.json", static)
        _write_json(
            output_dir / "model_snapshot_recheck.json",
            snapshot_check,
        )
        _write_json(
            output_dir / "selected_sources_before.json",
            source_checks_before,
        )
        shutil.copy2(
            args.protocol,
            output_dir / "locked_protocol.json",
        )
        shutil.copy2(
            args.implementation_lock,
            output_dir / "implementation_lock.json",
        )
        failure_context["stage"] = "fidelity_reference_selection"
        reference_selection = select_fidelity_reference_pairs(protocol, rows)
        _write_json(
            output_dir / "fidelity_reference_selection.json",
            reference_selection,
        )
        failure_context["stage"] = "fidelity_reference_compute"
        reference = compute_fidelity_reference(reference_selection)
        _write_json(output_dir / "fidelity_reference.json", reference)
        failure_context["fidelity_reference"] = {
            "selection_sha256": reference_selection["selection_sha256"],
            "rows_sha256": reference["rows_sha256"],
            "thresholds": reference["thresholds"],
        }
        failure_context["stage"] = "pre_generation_resource_gate"
        pre_generation_resources = f0.resource_snapshot()
        pre_generation_gates = f0.evaluate_f0_resource_gates(
            protocol,
            pre_generation_resources,
            model_snapshot_bytes=int(snapshot["total_bytes"]),
        )
        failure_context["pre_generation_resources"] = pre_generation_resources
        failure_context["pre_generation_gates"] = pre_generation_gates
        if not pre_generation_gates["passed"]:
            raise RuntimeError(
                "F1 pre-generation gates failed: {}".format(
                    pre_generation_gates["checks"]
                )
            )
        failure_context["stage"] = "generation_primary"
        worker_timeout = (
            float(
                protocol["phase_f1_tiny_output"]["resource_gates"][
                    "ten_output_estimated_minutes_max"
                ]
            )
            * 60.0
            + 300.0
        )
        primary = _run_generation_attempt(
            protocol=protocol,
            rows=rows,
            snapshot_path=Path(str(snapshot["snapshot_path"])),
            output_dir=output_dir,
            offload_mode="enable_model_cpu_offload",
            timeout_seconds=worker_timeout,
        )
        attempts = [primary]
        selected_attempt = primary
        if not primary.get("passed"):
            if int(primary.get("saved_output_count", 0)) != 0:
                raise RuntimeError(
                    "Primary generation failed after saving output; retry forbidden"
                )
            failure_context["stage"] = "generation_sequential_fallback"
            sequential = _run_generation_attempt(
                protocol=protocol,
                rows=rows,
                snapshot_path=Path(str(snapshot["snapshot_path"])),
                output_dir=output_dir,
                offload_mode="enable_sequential_cpu_offload",
                timeout_seconds=worker_timeout,
            )
            attempts.append(sequential)
            selected_attempt = sequential
        failure_context["generation_attempts"] = attempts
        if not selected_attempt.get("passed"):
            raise RuntimeError(
                "Both authorized F1 generation modes failed: {}".format(
                    selected_attempt.get("error", "unknown error")
                )
            )
        if int(selected_attempt["saved_output_count"]) != EXPECTED_OUTPUTS:
            raise RuntimeError("F1 generation did not save exactly ten outputs")
        attempt_dir = (
            output_dir
            / "attempt_{}".format(selected_attempt["offload_mode"])
        )
        provenance = _read_jsonl(attempt_dir / "provenance.jsonl")
        if len(provenance) != EXPECTED_OUTPUTS:
            raise RuntimeError("F1 provenance does not contain ten rows")
        failure_context["stage"] = "generated_fidelity"
        generated_fidelity = _evaluate_generated_fidelity(
            protocol,
            rows,
            provenance,
            attempt_dir,
            reference,
        )
        _write_json(
            output_dir / "generated_fidelity.json",
            generated_fidelity,
        )
        failure_context["stage"] = "contact_sheets"
        contact_sheets = _build_contact_sheets(
            protocol,
            rows,
            provenance,
            attempt_dir,
            output_dir,
        )
        _write_json(output_dir / "contact_sheets.json", contact_sheets)
        failure_context["stage"] = "raw_source_recheck"
        source_checks_after = verify_selected_sources(protocol, rows)
        _write_json(
            output_dir / "selected_sources_after.json",
            source_checks_after,
        )
        if (
            source_checks_before["rows_sha256"]
            != source_checks_after["rows_sha256"]
        ):
            raise RuntimeError("Selected source identity changed during F1")
        failure_context["stage"] = "immutable_input_recheck"
        static_after = verify_static_locks(
            args.protocol,
            args.implementation_lock,
            args.requirements,
            args.f0_root,
        )
        _write_json(output_dir / "static_locks_after.json", static_after)
        if (
            static["parent_static_inputs"]["files"]
            != static_after["parent_static_inputs"]["files"]
        ):
            raise RuntimeError("Immutable static input identity changed during F1")
        summary = {
            "schema": SCHEMA,
            "method": METHOD,
            "status": "GeneratedPendingBlindDuplicateAndXAI",
            "passed_generation": True,
            "passed_train_only_fidelity": generated_fidelity["passed"],
            "f1_final_passed": False,
            "started_unix": started_unix,
            "finished_unix": time.time(),
            "repo_state": repo,
            "static_locks_passed": static["passed"],
            "static_locks_after_passed": static_after["passed"],
            "model_snapshot_check": {
                "file_count": snapshot_check["file_count"],
                "total_bytes": snapshot_check["total_bytes"],
                "files_sha256": snapshot_check["files_sha256"],
            },
            "source_selection_sha256": selection["selection_sha256"],
            "fidelity_reference_selection_sha256": reference_selection[
                "selection_sha256"
            ],
            "fidelity_reference_rows_sha256": reference["rows_sha256"],
            "generation_attempts": attempts,
            "selected_offload_mode": selected_attempt["offload_mode"],
            "saved_output_count": selected_attempt["saved_output_count"],
            "provenance_sha256": selected_attempt["provenance_sha256"],
            "pipeline_invoked": selected_attempt["pipeline_invoked"],
            "synthetic_pixels_generated": selected_attempt[
                "synthetic_pixels_generated"
            ],
            "generated_fidelity": {
                "passed": generated_fidelity["passed"],
                "passed_count": generated_fidelity["passed_count"],
                "row_count": generated_fidelity["row_count"],
            },
            "contact_sheets": contact_sheets,
            "raw_dataset_modified": False,
            "generation_validation_test_pixels_opened": False,
            "generation_validation_test_labels_opened": False,
            "manual_blind_review_pending": True,
            "near_duplicate_audit_pending": True,
            "xai_diagnostic_pending": True,
            "a1_authorized": False,
            "training_authorized": False,
            "full_train_authorized": False,
            "current_best_command_updated": False,
        }
        _write_json(output_dir / "generation_summary.json", summary)
        manifest = _artifact_manifest(output_dir, "generation_manifest.json")
        _write_json(output_dir / "generation_manifest.json", manifest)
        result = {
            "summary_path": str(
                (output_dir / "generation_summary.json").resolve()
            ),
            "summary_sha256": _sha256(
                output_dir / "generation_summary.json"
            ),
            "manifest_path": str(
                (output_dir / "generation_manifest.json").resolve()
            ),
            "manifest_sha256": _sha256(
                output_dir / "generation_manifest.json"
            ),
            "saved_output_count": selected_attempt["saved_output_count"],
            "passed_generation": True,
            "passed_train_only_fidelity": generated_fidelity["passed"],
            "f1_final_passed": False,
        }
        print(json.dumps(result, indent=2, sort_keys=True))
        return result
    except Exception as exc:
        failure = {
            "schema": "trkh_saspa_f1_generation_failure_v1",
            "method": METHOD,
            "status": "FailClosed",
            "passed": False,
            "started_unix": started_unix,
            "finished_unix": time.time(),
            "error_type": type(exc).__name__,
            "error": str(exc),
            "traceback": traceback.format_exc(),
            "failure_context": failure_context,
            "a1_authorized": False,
            "training_authorized": False,
            "full_train_authorized": False,
            "current_best_command_updated": False,
        }
        _write_json(output_dir / "failure.json", failure)
        raise


def replay_generation(summary_path: Path) -> Dict[str, object]:
    summary_path = Path(summary_path).resolve()
    root = summary_path.parent
    summary = _read_json(summary_path)
    manifest = _read_json(root / "generation_manifest.json")
    if not isinstance(summary, dict) or not isinstance(manifest, dict):
        raise TypeError("F1 replay inputs must be JSON objects")
    failures: List[str] = []
    try:
        manifest_check = _verify_manifest_rows(
            root,
            root / "generation_manifest.json",
            label="F1 generation manifest",
        )
    except Exception as exc:
        manifest_check = {"passed": False, "error": str(exc)}
        failures.append("generation_manifest_rows")
    manifest_paths = {
        str(row["path"])
        for row in manifest.get("files", [])
        if isinstance(row, dict) and "path" in row
    }
    ignored_unmanifested = {
        "generation_manifest.json",
        "generation_replay.json",
        "generation_final_manifest.json",
    }
    actual_paths = {
        path.relative_to(root).as_posix()
        for path in root.rglob("*")
        if path.is_file()
    }
    unexpected = sorted(
        actual_paths - manifest_paths - ignored_unmanifested
    )
    if unexpected:
        failures.append("unmanifested_files:{}".format(unexpected))
    static = verify_static_locks(
        PROTOCOL_PATH,
        IMPLEMENTATION_LOCK_PATH,
        REQUIREMENTS_PATH,
        F0_ROOT,
    )
    protocol = _read_json(PROTOCOL_PATH)
    selection = _read_json(F0_ROOT / "source_selection.json")
    if not isinstance(protocol, dict) or not isinstance(selection, dict):
        raise TypeError("F1 replay locks must be JSON objects")
    selected_rows = sorted(
        selection["rows"],
        key=lambda row: (int(row["class_index"]), int(row["edge_rank"])),
    )
    source_checks = verify_selected_sources(protocol, selected_rows)
    selected_mode = str(summary.get("selected_offload_mode", ""))
    attempt_dir = root / "attempt_{}".format(selected_mode)
    provenance_path = attempt_dir / "provenance.jsonl"
    if not provenance_path.is_file():
        failures.append("provenance_missing")
        provenance = []
    else:
        provenance = _read_jsonl(provenance_path)
    if len(provenance) != EXPECTED_OUTPUTS:
        failures.append("provenance_row_count")
    expected_ids = [str(row["output_id"]) for row in selected_rows]
    actual_ids = [str(row.get("output_id")) for row in provenance]
    if actual_ids != expected_ids:
        failures.append("provenance_order")
    telemetry_path = attempt_dir / "resource_telemetry.json"
    telemetry_payload = (
        _read_json(telemetry_path) if telemetry_path.is_file() else {}
    )
    if not isinstance(telemetry_payload, dict):
        telemetry_payload = {}
    telemetry_rows = telemetry_payload.get("rows", [])
    telemetry_by_id = {
        str(row.get("output_id")): row
        for row in telemetry_rows
        if isinstance(row, dict)
    }
    provenance_by_id = {
        str(row.get("output_id")): row for row in provenance
    }
    from PIL import Image

    for output_index, selected in enumerate(selected_rows):
        output_id = str(selected["output_id"])
        row = provenance_by_id.get(output_id)
        telemetry_row = telemetry_by_id.get(output_id)
        if row is None or telemetry_row is None:
            failures.append("missing_output_record:{}".format(output_id))
            continue
        missing_fields = [
            field for field in PROVENANCE_FIELDS if field not in row
        ]
        if missing_fields:
            failures.append(
                "provenance_fields:{}:{}".format(
                    output_id,
                    missing_fields,
                )
            )
        field_pairs = (
            ("target_class_index", "class_index"),
            ("target_class_folder", "class_folder"),
            ("edge_relative_path", "edge_relative_path"),
            ("edge_leakage_group", "edge_leakage_group"),
            ("edge_file_sha256", "edge_file_sha256"),
            ("edge_label_file_sha256", "edge_label_file_sha256"),
            ("bbox_xywh_normalized", "bbox_xywh_normalized"),
            ("subject_relative_path", "subject_relative_path"),
            ("subject_leakage_group", "subject_leakage_group"),
            ("subject_file_sha256", "subject_file_sha256"),
            ("prompt_template_index", "prompt_template_index"),
            ("prompt_sha256", "prompt_sha256"),
            ("seed", "seed"),
        )
        if any(
            row.get(provenance_key) != selected.get(selection_key)
            for provenance_key, selection_key in field_pairs
        ):
            failures.append("provenance_identity:{}".format(output_id))
        if row.get("protocol_id") != protocol["protocol_id"]:
            failures.append("provenance_protocol:{}".format(output_id))
        output_path = (attempt_dir / str(row["output_relative_path"])).resolve()
        canny_path = (attempt_dir / str(row["canny_relative_path"])).resolve()
        if not f0._path_within(output_path, attempt_dir) or not f0._path_within(
            canny_path,
            attempt_dir,
        ):
            failures.append("artifact_path_escape:{}".format(output_id))
            continue
        if not output_path.is_file() or not canny_path.is_file():
            failures.append("artifact_missing:{}".format(output_id))
            continue
        with Image.open(output_path) as image:
            if image.mode != "RGB" or image.size != (512, 512):
                failures.append("output_encoding:{}".format(output_id))
            output_array = image.convert("RGB")
            output_decoded = _decoded_rgb_sha256(output_array)
        if _sha256(output_path) != str(row["output_sha256"]):
            failures.append("output_sha256:{}".format(output_id))
        if output_decoded != str(row["decoded_rgb_sha256"]):
            failures.append("output_decoded_sha256:{}".format(output_id))
        preprocessing = row.get("preprocessing")
        if (
            not isinstance(preprocessing, dict)
            or _canonical_sha256(preprocessing)
            != str(row["preprocessing_sha256"])
        ):
            failures.append("preprocessing_sha256:{}".format(output_id))
        else:
            canny = _load_rgb(canny_path)
            if (
                _decoded_rgb_sha256(canny)
                != preprocessing["canny_decoded_rgb_sha256"]
            ):
                failures.append("canny_decoded_sha256:{}".format(output_id))
        telemetry = telemetry_row.get("telemetry")
        if not isinstance(telemetry, dict) or not isinstance(
            telemetry.get("summary"), dict
        ):
            failures.append("telemetry_summary:{}".format(output_id))
            continue
        recomputed_gates = _generation_resource_checks(
            protocol["phase_f1_tiny_output"]["resource_gates"],
            generation_seconds=float(telemetry_row["generation_seconds"]),
            output_index=output_index,
            torch_peak_allocated_bytes=int(
                telemetry_row["torch_peak_allocated_bytes"]
            ),
            torch_peak_reserved_bytes=int(
                telemetry_row["torch_peak_reserved_bytes"]
            ),
            telemetry_summary=telemetry["summary"],
            nonfinite_detected=bool(
                telemetry_row["nonfinite_detected"]
            ),
        )
        if _canonical_sha256(recomputed_gates) != _canonical_sha256(
            telemetry_row.get("resource_gates")
        ):
            failures.append("telemetry_gate_arithmetic:{}".format(output_id))
        if _canonical_sha256(recomputed_gates) != _canonical_sha256(
            row.get("resource_gates")
        ):
            failures.append("provenance_gate_arithmetic:{}".format(output_id))
        if not recomputed_gates["passed"]:
            failures.append("resource_gate_failed:{}".format(output_id))
    if actual_ids == expected_ids:
        reference = _read_json(root / "fidelity_reference.json")
        recorded_fidelity = _read_json(root / "generated_fidelity.json")
        if not isinstance(reference, dict) or not isinstance(
            recorded_fidelity,
            dict,
        ):
            failures.append("fidelity_artifact_type")
        else:
            recomputed_fidelity = _evaluate_generated_fidelity(
                protocol,
                selected_rows,
                provenance,
                attempt_dir,
                reference,
            )
            if _canonical_sha256(
                recomputed_fidelity
            ) != _canonical_sha256(recorded_fidelity):
                failures.append("fidelity_recompute")
    commitment = _read_json(root / "blind_review_commitment.json")
    expected_mapping = _locked_blind_mapping(
        protocol,
        selected_rows,
        provenance,
    ) if actual_ids == expected_ids else []
    expected_commitment = [
        {
            "blind_id": row["blind_id"],
            "output_sha256": row["output_sha256"],
            "blind_rank_sha256": row["blind_rank_sha256"],
        }
        for row in expected_mapping
    ]
    if not isinstance(commitment, dict):
        failures.append("blind_commitment_type")
    else:
        if commitment.get("rows") != expected_commitment:
            failures.append("blind_commitment_rows")
        if commitment.get("mapping_sha256") != _canonical_sha256(
            expected_mapping
        ):
            failures.append("blind_mapping_commitment")
        if commitment.get("target_labels_exposed"):
            failures.append("blind_target_exposure")
    if (root / "blind_review_mapping.sealed.json").exists():
        failures.append("premature_blind_mapping_artifact")
    contact_sheets = _read_json(root / "contact_sheets.json")
    if not isinstance(contact_sheets, dict):
        failures.append("contact_sheet_type")
    else:
        sheet_pairs = (
            ("paired_contact_sheet.png", "paired_contact_sheet_sha256"),
            ("blind_review_contact_sheet.png", "blind_contact_sheet_sha256"),
            ("blind_review_commitment.json", "blind_commitment_sha256"),
        )
        for filename, key in sheet_pairs:
            path = root / filename
            if not path.is_file() or _sha256(path) != contact_sheets.get(key):
                failures.append("contact_sheet_hash:{}".format(filename))
        if contact_sheets.get("blind_mapping_artifact_written"):
            failures.append("blind_mapping_flag")
    if summary.get("saved_output_count") != EXPECTED_OUTPUTS:
        failures.append("summary_output_count")
    if summary.get("schema") != SCHEMA or not summary.get(
        "passed_generation"
    ):
        failures.append("summary_status")
    if not summary.get("pipeline_invoked") or not summary.get(
        "synthetic_pixels_generated"
    ):
        failures.append("summary_generation_flags")
    if summary.get("generation_validation_test_pixels_opened"):
        failures.append("generation_split_boundary")
    if summary.get("a1_authorized") or summary.get("training_authorized"):
        failures.append("premature_authorization")
    result = {
        "schema": "trkh_saspa_f1_generation_replay_v1",
        "summary_sha256": _sha256(summary_path),
        "manifest_sha256": _sha256(root / "generation_manifest.json"),
        "manifest_check": manifest_check,
        "static_locks_passed": static["passed"],
        "selected_sources_passed": source_checks["passed"],
        "provenance_row_count": len(provenance),
        "failures": failures,
        "passed": not failures,
        "synthetic_pixels_regenerated": False,
        "validation_test_pixels_opened": False,
        "a1_authorized": False,
    }
    _write_json(root / "generation_replay.json", result)
    final_manifest = _artifact_manifest(
        root,
        "generation_final_manifest.json",
    )
    _write_json(root / "generation_final_manifest.json", final_manifest)
    printed = dict(result)
    printed["final_manifest_sha256"] = _sha256(
        root / "generation_final_manifest.json"
    )
    print(json.dumps(printed, indent=2, sort_keys=True))
    if failures:
        raise RuntimeError("F1 generation replay failed: {}".format(failures))
    return result


def main(argv: Optional[Sequence[str]] = None) -> None:
    args = parse_args(argv)
    if args.generation_worker:
        _generation_worker(args)
    elif args.preflight_only:
        run_preflight(args)
    elif args.formal_generate:
        run_formal_generation(args)
    else:
        replay_generation(args.replay_summary)


if __name__ == "__main__":
    main()
