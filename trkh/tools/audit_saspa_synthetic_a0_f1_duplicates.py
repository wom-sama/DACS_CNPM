from __future__ import annotations

"""Isolated exact, pHash, and DINOv2 duplicate audit for SaSPA F1."""

import argparse
import json
import os
import time
import traceback
from pathlib import Path
from typing import Dict, List, Mapping, Optional, Sequence, Tuple

from trkh.tools import audit_saspa_synthetic_a0_f0 as f0
from trkh.tools import audit_saspa_synthetic_a0_f1 as f1


SCHEMA = "trkh_saspa_f1_duplicate_audit_v1"
REPO_ROOT = Path(__file__).resolve().parents[2]
LOCK_PATH = (
    REPO_ROOT
    / "docs"
    / "TRKH_5CLASS_SASPA_F1_DUPLICATE_AUDIT_LOCK_20260724.json"
)
LOCKED_LOCK_SHA256 = (
    "f0210c413049261320626c54613f8e1a39580ec8b80193fe27c2c52a49cb2619"
)
GENERATION_ROOT = f1.DEFAULT_OUTPUT_DIR
DEFAULT_AUDIT_DIR = GENERATION_ROOT / "duplicate_audit"
DEFAULT_MODEL_MANIFEST = DEFAULT_AUDIT_DIR / "dinov2_snapshot_manifest.json"
DEFAULT_REPORT = DEFAULT_AUDIT_DIR / "duplicate_report.json"
DEFAULT_CACHE_DIR = Path(r"D:\DataAI\Tools\hf_cache\trkh_saspa_dinov2")
MODEL_REPOSITORY = "facebook/dinov2-small"
MODEL_REVISION = "ed25f3a31f01632728cabb09d1542f84ab7b0056"
MODEL_SAFETENSORS_BYTES = 88_249_960
IMAGE_EXTENSIONS = {".bmp", ".jpeg", ".jpg", ".png", ".webp"}
PHASH_REJECT_MAX = 4
DINO_REJECT_MIN = 0.995
BATCH_SIZE = 32


def parse_args(argv: Optional[Sequence[str]] = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Locked SaSPA F1 cross-split duplicate audit."
    )
    parser.add_argument("--lock", type=Path, default=LOCK_PATH)
    parser.add_argument(
        "--generation-root",
        type=Path,
        default=GENERATION_ROOT,
    )
    parser.add_argument("--audit-dir", type=Path, default=DEFAULT_AUDIT_DIR)
    parser.add_argument("--cache-dir", type=Path, default=DEFAULT_CACHE_DIR)
    parser.add_argument(
        "--model-manifest",
        type=Path,
        default=DEFAULT_MODEL_MANIFEST,
    )
    mode = parser.add_mutually_exclusive_group(required=True)
    mode.add_argument("--prepare-model", action="store_true")
    mode.add_argument("--preflight-only", action="store_true")
    mode.add_argument("--formal-audit", action="store_true")
    mode.add_argument("--replay-report", type=Path)
    return parser.parse_args(argv)


def _read_json(path: Path) -> object:
    return f0._read_json(path)


def _write_json(path: Path, value: object) -> None:
    f0._write_json(path, value)


def _sha256(path: Path) -> str:
    return f0._sha256(path)


def _canonical_sha256(value: object) -> str:
    return f0._canonical_sha256(value)


def _locked_artifact_checks(
    lock: Mapping[str, object],
    generation_root: Path,
) -> Dict[str, object]:
    rows = []
    for filename, expected_sha in lock[
        "locked_generation_artifacts"
    ].items():
        path = generation_root / str(filename)
        if not path.is_file():
            raise FileNotFoundError(
                "Locked generation artifact is missing: {}".format(path)
            )
        digest = _sha256(path)
        if digest != str(expected_sha):
            raise RuntimeError(
                "Locked generation artifact differs: {}".format(filename)
            )
        rows.append(
            {
                "path": str(filename),
                "size_bytes": path.stat().st_size,
                "sha256": digest,
            }
        )
    return {
        "rows": rows,
        "rows_sha256": _canonical_sha256(rows),
        "passed": True,
    }


def _enumerate_reference_paths(
    lock: Mapping[str, object],
) -> Tuple[List[Dict[str, object]], List[Dict[str, object]]]:
    references: List[Dict[str, object]] = []
    scopes = []
    for scope_row in lock["reference_scopes"]:
        scope = str(scope_row["scope"])
        root = Path(str(scope_row["root"])).resolve()
        if not root.is_dir():
            raise FileNotFoundError(
                "Duplicate reference root is missing: {}".format(root)
            )
        paths = sorted(
            (
                path.resolve()
                for path in root.rglob("*")
                if path.is_file()
                and path.suffix.lower() in IMAGE_EXTENSIONS
            ),
            key=lambda path: str(path).lower(),
        )
        expected_count = int(scope_row["locked_file_count"])
        if len(paths) != expected_count:
            raise RuntimeError(
                "{} file count differs: {} != {}".format(
                    scope,
                    len(paths),
                    expected_count,
                )
            )
        scopes.append(
            {
                "scope": scope,
                "root": str(root),
                "file_count": len(paths),
            }
        )
        references.extend(
            {
                "scope": scope,
                "root": root,
                "path": path,
                "relative_path": path.relative_to(root).as_posix(),
            }
            for path in paths
        )
    references.sort(key=lambda row: str(row["path"]).lower())
    if len(references) != int(
        lock["enumeration"]["total_reference_files"]
    ):
        raise RuntimeError("Total duplicate reference count differs")
    return references, scopes


def verify_static_inputs(
    lock_path: Path,
    generation_root: Path,
) -> Dict[str, object]:
    if _sha256(lock_path) != LOCKED_LOCK_SHA256:
        raise RuntimeError("Duplicate audit lock SHA256 differs")
    lock = _read_json(lock_path)
    if not isinstance(lock, dict):
        raise TypeError("Duplicate audit lock must be a JSON object")
    if (
        lock["state"]
        != "prospective_before_validation_or_test_pixels_are_opened"
    ):
        raise RuntimeError("Duplicate audit lock state differs")
    if Path(str(lock["generation_root"])).as_posix() != (
        Path("runs")
        / "audit_saspa_dual_view_synthetic_a0_20260723"
        / "f1_tiny_output_v2"
    ).as_posix():
        raise RuntimeError("Duplicate audit generation root differs")
    artifact_checks = _locked_artifact_checks(lock, generation_root)
    references, scopes = _enumerate_reference_paths(lock)
    provenance_paths = list(
        generation_root.glob("attempt_*/provenance.jsonl")
    )
    if len(provenance_paths) != 1:
        raise RuntimeError("Expected exactly one F1 provenance JSONL")
    provenance = f1._read_jsonl(provenance_paths[0])
    if len(provenance) != f1.EXPECTED_OUTPUTS:
        raise RuntimeError("Expected ten generated outputs")
    outputs = []
    for row in provenance:
        path = (
            provenance_paths[0].parent
            / str(row["output_relative_path"])
        ).resolve()
        if not f0._path_within(path, provenance_paths[0].parent):
            raise RuntimeError("Generated output path escaped its attempt")
        if not path.is_file() or _sha256(path) != row["output_sha256"]:
            raise RuntimeError(
                "Generated output differs: {}".format(row["output_id"])
            )
        outputs.append(
            {
                "output_id": row["output_id"],
                "path": str(path),
                "sha256": row["output_sha256"],
            }
        )
    return {
        "schema": "trkh_saspa_f1_duplicate_static_v1",
        "lock_sha256": LOCKED_LOCK_SHA256,
        "generation_artifacts": artifact_checks,
        "reference_scopes": scopes,
        "reference_file_count": len(references),
        "output_rows": outputs,
        "output_rows_sha256": _canonical_sha256(outputs),
        "validation_test_pixels_opened": False,
        "validation_test_labels_opened": False,
        "passed": True,
    }


def _decoded_rgb_descriptor(image) -> Dict[str, object]:
    import hashlib
    import numpy as np

    value = np.ascontiguousarray(image, dtype=np.uint8)
    height, width = value.shape[:2]
    return {
        "height": int(height),
        "width": int(width),
        "decoded_rgb_sha256": hashlib.sha256(value.tobytes()).hexdigest(),
    }


def _phash_bits(image):
    import cv2
    import numpy as np
    from PIL import Image

    gray = Image.fromarray(image).convert("L").resize(
        (32, 32),
        Image.Resampling.LANCZOS,
    )
    values = np.asarray(gray, dtype=np.float32)
    coefficients = cv2.dct(values)[:8, :8]
    median = float(np.median(coefficients))
    return (coefficients > median).reshape(-1)


def _phash_hamming(first, second) -> int:
    import numpy as np

    return int(
        np.count_nonzero(
            np.asarray(first, dtype=np.bool_)
            != np.asarray(second, dtype=np.bool_)
        )
    )


def _load_rgb_array(path: Path):
    import numpy as np
    from PIL import Image

    with Image.open(path) as image:
        return np.asarray(image.convert("RGB"), dtype=np.uint8)


def _snapshot_files(snapshot_root: Path) -> List[Dict[str, object]]:
    rows = []
    model_file_count = 0
    for path in sorted(
        (
            path
            for path in snapshot_root.rglob("*")
            if path.is_file()
            and path.suffix.lower() in {".json", ".safetensors"}
        ),
        key=lambda path: path.relative_to(snapshot_root).as_posix().lower(),
    ):
        relative = path.relative_to(snapshot_root).as_posix()
        size_bytes = path.stat().st_size
        if relative == "model.safetensors":
            model_file_count += 1
            if size_bytes != MODEL_SAFETENSORS_BYTES:
                raise RuntimeError("DINOv2 safetensors byte count differs")
        rows.append(
            {
                "path": relative,
                "size_bytes": size_bytes,
                "sha256": _sha256(path),
            }
        )
    if model_file_count != 1:
        raise RuntimeError("Expected one locked DINOv2 model.safetensors")
    return rows


def prepare_model(args: argparse.Namespace) -> Dict[str, object]:
    if args.model_manifest.exists():
        raise FileExistsError(
            "DINO snapshot manifest already exists: {}".format(
                args.model_manifest
            )
        )
    if _sha256(args.lock) != LOCKED_LOCK_SHA256:
        raise RuntimeError("Duplicate audit lock SHA256 differs")
    from huggingface_hub import HfApi, snapshot_download

    info = HfApi().model_info(
        repo_id=MODEL_REPOSITORY,
        revision=MODEL_REVISION,
        files_metadata=True,
    )
    if info.sha != MODEL_REVISION:
        raise RuntimeError("DINOv2 resolved revision differs")
    snapshot_root = Path(
        snapshot_download(
            repo_id=MODEL_REPOSITORY,
            revision=MODEL_REVISION,
            cache_dir=str(args.cache_dir.resolve()),
            allow_patterns=("*.json", "*.safetensors"),
            max_workers=2,
        )
    ).resolve()
    rows = _snapshot_files(snapshot_root)
    result = {
        "schema": "trkh_saspa_f1_dinov2_snapshot_v1",
        "repository": MODEL_REPOSITORY,
        "requested_revision": MODEL_REVISION,
        "resolved_revision": info.sha,
        "snapshot_path": str(snapshot_root),
        "files": rows,
        "files_sha256": _canonical_sha256(rows),
        "safetensors_bytes": MODEL_SAFETENSORS_BYTES,
        "raw_dataset_pixels_opened": False,
        "validation_test_pixels_opened": False,
        "passed": True,
    }
    _write_json(args.model_manifest, result)
    print(json.dumps(result, indent=2, sort_keys=True))
    return result


def verify_model_manifest(path: Path) -> Dict[str, object]:
    manifest = _read_json(path)
    if not isinstance(manifest, dict):
        raise TypeError("DINO snapshot manifest must be a JSON object")
    if (
        manifest.get("repository") != MODEL_REPOSITORY
        or manifest.get("resolved_revision") != MODEL_REVISION
        or int(manifest.get("safetensors_bytes", 0))
        != MODEL_SAFETENSORS_BYTES
    ):
        raise RuntimeError("DINO snapshot identity differs")
    snapshot_root = Path(str(manifest["snapshot_path"])).resolve()
    expected_rows = manifest.get("files")
    if not isinstance(expected_rows, list):
        raise TypeError("DINO snapshot files must be a list")
    actual_rows = _snapshot_files(snapshot_root)
    if actual_rows != expected_rows:
        raise RuntimeError("DINO snapshot files differ")
    if _canonical_sha256(actual_rows) != manifest["files_sha256"]:
        raise RuntimeError("DINO snapshot manifest content differs")
    return {
        "manifest_path": str(path.resolve()),
        "manifest_sha256": _sha256(path),
        "snapshot_path": str(snapshot_root),
        "files_sha256": manifest["files_sha256"],
        "file_count": len(actual_rows),
        "passed": True,
    }


def _load_synthetic_rows(generation_root: Path) -> List[Dict[str, object]]:
    provenance_paths = list(
        generation_root.glob("attempt_*/provenance.jsonl")
    )
    if len(provenance_paths) != 1:
        raise RuntimeError("Expected exactly one F1 provenance JSONL")
    attempt_dir = provenance_paths[0].parent
    provenance = f1._read_jsonl(provenance_paths[0])
    rows = []
    for row in provenance:
        path = (
            attempt_dir / str(row["output_relative_path"])
        ).resolve()
        array = _load_rgb_array(path)
        descriptor = _decoded_rgb_descriptor(array)
        if descriptor["decoded_rgb_sha256"] != row["decoded_rgb_sha256"]:
            raise RuntimeError(
                "Synthetic decoded hash differs: {}".format(row["output_id"])
            )
        rows.append(
            {
                "output_id": str(row["output_id"]),
                "path": path,
                "output_sha256": str(row["output_sha256"]),
                "array": array,
                "decoded": descriptor,
                "phash": _phash_bits(array),
            }
        )
    if len(rows) != f1.EXPECTED_OUTPUTS:
        raise RuntimeError("Expected ten synthetic rows")
    return rows


def _dino_embeddings(model, processor, arrays, torch_module):
    import numpy as np
    from PIL import Image

    images = [Image.fromarray(array) for array in arrays]
    inputs = processor(images=images, return_tensors="pt")
    model_inputs = {
        key: value.to(device="cuda", non_blocking=False)
        for key, value in inputs.items()
    }
    with torch_module.inference_mode():
        outputs = model(**model_inputs)
        embeddings = outputs.last_hidden_state[:, 0, :].float()
        embeddings = torch_module.nn.functional.normalize(
            embeddings,
            p=2,
            dim=1,
        )
    return np.asarray(embeddings.cpu(), dtype=np.float32)


def _reference_record(
    reference: Mapping[str, object],
    descriptor: Mapping[str, object],
    *,
    distance_key: str,
    distance_value: object,
) -> Dict[str, object]:
    path = Path(str(reference["path"]))
    return {
        "scope": reference["scope"],
        "relative_path": reference["relative_path"],
        "file_sha256": _sha256(path),
        "decoded_rgb_sha256": descriptor["decoded_rgb_sha256"],
        "height": descriptor["height"],
        "width": descriptor["width"],
        distance_key: distance_value,
    }


def _finalize_nearest(
    state: Optional[Mapping[str, object]],
    *,
    distance_key: str,
) -> Optional[Dict[str, object]]:
    if state is None:
        return None
    return _reference_record(
        state["reference"],  # type: ignore[arg-type]
        state["descriptor"],  # type: ignore[arg-type]
        distance_key=distance_key,
        distance_value=state["value"],
    )


def _synthetic_pairs(
    synthetic_rows: Sequence[Mapping[str, object]],
    embeddings,
) -> List[Dict[str, object]]:
    import numpy as np

    pairs = []
    for left_index in range(len(synthetic_rows)):
        for right_index in range(left_index + 1, len(synthetic_rows)):
            left = synthetic_rows[left_index]
            right = synthetic_rows[right_index]
            exact_match = (
                left["decoded"]["height"] == right["decoded"]["height"]
                and left["decoded"]["width"] == right["decoded"]["width"]
                and left["decoded"]["decoded_rgb_sha256"]
                == right["decoded"]["decoded_rgb_sha256"]
            )
            hamming = _phash_hamming(left["phash"], right["phash"])
            cosine = float(
                np.dot(embeddings[left_index], embeddings[right_index])
            )
            checks = {
                "no_exact_match": not exact_match,
                "phash_hamming_gt_4": hamming > PHASH_REJECT_MAX,
                "dinov2_cosine_lt_0_995": cosine < DINO_REJECT_MIN,
            }
            pairs.append(
                {
                    "left_output_id": left["output_id"],
                    "right_output_id": right["output_id"],
                    "left_output_sha256": left["output_sha256"],
                    "right_output_sha256": right["output_sha256"],
                    "exact_match": exact_match,
                    "phash_hamming": hamming,
                    "dinov2_cosine": cosine,
                    "checks": checks,
                    "passed": all(checks.values()),
                }
            )
    if len(pairs) != 45:
        raise RuntimeError("Expected 45 unordered synthetic pairs")
    return pairs


def _artifact_manifest(root: Path, filename: str) -> Dict[str, object]:
    rows = []
    for path in sorted(
        root.rglob("*"),
        key=lambda item: item.as_posix().lower(),
    ):
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
        "schema": "trkh_saspa_f1_duplicate_manifest_v1",
        "files": rows,
        "files_sha256": _canonical_sha256(rows),
    }


def run_preflight(args: argparse.Namespace) -> Dict[str, object]:
    static = verify_static_inputs(args.lock, args.generation_root)
    model = (
        verify_model_manifest(args.model_manifest)
        if args.model_manifest.is_file()
        else {
            "passed": False,
            "prepared": False,
            "raw_dataset_pixels_opened": False,
        }
    )
    result = {
        "schema": "trkh_saspa_f1_duplicate_preflight_v1",
        "static": static,
        "model": model,
        "raw_dataset_pixels_opened": False,
        "validation_test_pixels_opened": False,
        "validation_test_labels_opened": False,
        "passed": static["passed"],
    }
    print(json.dumps(result, indent=2, sort_keys=True))
    return result


def run_formal_audit(args: argparse.Namespace) -> Dict[str, object]:
    report_path = args.audit_dir / "duplicate_report.json"
    failure_path = args.audit_dir / "duplicate_failure.json"
    if report_path.exists() or failure_path.exists():
        raise FileExistsError("Duplicate audit output already exists")
    args.audit_dir.mkdir(parents=True, exist_ok=True)
    started_unix = time.time()
    failure_context: Dict[str, object] = {"stage": "repo_state"}
    try:
        repo = f0._repo_state()
        failure_context["repo_state"] = repo
        if not repo["passed"]:
            raise RuntimeError("Formal duplicate audit requires clean pushed state")
        failure_context["stage"] = "static_inputs"
        static = verify_static_inputs(args.lock, args.generation_root)
        failure_context["stage"] = "model_manifest"
        model_check = verify_model_manifest(args.model_manifest)
        failure_context["stage"] = "resource_gate"
        resources = f0.resource_snapshot(
            allowed_process_ids={os.getppid()}
        )
        if resources["external_python_or_trtexec"]:
            raise RuntimeError(
                "Unknown Python/TensorRT process exists before duplicate audit"
            )
        os.environ["HF_HUB_OFFLINE"] = "1"
        os.environ["TRANSFORMERS_OFFLINE"] = "1"
        os.environ["CUBLAS_WORKSPACE_CONFIG"] = ":4096:8"
        import numpy as np
        import torch
        from transformers import AutoImageProcessor, AutoModel

        if not torch.cuda.is_available():
            raise RuntimeError("CUDA is required for the locked DINOv2 audit")
        torch.use_deterministic_algorithms(True)
        torch.backends.cuda.matmul.allow_tf32 = False
        torch.backends.cudnn.allow_tf32 = False
        torch.backends.cudnn.benchmark = False
        manifest = _read_json(args.model_manifest)
        if not isinstance(manifest, dict):
            raise TypeError("DINO snapshot manifest must be an object")
        snapshot_path = str(manifest["snapshot_path"])
        failure_context["stage"] = "model_load"
        processor = AutoImageProcessor.from_pretrained(
            snapshot_path,
            local_files_only=True,
        )
        model = AutoModel.from_pretrained(
            snapshot_path,
            local_files_only=True,
            use_safetensors=True,
        ).to(device="cuda", dtype=torch.float32)
        model.eval()
        failure_context["stage"] = "synthetic_embeddings"
        synthetic_rows = _load_synthetic_rows(args.generation_root)
        synthetic_embeddings = _dino_embeddings(
            model,
            processor,
            [row["array"] for row in synthetic_rows],
            torch,
        )
        if synthetic_embeddings.shape[0] != f1.EXPECTED_OUTPUTS:
            raise RuntimeError("DINOv2 synthetic embedding count differs")
        lock = _read_json(args.lock)
        if not isinstance(lock, dict):
            raise TypeError("Duplicate audit lock must be an object")
        references, unused_scopes = _enumerate_reference_paths(lock)
        states = [
            {"exact": None, "phash": None, "dino": None}
            for unused_row in synthetic_rows
        ]
        failure_context["stage"] = "reference_stream"
        for batch_start in range(0, len(references), BATCH_SIZE):
            batch = references[batch_start : batch_start + BATCH_SIZE]
            arrays = []
            descriptors = []
            for reference in batch:
                array = _load_rgb_array(Path(str(reference["path"])))
                descriptor = _decoded_rgb_descriptor(array)
                phash = _phash_bits(array)
                arrays.append(array)
                descriptors.append(descriptor)
                for output_index, synthetic in enumerate(synthetic_rows):
                    state = states[output_index]
                    exact = (
                        descriptor["height"]
                        == synthetic["decoded"]["height"]
                        and descriptor["width"]
                        == synthetic["decoded"]["width"]
                        and descriptor["decoded_rgb_sha256"]
                        == synthetic["decoded"]["decoded_rgb_sha256"]
                    )
                    if exact and state["exact"] is None:
                        state["exact"] = {
                            "reference": reference,
                            "descriptor": descriptor,
                            "value": True,
                        }
                    hamming = _phash_hamming(
                        synthetic["phash"],
                        phash,
                    )
                    if (
                        state["phash"] is None
                        or hamming < int(state["phash"]["value"])
                    ):
                        state["phash"] = {
                            "reference": reference,
                            "descriptor": descriptor,
                            "value": hamming,
                        }
            reference_embeddings = _dino_embeddings(
                model,
                processor,
                arrays,
                torch,
            )
            similarities = synthetic_embeddings @ reference_embeddings.T
            for output_index in range(len(synthetic_rows)):
                best_batch_index = int(
                    np.argmax(similarities[output_index])
                )
                cosine = float(
                    similarities[output_index, best_batch_index]
                )
                state = states[output_index]
                if (
                    state["dino"] is None
                    or cosine > float(state["dino"]["value"])
                ):
                    state["dino"] = {
                        "reference": batch[best_batch_index],
                        "descriptor": descriptors[best_batch_index],
                        "value": cosine,
                    }
            if batch_start % (BATCH_SIZE * 25) == 0:
                print(
                    "Duplicate audit processed {}/{} references".format(
                        min(batch_start + len(batch), len(references)),
                        len(references),
                    ),
                    flush=True,
                )
        failure_context["stage"] = "report_build"
        output_rows = []
        for synthetic, state in zip(synthetic_rows, states):
            exact_match = _finalize_nearest(
                state["exact"],
                distance_key="exact_match",
            )
            phash_nearest = _finalize_nearest(
                state["phash"],
                distance_key="phash_hamming",
            )
            dino_nearest = _finalize_nearest(
                state["dino"],
                distance_key="dinov2_cosine",
            )
            if phash_nearest is None or dino_nearest is None:
                raise RuntimeError("Duplicate nearest reference is missing")
            checks = {
                "no_exact_match": exact_match is None,
                "phash_hamming_gt_4": int(
                    phash_nearest["phash_hamming"]
                )
                > PHASH_REJECT_MAX,
                "dinov2_cosine_lt_0_995": float(
                    dino_nearest["dinov2_cosine"]
                )
                < DINO_REJECT_MIN,
            }
            output_rows.append(
                {
                    "output_id": synthetic["output_id"],
                    "output_sha256": synthetic["output_sha256"],
                    "decoded_rgb_sha256": synthetic["decoded"][
                        "decoded_rgb_sha256"
                    ],
                    "exact_match": exact_match,
                    "phash_nearest": phash_nearest,
                    "dinov2_nearest": dino_nearest,
                    "checks": checks,
                    "passed": all(checks.values()),
                }
            )
        synthetic_pairs = _synthetic_pairs(
            synthetic_rows,
            synthetic_embeddings,
        )
        passed = all(row["passed"] for row in output_rows) and all(
            row["passed"] for row in synthetic_pairs
        )
        report = {
            "schema": SCHEMA,
            "status": "Pass" if passed else "Reject",
            "passed": passed,
            "started_unix": started_unix,
            "finished_unix": time.time(),
            "repo_state": repo,
            "lock_sha256": LOCKED_LOCK_SHA256,
            "static_inputs_sha256": _canonical_sha256(static),
            "model_manifest": model_check,
            "runtime_versions": f1._runtime_versions(),
            "reference_file_count": len(references),
            "output_rows": output_rows,
            "synthetic_pairs": synthetic_pairs,
            "thresholds": {
                "exact_decoded_rgb_match_reject": True,
                "phash_hamming_reject_max": PHASH_REJECT_MAX,
                "dinov2_cosine_reject_min": DINO_REJECT_MIN,
            },
            "raw_dataset_modified": False,
            "validation_test_pixels_opened": True,
            "validation_test_labels_opened": False,
            "reference_embeddings_persisted": False,
            "reference_images_persisted": False,
            "class_aggregates_persisted": False,
            "split_aggregates_persisted": False,
            "regeneration_authorized": False,
            "a1_authorized": False,
            "training_authorized": False,
            "current_best_command_updated": False,
        }
        _write_json(report_path, report)
        manifest_payload = _artifact_manifest(
            args.audit_dir,
            "duplicate_manifest.json",
        )
        _write_json(
            args.audit_dir / "duplicate_manifest.json",
            manifest_payload,
        )
        result = {
            "report_path": str(report_path.resolve()),
            "report_sha256": _sha256(report_path),
            "manifest_sha256": _sha256(
                args.audit_dir / "duplicate_manifest.json"
            ),
            "passed": passed,
            "validation_test_pixels_opened": True,
            "a1_authorized": False,
        }
        print(json.dumps(result, indent=2, sort_keys=True))
        return result
    except Exception as exc:
        failure = {
            "schema": "trkh_saspa_f1_duplicate_failure_v1",
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
        }
        _write_json(failure_path, failure)
        raise


def replay_report(report_path: Path) -> Dict[str, object]:
    report_path = report_path.resolve()
    audit_dir = report_path.parent
    report = _read_json(report_path)
    manifest_path = audit_dir / "duplicate_manifest.json"
    manifest = _read_json(manifest_path)
    if not isinstance(report, dict) or not isinstance(manifest, dict):
        raise TypeError("Duplicate replay inputs must be JSON objects")
    failures = []
    for row in manifest.get("files", []):
        if not isinstance(row, dict):
            failures.append("manifest_row_type")
            continue
        path = (audit_dir / str(row["path"])).resolve()
        if not f0._path_within(path, audit_dir):
            failures.append("manifest_path_escape")
            continue
        if not path.is_file():
            failures.append("missing:{}".format(row["path"]))
            continue
        if path.stat().st_size != int(row["size_bytes"]):
            failures.append("size:{}".format(row["path"]))
        if _sha256(path) != str(row["sha256"]):
            failures.append("sha256:{}".format(row["path"]))
    static = verify_static_inputs(LOCK_PATH, GENERATION_ROOT)
    model = verify_model_manifest(
        audit_dir / "dinov2_snapshot_manifest.json"
    )
    output_rows = report.get("output_rows", [])
    pairs = report.get("synthetic_pairs", [])
    if not isinstance(output_rows, list) or len(output_rows) != 10:
        failures.append("output_row_count")
        output_rows = []
    if not isinstance(pairs, list) or len(pairs) != 45:
        failures.append("synthetic_pair_count")
        pairs = []
    for row in output_rows:
        exact_match = row.get("exact_match")
        phash = row.get("phash_nearest")
        dino = row.get("dinov2_nearest")
        if not isinstance(phash, dict) or not isinstance(dino, dict):
            failures.append(
                "nearest_type:{}".format(row.get("output_id"))
            )
            continue
        checks = {
            "no_exact_match": exact_match is None,
            "phash_hamming_gt_4": int(phash["phash_hamming"])
            > PHASH_REJECT_MAX,
            "dinov2_cosine_lt_0_995": float(dino["dinov2_cosine"])
            < DINO_REJECT_MIN,
        }
        if checks != row.get("checks"):
            failures.append(
                "output_gate_arithmetic:{}".format(row.get("output_id"))
            )
        if bool(row.get("passed")) != all(checks.values()):
            failures.append(
                "output_pass_arithmetic:{}".format(row.get("output_id"))
            )
    for index, row in enumerate(pairs):
        checks = {
            "no_exact_match": not bool(row["exact_match"]),
            "phash_hamming_gt_4": int(row["phash_hamming"])
            > PHASH_REJECT_MAX,
            "dinov2_cosine_lt_0_995": float(row["dinov2_cosine"])
            < DINO_REJECT_MIN,
        }
        if checks != row.get("checks") or bool(
            row.get("passed")
        ) != all(checks.values()):
            failures.append("synthetic_pair_arithmetic:{}".format(index))
    recomputed_passed = (
        bool(output_rows)
        and bool(pairs)
        and all(bool(row.get("passed")) for row in output_rows)
        and all(bool(row.get("passed")) for row in pairs)
    )
    if bool(report.get("passed")) != recomputed_passed:
        failures.append("report_pass_arithmetic")
    boundary_checks = {
        "report_schema": report.get("schema") == SCHEMA,
        "test_pixels_only_in_audit": bool(
            report.get("validation_test_pixels_opened")
        ),
        "no_test_labels": not bool(
            report.get("validation_test_labels_opened")
        ),
        "no_reference_embeddings": not bool(
            report.get("reference_embeddings_persisted")
        ),
        "no_reference_images": not bool(
            report.get("reference_images_persisted")
        ),
        "no_class_aggregates": not bool(
            report.get("class_aggregates_persisted")
        ),
        "no_split_aggregates": not bool(
            report.get("split_aggregates_persisted")
        ),
        "no_regeneration": not bool(
            report.get("regeneration_authorized")
        ),
        "no_a1": not bool(report.get("a1_authorized")),
        "no_training": not bool(report.get("training_authorized")),
    }
    if not all(boundary_checks.values()):
        failures.append("report_boundary")
    result = {
        "schema": "trkh_saspa_f1_duplicate_replay_v1",
        "report_sha256": _sha256(report_path),
        "manifest_sha256": _sha256(manifest_path),
        "static_inputs_passed": static["passed"],
        "model_manifest_passed": model["passed"],
        "boundary_checks": boundary_checks,
        "failures": failures,
        "passed": not failures,
        "reference_pixels_opened": False,
        "validation_test_pixels_opened": False,
        "validation_test_labels_opened": False,
        "a1_authorized": False,
    }
    _write_json(audit_dir / "duplicate_replay.json", result)
    final_manifest = _artifact_manifest(
        audit_dir,
        "duplicate_final_manifest.json",
    )
    _write_json(
        audit_dir / "duplicate_final_manifest.json",
        final_manifest,
    )
    printed = dict(result)
    printed["final_manifest_sha256"] = _sha256(
        audit_dir / "duplicate_final_manifest.json"
    )
    print(json.dumps(printed, indent=2, sort_keys=True))
    if failures:
        raise RuntimeError(
            "Duplicate audit replay failed: {}".format(failures)
        )
    return result


def main(argv: Optional[Sequence[str]] = None) -> None:
    args = parse_args(argv)
    if args.prepare_model:
        prepare_model(args)
    elif args.preflight_only:
        run_preflight(args)
    elif args.formal_audit:
        run_formal_audit(args)
    else:
        replay_report(args.replay_report)


if __name__ == "__main__":
    main()
