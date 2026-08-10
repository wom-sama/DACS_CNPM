from __future__ import annotations

"""Prospectively locked SaSPA synthetic-data A0 no-output feasibility audit."""

import argparse
import csv
import gc
import hashlib
import importlib.metadata
import json
import os
import platform
import re
import shutil
import subprocess
import sys
import tempfile
import threading
import time
import traceback
from pathlib import Path
from typing import Dict, Iterable, List, Mapping, Optional, Sequence, Set, Tuple


METHOD = "saspa_dual_view_synthetic_a0_f0"
SCHEMA = "trkh_saspa_dual_view_synthetic_a0_f0_v1"
PIPELINE_WORKER_SCHEMA = "trkh_saspa_a0_f0_pipeline_load_worker_v1"
REPO_ROOT = Path(__file__).resolve().parents[2]
PROTOCOL_PATH = (
    REPO_ROOT
    / "docs"
    / "TRKH_5CLASS_SASPA_SYNTHETIC_A0_PROTOCOL_20260723.json"
)
PROTOCOL_MD_PATH = (
    REPO_ROOT
    / "docs"
    / "TRKH_5CLASS_SASPA_SYNTHETIC_A0_PROTOCOL_20260723.md"
)
PROTOCOL_LOCK_PATH = (
    REPO_ROOT
    / "docs"
    / "TRKH_5CLASS_SASPA_SYNTHETIC_A0_PROTOCOL_20260723.sha256"
)
REQUIREMENTS_PATH = REPO_ROOT / "requirements" / "trkh_saspa_a0.txt"
DEFAULT_OUTPUT_DIR = (
    REPO_ROOT / "runs" / "audit_saspa_dual_view_synthetic_a0_20260723"
)

LOCKED_PROTOCOL_SHA256 = (
    "81e479b68d22daebed769c31e43b378a96a6c14c1d183545398658100ca77477"
)
LOCKED_PROTOCOL_MD_SHA256 = (
    "ae2f390246c62994fdaa2ccdedf3308defff7dd278a47f4ad88e987f6ed76167"
)
LOCKED_PROTOCOL_LOCK_SHA256 = (
    "b7cb18ca76b37d4fe5ddb7b83f122b3caf518f398561b310b5cf70b8840d898f"
)
LOCKED_REQUIREMENTS_SHA256 = (
    "f218aaa00ed1a22530862bc626ac0a36277cec2edb78dcfd8c4fc22d29604764"
)
LOCKED_CURRENT_COMMAND_SHA256 = (
    "36b9aa1a21b765829acf4c8321be147bd76297de4ccdb8a40e6dee8e37940faf"
)
LOCKED_COMMAND_HISTORY_SHA256 = (
    "39bd2879ce66fddf36a953021ea1e40f8d9de6cb4334b9b825011b2b8dc98f53"
)
LOCKED_SASPA_COMMIT = "054230411863e8152ce475d48c1869e600a60f0c"
LOCKED_SASPA_ENV_SHA256 = (
    "20a0ebf450cab6ca3a15a9480e18aeda99dacc4d58beb81472f2af6b42f328f7"
)
LOCKED_SASPA_LICENSE_SHA256 = (
    "0f530a5c6cfa2464d836f185b26a48e92a09b66a000f327cfe7934f224f4a40d"
)
LOCKED_SASPA_RUN_AUG_SHA256 = (
    "3581ebb323d133715de3e164dc35e6737ba0f4873348ffc6b8f4b666a0458ba9"
)
LOCKED_SASPA_PAPER_SHA256 = (
    "4bf2b451fe720882bc3c0e7d7a80fa87e3fc53406832063cd08c3ed4ded3cfb7"
)
LOCKED_MODEL_REPOSITORY = "Salesforce/blipdiffusion-controlnet"
LOCKED_MODEL_REVISION = "e9e2aafc154c6a9d1593c8e4fb94c6a9a8a68593"
LOCKED_MODEL_REMOTE_BYTES = 7_688_436_558
EXPECTED_ELIGIBLE_COUNTS = (1507, 347, 1118, 1256, 764)
EXPECTED_OUTPUTS = 10

SASPA_ROOT = Path(r"D:\DataAI\Tools\research_sources\SaSPA-Aug")
SASPA_PAPER_PATH = Path(
    r"D:\DataAI\Tools\research_sources\papers\SaSPA_NeurIPS_2024.pdf"
)
CURRENT_COMMAND_PATH = (
    REPO_ROOT / "docs" / "TRKH_CURRENT_BEST_FULL_TRAIN_COMMANDS_20260706.txt"
)
COMMAND_HISTORY_PATH = (
    REPO_ROOT / "docs" / "TRKH_CURRENT_BEST_COMMAND_UPDATE_HISTORY.txt"
)
IMPLEMENTATION_PATHS = (
    "requirements/trkh_saspa_a0.txt",
    "trkh/tools/audit_saspa_synthetic_a0_f0.py",
    "tests/test_audit_saspa_synthetic_a0_f0.py",
    "scripts/setup_trkh_saspa_a0_env.ps1",
    "scripts/run_trkh_saspa_synthetic_a0_f0.ps1",
    "docs/TRKH_5CLASS_SASPA_SYNTHETIC_A0_PROTOCOL_20260723.json",
    "docs/TRKH_5CLASS_SASPA_SYNTHETIC_A0_PROTOCOL_20260723.md",
    "docs/TRKH_5CLASS_SASPA_SYNTHETIC_A0_PROTOCOL_20260723.sha256",
)


def parse_args(argv: Optional[Sequence[str]] = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Locked SaSPA A0 F0 audit; this tool never generates pixels."
    )
    parser.add_argument("--protocol", type=Path, default=PROTOCOL_PATH)
    parser.add_argument("--requirements", type=Path, default=REQUIREMENTS_PATH)
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT_DIR)
    parser.add_argument(
        "--cache-root",
        type=Path,
        default=Path(r"D:\DataAI\Tools\hf_cache\trkh_saspa_a0"),
    )
    mode = parser.add_mutually_exclusive_group(required=True)
    mode.add_argument("--preflight-only", action="store_true")
    mode.add_argument("--runtime-check-only", action="store_true")
    mode.add_argument("--formal-f0", action="store_true")
    mode.add_argument("--replay-summary", type=Path)
    mode.add_argument(
        "--pipeline-load-worker",
        action="store_true",
        help=argparse.SUPPRESS,
    )
    parser.add_argument(
        "--worker-snapshot-manifest",
        type=Path,
        help=argparse.SUPPRESS,
    )
    parser.add_argument(
        "--worker-result",
        type=Path,
        help=argparse.SUPPRESS,
    )
    parser.add_argument(
        "--worker-allowed-pid",
        action="append",
        type=int,
        default=[],
        help=argparse.SUPPRESS,
    )
    return parser.parse_args(argv)


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with Path(path).open("rb") as handle:
        for chunk in iter(lambda: handle.read(8 * 1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _canonical_sha256(value: object) -> str:
    payload = json.dumps(
        value, sort_keys=True, separators=(",", ":"), ensure_ascii=True
    ).encode("utf-8")
    return hashlib.sha256(payload).hexdigest()


def _read_json(path: Path) -> object:
    with Path(path).open("r", encoding="utf-8") as handle:
        return json.load(handle)


def _write_json(path: Path, value: object) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(path.name + ".tmp")
    with temporary.open("w", encoding="utf-8", newline="\n") as handle:
        json.dump(value, handle, indent=2, sort_keys=True, ensure_ascii=True)
        handle.write("\n")
    temporary.replace(path)


def _verify_hash(path: Path, expected: str, label: str) -> Dict[str, object]:
    path = Path(path)
    if not path.is_file():
        raise FileNotFoundError("{} is missing: {}".format(label, path))
    actual = _sha256(path)
    if actual.lower() != expected.lower():
        raise ValueError(
            "{} SHA-256 mismatch: expected {}, got {}".format(
                label, expected, actual
            )
        )
    return {
        "path": str(path.resolve()),
        "sha256": actual,
        "size_bytes": path.stat().st_size,
    }


def _git(repo: Path, *arguments: str) -> str:
    return subprocess.check_output(
        ["git", "-C", str(Path(repo).resolve()), *arguments],
        text=True,
        encoding="utf-8",
        errors="replace",
    ).strip()


def _normalize_distribution_name(name: str) -> str:
    return re.sub(r"[-_.]+", "-", name).lower()


def parse_requirement_pins(path: Path) -> Dict[str, str]:
    pins: Dict[str, str] = {}
    for raw_line in Path(path).read_text(encoding="utf-8").splitlines():
        line = raw_line.strip()
        if not line or line.startswith("#") or line.startswith("--"):
            continue
        if "==" not in line:
            raise ValueError("Requirement is not exactly pinned: {}".format(line))
        name, version = line.split("==", 1)
        normalized = _normalize_distribution_name(name.strip())
        if not normalized or not version.strip() or normalized in pins:
            raise ValueError("Invalid or duplicate requirement: {}".format(line))
        pins[normalized] = version.strip()
    if not pins:
        raise ValueError("No package pins found")
    return pins


def _installed_runtime_manifest(
    requirements_path: Path,
    *,
    hash_distribution_files: bool,
    import_runtime: bool,
) -> Dict[str, object]:
    pins = parse_requirement_pins(requirements_path)
    packages: List[Dict[str, object]] = []
    mismatches: List[Dict[str, str]] = []
    installed_versions: Dict[str, str] = {}
    prefix = Path(sys.prefix).resolve()
    for name in sorted(pins):
        expected = pins[name]
        try:
            distribution = importlib.metadata.distribution(name)
            actual = distribution.version
        except importlib.metadata.PackageNotFoundError:
            actual = "<missing>"
            distribution = None
        if actual != expected:
            mismatches.append(
                {"package": name, "expected": expected, "actual": actual}
            )
        row: Dict[str, object] = {
            "package": name,
            "expected_version": expected,
            "actual_version": actual,
        }
        installed_versions[name] = actual
        if distribution is not None:
            metadata_path = Path(distribution._path).resolve()  # type: ignore[attr-defined]
            row["metadata_path"] = str(metadata_path)
            try:
                metadata_path.relative_to(prefix)
                row["inside_runtime_prefix"] = True
            except ValueError:
                row["inside_runtime_prefix"] = False
                mismatches.append(
                    {
                        "package": name,
                        "expected": "installed under {}".format(prefix),
                        "actual": str(metadata_path),
                    }
                )
            record_path = metadata_path / "RECORD"
            row["record_sha256"] = (
                _sha256(record_path) if record_path.is_file() else None
            )
            if hash_distribution_files:
                aggregate = hashlib.sha256()
                file_count = 0
                total_bytes = 0
                for entry in sorted(
                    distribution.files or (), key=lambda item: str(item).lower()
                ):
                    installed = Path(distribution.locate_file(entry))
                    if not installed.is_file():
                        continue
                    relative = str(entry).replace("\\", "/")
                    file_hash = _sha256(installed)
                    size = installed.stat().st_size
                    aggregate.update(relative.encode("utf-8"))
                    aggregate.update(b"\0")
                    aggregate.update(str(size).encode("ascii"))
                    aggregate.update(b"\0")
                    aggregate.update(file_hash.encode("ascii"))
                    aggregate.update(b"\n")
                    file_count += 1
                    total_bytes += size
                row["installed_file_count"] = file_count
                row["installed_total_bytes"] = total_bytes
                row["installed_content_sha256"] = aggregate.hexdigest()
        packages.append(row)

    pip_check = subprocess.run(
        [sys.executable, "-m", "pip", "check"],
        stdout=subprocess.PIPE,
        stderr=subprocess.STDOUT,
        text=True,
        encoding="utf-8",
        errors="replace",
        check=False,
    )
    protocol_versions = {
        "python": "3.11.9",
        "torch": "2.6.0+cu124",
        "torchvision": "0.21.0+cu124",
    }
    runtime_versions = {
        "python": platform.python_version(),
        "torch": installed_versions.get("torch", "<missing>"),
        "torchvision": installed_versions.get("torchvision", "<missing>"),
    }
    for name, expected in protocol_versions.items():
        actual = runtime_versions[name]
        if actual != expected:
            mismatches.append(
                {"package": name, "expected": expected, "actual": actual}
            )
    runtime_import: Dict[str, object] = {
        "attempted": import_runtime,
        "cuda_available": None,
        "pipeline_class": None,
    }
    if import_runtime:
        import torch
        import torchvision
        from diffusers import BlipDiffusionControlNetPipeline

        runtime_import = {
            "attempted": True,
            "torch_version": str(torch.__version__),
            "torchvision_version": str(torchvision.__version__),
            "cuda_version": str(torch.version.cuda),
            "cuda_available": bool(torch.cuda.is_available()),
            "pipeline_class": BlipDiffusionControlNetPipeline.__name__,
        }
        if str(torch.__version__) != protocol_versions["torch"]:
            mismatches.append(
                {
                    "package": "torch import",
                    "expected": protocol_versions["torch"],
                    "actual": str(torch.__version__),
                }
            )
        if str(torchvision.__version__) != protocol_versions["torchvision"]:
            mismatches.append(
                {
                    "package": "torchvision import",
                    "expected": protocol_versions["torchvision"],
                    "actual": str(torchvision.__version__),
                }
            )
        if not torch.cuda.is_available():
            mismatches.append(
                {
                    "package": "CUDA runtime",
                    "expected": "available",
                    "actual": "unavailable",
                }
            )
    isolation = {
        "prefix": str(prefix),
        "base_prefix": str(Path(sys.base_prefix).resolve()),
        "is_venv": Path(sys.prefix).resolve() != Path(sys.base_prefix).resolve(),
        "system_site_packages_inherited": any(
            "appdata/local/programs/python" in str(Path(item)).replace("\\", "/").lower()
            and "site-packages" in str(Path(item)).replace("\\", "/").lower()
            for item in sys.path
        ),
        "trkh_training_venv_inherited": any(
            str(Path(item)).replace("\\", "/").lower().startswith(
                "d:/dataai/.venv/"
            )
            for item in sys.path
        ),
    }
    if not isolation["is_venv"]:
        mismatches.append(
            {"package": "runtime", "expected": "isolated venv", "actual": str(prefix)}
        )
    if isolation["system_site_packages_inherited"]:
        mismatches.append(
            {
                "package": "runtime",
                "expected": "no global site-packages",
                "actual": "global site-packages present on sys.path",
            }
        )
    if isolation["trkh_training_venv_inherited"]:
        mismatches.append(
            {
                "package": "runtime",
                "expected": "no D:/DataAI/.venv inheritance",
                "actual": "training venv present on sys.path",
            }
        )
    return {
        "schema": "trkh_saspa_a0_environment_manifest_v1",
        "python_executable": str(Path(sys.executable).resolve()),
        "runtime_versions": runtime_versions,
        "runtime_import": runtime_import,
        "isolation": isolation,
        "requirements": _verify_hash(
            requirements_path, LOCKED_REQUIREMENTS_SHA256, "SaSPA requirements"
        ),
        "packages": packages,
        "pip_check": {
            "exit_code": pip_check.returncode,
            "output": pip_check.stdout.strip(),
        },
        "mismatches": mismatches,
        "distribution_files_hashed": hash_distribution_files,
        "passed": not mismatches and pip_check.returncode == 0,
    }


def _repo_state() -> Dict[str, object]:
    status = _git(REPO_ROOT, "status", "--porcelain=v1", "--untracked-files=all")
    tracked_dirty: List[str] = []
    untracked: List[str] = []
    for line in status.splitlines():
        code = line[:2]
        path = line[3:].strip().strip('"').replace("\\", "/")
        if code == "??":
            untracked.append(path)
        else:
            tracked_dirty.append(path)
    tracked_files = set(_git(REPO_ROOT, "ls-files").splitlines())
    head = _git(REPO_ROOT, "rev-parse", "HEAD")
    upstream = _git(REPO_ROOT, "rev-parse", "@{upstream}")
    missing = sorted(path for path in IMPLEMENTATION_PATHS if path not in tracked_files)
    dirty_implementation = sorted(
        path for path in tracked_dirty if path in IMPLEMENTATION_PATHS
    )
    return {
        "head": head,
        "upstream": upstream,
        "head_equals_upstream": head == upstream,
        "tracked_dirty": tracked_dirty,
        "untracked_preserved": untracked,
        "missing_implementation_paths": missing,
        "dirty_implementation_paths": dirty_implementation,
        "passed": head == upstream
        and not tracked_dirty
        and not missing
        and not dirty_implementation,
    }


def verify_static_inputs(
    protocol_path: Path = PROTOCOL_PATH,
    requirements_path: Path = REQUIREMENTS_PATH,
) -> Dict[str, object]:
    files = {
        "protocol_json": _verify_hash(
            protocol_path, LOCKED_PROTOCOL_SHA256, "SaSPA protocol JSON"
        ),
        "protocol_markdown": _verify_hash(
            PROTOCOL_MD_PATH,
            LOCKED_PROTOCOL_MD_SHA256,
            "SaSPA protocol Markdown",
        ),
        "protocol_lock": _verify_hash(
            PROTOCOL_LOCK_PATH,
            LOCKED_PROTOCOL_LOCK_SHA256,
            "SaSPA protocol lock",
        ),
        "requirements": _verify_hash(
            requirements_path,
            LOCKED_REQUIREMENTS_SHA256,
            "SaSPA requirements",
        ),
        "current_commands": _verify_hash(
            CURRENT_COMMAND_PATH,
            LOCKED_CURRENT_COMMAND_SHA256,
            "current-best commands",
        ),
        "command_history": _verify_hash(
            COMMAND_HISTORY_PATH,
            LOCKED_COMMAND_HISTORY_SHA256,
            "current-best command history",
        ),
        "saspa_environment": _verify_hash(
            SASPA_ROOT / "environment.yml",
            LOCKED_SASPA_ENV_SHA256,
            "SaSPA environment",
        ),
        "saspa_license": _verify_hash(
            SASPA_ROOT / "LICENSE",
            LOCKED_SASPA_LICENSE_SHA256,
            "SaSPA license",
        ),
        "saspa_run_aug": _verify_hash(
            SASPA_ROOT / "run_aug" / "run_aug.py",
            LOCKED_SASPA_RUN_AUG_SHA256,
            "SaSPA augmentation source",
        ),
        "saspa_paper": _verify_hash(
            SASPA_PAPER_PATH,
            LOCKED_SASPA_PAPER_SHA256,
            "SaSPA paper",
        ),
    }
    protocol = _read_json(protocol_path)
    if not isinstance(protocol, dict):
        raise TypeError("Protocol must be a JSON object")
    immutable = protocol["immutable_inputs"]
    for key, label in (
        ("yolo_data_yaml", "yolo_f data YAML"),
        ("balance_yaml", "yolo_f balance YAML"),
        ("source_manifest", "yolo_f source manifest"),
    ):
        item = immutable[key]
        files[key] = _verify_hash(
            Path(item["path"]), str(item["sha256"]), label
        )
    commit = _git(SASPA_ROOT, "rev-parse", "HEAD")
    status = _git(SASPA_ROOT, "status", "--porcelain=v1")
    if commit != LOCKED_SASPA_COMMIT or status:
        raise ValueError(
            "SaSPA repository differs from locked clean commit: {} {!r}".format(
                commit, status
            )
        )
    if int(protocol.get("protocol_revision", 0)) != 2:
        raise ValueError("SaSPA protocol revision 2 is required")
    generator = protocol["generator"]
    if (
        generator["model_repository"] != LOCKED_MODEL_REPOSITORY
        or generator["model_revision"] != LOCKED_MODEL_REVISION
        or int(generator["remote_snapshot_bytes"]) != LOCKED_MODEL_REMOTE_BYTES
    ):
        raise ValueError("Generator metadata differs from the prospective lock")
    if bool(protocol["phase_f0_no_output"]["may_generate_pixels"]):
        raise ValueError("F0 protocol unexpectedly allows pixel generation")
    return {
        "schema": "trkh_saspa_a0_static_inputs_v1",
        "files": files,
        "saspa_repository": {
            "path": str(SASPA_ROOT.resolve()),
            "commit": commit,
            "clean": not status,
        },
        "protocol_id": protocol["protocol_id"],
        "protocol_revision": protocol["protocol_revision"],
        "generator_repository": generator["model_repository"],
        "generator_revision": generator["model_revision"],
        "passed": True,
    }


def _path_within(path: Path, root: Path) -> bool:
    try:
        path.resolve().relative_to(root.resolve())
        return True
    except ValueError:
        return False


def _read_yolo_object(label_path: Path) -> Optional[Tuple[int, Tuple[float, ...]]]:
    lines = [
        line.strip()
        for line in label_path.read_text(encoding="utf-8").splitlines()
        if line.strip()
    ]
    if len(lines) != 1:
        return None
    fields = lines[0].split()
    if len(fields) != 5:
        return None
    class_float = float(fields[0])
    class_index = int(class_float)
    if class_float != float(class_index):
        return None
    bbox = tuple(float(value) for value in fields[1:])
    if not all(0.0 <= value <= 1.0 for value in bbox):
        return None
    return class_index, bbox


def _selection_hash(namespace: str, *parts: object) -> str:
    joined = "|".join([namespace] + [str(part) for part in parts])
    return hashlib.sha256(joined.encode("utf-8")).hexdigest()


def _eligible_candidates(protocol: Mapping[str, object]) -> Tuple[List[Dict[str, object]], Dict[str, object]]:
    immutable = protocol["immutable_inputs"]  # type: ignore[index]
    tiny = protocol["tiny_cohort"]  # type: ignore[index]
    eligibility = tiny["eligibility"]  # type: ignore[index]
    yolo_images = Path(immutable["yolo_train_images"])  # type: ignore[index]
    yolo_labels = Path(immutable["yolo_train_labels"])  # type: ignore[index]
    class_train = Path(immutable["classification_train_root"])  # type: ignore[index]
    yolo_root = yolo_images.parents[1]
    class_root = class_train.parent
    class_folders = {
        int(item["index"]): str(item["folder"])
        for item in protocol["classes"]  # type: ignore[index]
    }
    manifest_path = Path(immutable["source_manifest"]["path"])  # type: ignore[index]
    candidates: List[Dict[str, object]] = []
    rejected = {
        "non_train": 0,
        "outside_locked_roots": 0,
        "missing_input": 0,
        "not_single_object": 0,
        "class_out_of_range": 0,
        "bbox_area": 0,
        "bbox_border_margin": 0,
        "missing_classification_crop": 0,
    }
    with manifest_path.open("r", encoding="utf-8-sig", newline="") as handle:
        for row in csv.DictReader(handle):
            if row.get("split") != "train":
                rejected["non_train"] += 1
                continue
            image_path = Path(str(row["output_image"]))
            label_path = Path(str(row["output_label"]))
            if not _path_within(image_path, yolo_images) or not _path_within(
                label_path, yolo_labels
            ):
                rejected["outside_locked_roots"] += 1
                continue
            if not image_path.is_file() or not label_path.is_file():
                rejected["missing_input"] += 1
                continue
            parsed = _read_yolo_object(label_path)
            if parsed is None:
                rejected["not_single_object"] += 1
                continue
            class_index, bbox = parsed
            if class_index not in class_folders:
                rejected["class_out_of_range"] += 1
                continue
            x_center, y_center, width, height = bbox
            area = width * height
            if not (
                float(eligibility["bbox_area_ratio_min"])
                <= area
                <= float(eligibility["bbox_area_ratio_max"])
            ):
                rejected["bbox_area"] += 1
                continue
            margin = min(
                x_center - width / 2.0,
                1.0 - (x_center + width / 2.0),
                y_center - height / 2.0,
                1.0 - (y_center + height / 2.0),
            )
            if margin < float(eligibility["bbox_normalized_border_margin_min"]):
                rejected["bbox_border_margin"] += 1
                continue
            crop_path = (
                class_train
                / class_folders[class_index]
                / "{}_box000.jpg".format(image_path.stem)
            )
            if not crop_path.is_file():
                rejected["missing_classification_crop"] += 1
                continue
            candidates.append(
                {
                    "class_index": class_index,
                    "leakage_group": str(row["leakage_group"]),
                    "image_stem": image_path.stem,
                    "edge_path": image_path,
                    "edge_relative_path": image_path.relative_to(yolo_root).as_posix(),
                    "label_path": label_path,
                    "label_relative_path": label_path.relative_to(yolo_root).as_posix(),
                    "subject_path": crop_path,
                    "subject_relative_path": crop_path.relative_to(class_root).as_posix(),
                    "bbox_xywh_normalized": list(bbox),
                    "bbox_area_ratio": area,
                    "bbox_border_margin": margin,
                }
            )
    counts = [
        sum(int(item["class_index"]) == class_index for item in candidates)
        for class_index in range(5)
    ]
    population_identity = [
        {
            "class_index": item["class_index"],
            "leakage_group": item["leakage_group"],
            "edge_relative_path": item["edge_relative_path"],
            "subject_relative_path": item["subject_relative_path"],
            "bbox_xywh_normalized": item["bbox_xywh_normalized"],
        }
        for item in sorted(
            candidates,
            key=lambda value: (
                int(value["class_index"]),
                str(value["edge_relative_path"]),
            ),
        )
    ]
    return candidates, {
        "eligible_counts": counts,
        "eligible_total": len(candidates),
        "rejected": rejected,
        "population_identity_sha256": _canonical_sha256(population_identity),
        "train_only": True,
        "validation_test_pixels_opened": False,
    }


def select_tiny_cohort(protocol: Mapping[str, object]) -> Dict[str, object]:
    candidates, population = _eligible_candidates(protocol)
    tiny = protocol["tiny_cohort"]  # type: ignore[index]
    protocol_id = str(protocol["protocol_id"])
    outputs_per_class = int(tiny["outputs_per_class"])  # type: ignore[index]
    descriptors = {
        int(item["index"]): str(item["english_descriptor"])
        for item in protocol["classes"]  # type: ignore[index]
    }
    templates = [str(value) for value in protocol["prompt_templates"]]  # type: ignore[index]
    used_groups: Set[str] = set()
    selected: List[Dict[str, object]] = []
    by_class = {
        class_index: [
            item for item in candidates if int(item["class_index"]) == class_index
        ]
        for class_index in range(5)
    }
    for class_index in range(5):
        edge_ranked = sorted(
            by_class[class_index],
            key=lambda item: _selection_hash(
                "edge",
                protocol_id,
                item["leakage_group"],
                item["edge_relative_path"],
            ),
        )
        edges: List[Dict[str, object]] = []
        for candidate in edge_ranked:
            group = str(candidate["leakage_group"])
            if group in used_groups:
                continue
            edges.append(candidate)
            used_groups.add(group)
            if len(edges) == outputs_per_class:
                break
        if len(edges) != outputs_per_class:
            raise RuntimeError(
                "Unable to select {} edge rows for class {}".format(
                    outputs_per_class, class_index
                )
            )
        for edge_rank, edge in enumerate(edges):
            subjects = sorted(
                by_class[class_index],
                key=lambda item: _selection_hash(
                    "subject",
                    protocol_id,
                    edge["leakage_group"],
                    item["leakage_group"],
                    item["subject_relative_path"],
                ),
            )
            subject = next(
                (
                    item
                    for item in subjects
                    if str(item["leakage_group"]) not in used_groups
                    and str(item["leakage_group"]) != str(edge["leakage_group"])
                    and str(item["image_stem"]) != str(edge["image_stem"])
                ),
                None,
            )
            if subject is None:
                raise RuntimeError(
                    "Unable to select a distinct subject for class {} edge {}".format(
                        class_index, edge_rank
                    )
                )
            used_groups.add(str(subject["leakage_group"]))
            prompt_index = edge_rank % len(templates)
            prompt = templates[prompt_index].format(
                subclass=descriptors[class_index]
            )
            seed_hash = _selection_hash(
                "seed",
                protocol_id,
                edge["edge_relative_path"],
                subject["subject_relative_path"],
                prompt_index,
            )
            seed = int(seed_hash[:16], 16) >> 1
            edge_hash = _selection_hash(
                "edge",
                protocol_id,
                edge["leakage_group"],
                edge["edge_relative_path"],
            )
            output_id = "c{}_r{}_{}".format(
                class_index, edge_rank, edge_hash[:12]
            )
            selected.append(
                {
                    "output_id": output_id,
                    "class_index": class_index,
                    "class_folder": next(
                        str(item["folder"])
                        for item in protocol["classes"]  # type: ignore[index]
                        if int(item["index"]) == class_index
                    ),
                    "edge_rank": edge_rank,
                    "edge_rank_sha256": edge_hash,
                    "edge_relative_path": edge["edge_relative_path"],
                    "edge_leakage_group": edge["leakage_group"],
                    "edge_file_sha256": _sha256(Path(edge["edge_path"])),
                    "edge_label_relative_path": edge["label_relative_path"],
                    "edge_label_file_sha256": _sha256(Path(edge["label_path"])),
                    "bbox_xywh_normalized": edge["bbox_xywh_normalized"],
                    "bbox_area_ratio": edge["bbox_area_ratio"],
                    "bbox_border_margin": edge["bbox_border_margin"],
                    "subject_relative_path": subject["subject_relative_path"],
                    "subject_leakage_group": subject["leakage_group"],
                    "subject_file_sha256": _sha256(Path(subject["subject_path"])),
                    "prompt_template_index": prompt_index,
                    "prompt": prompt,
                    "prompt_sha256": hashlib.sha256(
                        prompt.encode("utf-8")
                    ).hexdigest(),
                    "seed": seed,
                }
            )
    if len(selected) != EXPECTED_OUTPUTS:
        raise RuntimeError("Tiny cohort does not contain ten rows")
    all_groups: List[str] = []
    for row in selected:
        all_groups.extend(
            [str(row["edge_leakage_group"]), str(row["subject_leakage_group"])]
        )
    if len(all_groups) != len(set(all_groups)):
        raise RuntimeError("A leakage group was reused in the tiny cohort")
    return {
        "schema": "trkh_saspa_a0_source_selection_v1",
        "protocol_id": protocol_id,
        "population": population,
        "rows": selected,
        "row_count": len(selected),
        "all_leakage_groups_unique": True,
        "selection_sha256": _canonical_sha256(selected),
        "train_only": True,
        "validation_test_pixels_opened": False,
        "synthetic_pixels_generated": False,
    }


def _write_source_csv(path: Path, rows: Iterable[Mapping[str, object]]) -> None:
    fieldnames = (
        "output_id",
        "class_index",
        "class_folder",
        "edge_rank",
        "edge_rank_sha256",
        "edge_relative_path",
        "edge_leakage_group",
        "edge_file_sha256",
        "edge_label_relative_path",
        "edge_label_file_sha256",
        "bbox_xywh_normalized",
        "bbox_area_ratio",
        "bbox_border_margin",
        "subject_relative_path",
        "subject_leakage_group",
        "subject_file_sha256",
        "prompt_template_index",
        "prompt",
        "prompt_sha256",
        "seed",
    )
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames, lineterminator="\n")
        writer.writeheader()
        for item in rows:
            row = dict(item)
            row["bbox_xywh_normalized"] = json.dumps(
                row["bbox_xywh_normalized"], separators=(",", ":")
            )
            writer.writerow(row)


def _nvidia_snapshot() -> Dict[str, object]:
    command = shutil.which("nvidia-smi")
    if command is None:
        return {"available": False, "error": "nvidia-smi not found"}
    gpu = subprocess.run(
        [
            command,
            "--query-gpu=name,memory.total,memory.used,utilization.gpu,temperature.gpu,power.draw",
            "--format=csv,noheader,nounits",
        ],
        stdout=subprocess.PIPE,
        stderr=subprocess.STDOUT,
        text=True,
        encoding="utf-8",
        errors="replace",
        check=False,
    )
    compute = subprocess.run(
        [
            command,
            "--query-compute-apps=pid,process_name,used_memory",
            "--format=csv,noheader,nounits",
        ],
        stdout=subprocess.PIPE,
        stderr=subprocess.STDOUT,
        text=True,
        encoding="utf-8",
        errors="replace",
        check=False,
    )
    used_mib: Optional[float] = None
    if gpu.returncode == 0 and gpu.stdout.strip():
        fields = [value.strip() for value in gpu.stdout.splitlines()[0].split(",")]
        if len(fields) >= 3:
            try:
                used_mib = float(fields[2])
            except ValueError:
                used_mib = None
    return {
        "available": gpu.returncode == 0,
        "gpu_query": gpu.stdout.strip(),
        "compute_query": compute.stdout.strip(),
        "gpu_exit_code": gpu.returncode,
        "compute_exit_code": compute.returncode,
        "memory_used_mib": used_mib,
    }


def _is_known_venv_redirector(
    candidate: Mapping[str, object],
    *,
    current_parent_pid: int,
    current_command_line: Sequence[str],
    current_create_time: float,
    runtime_prefix: Path,
) -> bool:
    if int(candidate["pid"]) != int(current_parent_pid):
        return False
    if str(candidate.get("name") or "").lower() not in (
        "python.exe",
        "pythonw.exe",
    ):
        return False
    executable = candidate.get("exe")
    if not executable:
        return False
    expected = runtime_prefix / "Scripts" / "python.exe"
    if os.path.normcase(os.path.abspath(str(executable))) != os.path.normcase(
        os.path.abspath(str(expected))
    ):
        return False
    candidate_command_line = [
        str(value) for value in candidate.get("cmdline") or ()
    ]
    if candidate_command_line[1:] != [
        str(value) for value in current_command_line[1:]
    ]:
        return False
    candidate_create_time = float(candidate.get("create_time") or 0.0)
    return (
        candidate_create_time <= current_create_time
        and current_create_time - candidate_create_time <= 10.0
    )


def resource_snapshot(
    *,
    allowed_process_ids: Optional[Set[int]] = None,
) -> Dict[str, object]:
    import psutil

    explicitly_allowed = set(allowed_process_ids or ())
    physical = psutil.virtual_memory()
    swap = psutil.swap_memory()
    virtual_total = int(physical.total + swap.total)
    virtual_free = int(physical.available + swap.free)
    process = psutil.Process()
    current_command_line = process.cmdline()
    current_create_time = process.create_time()
    current_parent_pid = process.ppid()
    external_workflows = []
    allowed_runtime_launchers = []
    for item in psutil.process_iter(
        attrs=("pid", "name", "exe", "create_time", "cmdline")
    ):
        try:
            name = str(item.info["name"] or "").lower()
            if name not in ("python.exe", "pythonw.exe", "trtexec.exe"):
                continue
            if int(item.info["pid"]) == os.getpid():
                continue
            if int(item.info["pid"]) in explicitly_allowed:
                allowed_runtime_launchers.append(
                    {
                        "pid": int(item.info["pid"]),
                        "name": item.info["name"],
                        "executable": item.info["exe"],
                        "create_time": item.info["create_time"],
                        "command_line": list(item.info["cmdline"] or ()),
                        "reason": "exact parent process of the isolated F0 load worker",
                    }
                )
                continue
            if _is_known_venv_redirector(
                item.info,
                current_parent_pid=current_parent_pid,
                current_command_line=current_command_line,
                current_create_time=current_create_time,
                runtime_prefix=Path(sys.prefix),
            ):
                allowed_runtime_launchers.append(
                    {
                        "pid": int(item.info["pid"]),
                        "name": item.info["name"],
                        "executable": item.info["exe"],
                        "create_time": item.info["create_time"],
                        "command_line": list(item.info["cmdline"] or ()),
                        "reason": "exact Windows venv redirector parent for this invocation",
                    }
                )
                continue
            external_workflows.append(
                {
                    "pid": int(item.info["pid"]),
                    "name": item.info["name"],
                    "executable": item.info["exe"],
                    "create_time": item.info["create_time"],
                    "command_line": list(item.info["cmdline"] or ()),
                }
            )
        except (psutil.AccessDenied, psutil.NoSuchProcess):
            continue
    disk = psutil.disk_usage(r"D:\\")
    memory = process.memory_info()
    return {
        "timestamp_unix": time.time(),
        "pid": os.getpid(),
        "physical_total_bytes": int(physical.total),
        "physical_free_bytes": int(physical.available),
        "swap_total_bytes": int(swap.total),
        "swap_free_bytes": int(swap.free),
        "virtual_total_bytes": virtual_total,
        "virtual_free_bytes": virtual_free,
        "virtual_used_fraction": (
            1.0 - virtual_free / virtual_total if virtual_total else 1.0
        ),
        "d_drive_total_bytes": int(disk.total),
        "d_drive_free_bytes": int(disk.free),
        "process_rss_bytes": int(memory.rss),
        "process_vms_bytes": int(memory.vms),
        "allowed_runtime_launchers": allowed_runtime_launchers,
        "external_python_or_trtexec": external_workflows,
        "nvidia": _nvidia_snapshot(),
    }


def evaluate_f0_resource_gates(
    protocol: Mapping[str, object],
    snapshot: Mapping[str, object],
    *,
    model_snapshot_bytes: Optional[int] = None,
    pipeline_load_seconds: Optional[float] = None,
) -> Dict[str, object]:
    gib = float(1024 ** 3)
    limits = protocol["phase_f0_no_output"]["resource_gates"]  # type: ignore[index]
    checks = {
        "d_drive_free": float(snapshot["d_drive_free_bytes"]) / gib
        >= float(limits["d_drive_free_gib_min"]),
        "physical_ram_free": float(snapshot["physical_free_bytes"]) / gib
        >= float(limits["free_physical_ram_gib_min"]),
        "virtual_memory_free": float(snapshot["virtual_free_bytes"]) / gib
        >= float(limits["free_virtual_memory_gib_min"]),
        "virtual_memory_fraction": float(snapshot["virtual_used_fraction"])
        < float(limits["host_virtual_memory_fraction_max"]),
        "no_unknown_python_or_trtexec": not bool(
            snapshot["external_python_or_trtexec"]
        ),
    }
    if model_snapshot_bytes is not None:
        checks["model_snapshot_size"] = model_snapshot_bytes / gib <= float(
            limits["model_snapshot_gib_max"]
        )
    if pipeline_load_seconds is not None:
        checks["pipeline_load_time"] = pipeline_load_seconds / 60.0 <= float(
            limits["pipeline_load_minutes_max"]
        )
    return {
        "limits": limits,
        "checks": checks,
        "passed": all(checks.values()),
    }


class ResourceSampler:
    def __init__(
        self,
        interval_seconds: float = 2.0,
        *,
        allowed_process_ids: Optional[Set[int]] = None,
    ) -> None:
        self.interval_seconds = interval_seconds
        self.allowed_process_ids = set(allowed_process_ids or ())
        self.samples: List[Dict[str, object]] = []
        self._stop_event = threading.Event()
        self._thread = threading.Thread(target=self._run, daemon=True)

    def __enter__(self) -> "ResourceSampler":
        self._thread.start()
        return self

    def __exit__(self, *unused: object) -> None:
        self._stop_event.set()
        self._thread.join(timeout=max(5.0, self.interval_seconds * 2.0))

    def _run(self) -> None:
        while not self._stop_event.is_set():
            try:
                snapshot = resource_snapshot(
                    allowed_process_ids=self.allowed_process_ids
                )
                try:
                    import torch

                    snapshot["torch_cuda_allocated_bytes"] = int(
                        torch.cuda.memory_allocated()
                    )
                    snapshot["torch_cuda_reserved_bytes"] = int(
                        torch.cuda.memory_reserved()
                    )
                except Exception:
                    snapshot["torch_cuda_allocated_bytes"] = None
                    snapshot["torch_cuda_reserved_bytes"] = None
                self.samples.append(snapshot)
            except Exception as exc:
                self.samples.append(
                    {
                        "timestamp_unix": time.time(),
                        "sampling_error": "{}: {}".format(
                            type(exc).__name__, exc
                        ),
                    }
                )
            self._stop_event.wait(self.interval_seconds)

    def summary(self) -> Dict[str, object]:
        def maximum(key: str) -> Optional[float]:
            values = [
                float(item[key])
                for item in self.samples
                if item.get(key) is not None
            ]
            return max(values) if values else None

        nvidia_values = [
            float(item["nvidia"]["memory_used_mib"])
            for item in self.samples
            if isinstance(item.get("nvidia"), dict)
            and item["nvidia"].get("memory_used_mib") is not None
        ]
        external_process_ids = sorted(
            {
                int(process["pid"])
                for item in self.samples
                for process in item.get("external_python_or_trtexec", [])
                if isinstance(process, dict) and process.get("pid") is not None
            }
        )
        return {
            "sample_count": len(self.samples),
            "max_process_rss_bytes": maximum("process_rss_bytes"),
            "max_process_vms_bytes": maximum("process_vms_bytes"),
            "max_virtual_used_fraction": maximum("virtual_used_fraction"),
            "max_torch_cuda_allocated_bytes": maximum(
                "torch_cuda_allocated_bytes"
            ),
            "max_torch_cuda_reserved_bytes": maximum("torch_cuda_reserved_bytes"),
            "max_nvml_used_mib": max(nvidia_values) if nvidia_values else None,
            "external_python_or_trtexec_pids": external_process_ids,
        }


def _remote_model_metadata(protocol: Mapping[str, object]) -> Dict[str, object]:
    from huggingface_hub import HfApi

    generator = protocol["generator"]  # type: ignore[index]
    info = HfApi().model_info(
        repo_id=str(generator["model_repository"]),
        revision=str(generator["model_revision"]),
        files_metadata=True,
    )
    files = []
    total_bytes = 0
    for sibling in sorted(info.siblings, key=lambda item: item.rfilename):
        size = int(sibling.size or 0)
        files.append({"path": sibling.rfilename, "size_bytes": size})
        total_bytes += size
    if info.sha != LOCKED_MODEL_REVISION:
        raise ValueError(
            "Resolved model revision differs: {}".format(info.sha)
        )
    if total_bytes != LOCKED_MODEL_REMOTE_BYTES:
        raise ValueError(
            "Remote snapshot size differs: expected {}, got {}".format(
                LOCKED_MODEL_REMOTE_BYTES, total_bytes
            )
        )
    return {
        "repository": generator["model_repository"],
        "requested_revision": generator["model_revision"],
        "resolved_revision": info.sha,
        "total_bytes": total_bytes,
        "files": files,
        "passed": True,
    }


def _download_and_hash_snapshot(
    protocol: Mapping[str, object],
    cache_root: Path,
    remote: Mapping[str, object],
) -> Dict[str, object]:
    from huggingface_hub import snapshot_download

    generator = protocol["generator"]  # type: ignore[index]
    snapshot_path = Path(
        snapshot_download(
            repo_id=str(generator["model_repository"]),
            revision=str(generator["model_revision"]),
            cache_dir=str(cache_root),
            max_workers=2,
        )
    )
    files: List[Dict[str, object]] = []
    for remote_file in remote["files"]:  # type: ignore[index]
        relative = str(remote_file["path"])
        path = snapshot_path / Path(relative)
        if not path.is_file():
            raise FileNotFoundError(
                "Pinned snapshot file is missing: {}".format(relative)
            )
        size = path.stat().st_size
        if size != int(remote_file["size_bytes"]):
            raise ValueError(
                "Snapshot size mismatch for {}: {} != {}".format(
                    relative, size, remote_file["size_bytes"]
                )
            )
        files.append(
            {"path": relative, "size_bytes": size, "sha256": _sha256(path)}
        )
    total_bytes = sum(int(item["size_bytes"]) for item in files)
    return {
        "repository": generator["model_repository"],
        "revision": generator["model_revision"],
        "snapshot_path": str(snapshot_path.resolve()),
        "total_bytes": total_bytes,
        "files": files,
        "file_manifest_sha256": _canonical_sha256(files),
        "passed": total_bytes == LOCKED_MODEL_REMOTE_BYTES,
    }


def _pipeline_component_state(pipe: object) -> List[Dict[str, object]]:
    states: List[Dict[str, object]] = []
    for name, component in sorted(pipe.components.items()):  # type: ignore[attr-defined]
        device = None
        try:
            parameter = next(component.parameters())
            device = str(parameter.device)
        except (AttributeError, StopIteration):
            device = None
        states.append(
            {
                "name": name,
                "class": type(component).__name__,
                "device": device,
                "has_hf_hook": bool(getattr(component, "_hf_hook", None)),
            }
        )
    return states


def _enable_locked_vae_slicing(pipe: object) -> str:
    pipeline_hook = getattr(pipe, "enable_vae_slicing", None)
    if callable(pipeline_hook):
        pipeline_hook()
        return "pipeline.enable_vae_slicing"
    vae = getattr(pipe, "vae", None)
    component_hook = getattr(vae, "enable_slicing", None)
    if callable(component_hook):
        component_hook()
        return "pipeline.vae.enable_slicing"
    raise RuntimeError("Pipeline and VAE do not expose a slicing hook")


def _load_pipeline_no_output(
    snapshot_manifest: Mapping[str, object],
    *,
    allowed_process_ids: Optional[Set[int]] = None,
) -> Tuple[Dict[str, object], Dict[str, object]]:
    os.environ["HF_HUB_OFFLINE"] = "1"
    os.environ["TRANSFORMERS_OFFLINE"] = "1"
    os.environ["DIFFUSERS_OFFLINE"] = "1"
    import torch
    from diffusers import BlipDiffusionControlNetPipeline

    if not torch.cuda.is_available():
        raise RuntimeError("CUDA is unavailable in the isolated runtime")
    torch.cuda.empty_cache()
    torch.cuda.reset_peak_memory_stats()
    start = time.perf_counter()
    with ResourceSampler(
        interval_seconds=2.0,
        allowed_process_ids=allowed_process_ids,
    ) as sampler:
        pipe = BlipDiffusionControlNetPipeline.from_pretrained(
            str(snapshot_manifest["snapshot_path"]),
            torch_dtype=torch.float16,
            local_files_only=True,
            low_cpu_mem_usage=True,
            use_safetensors=False,
        )
        pipe.enable_model_cpu_offload()
        pipe.enable_attention_slicing("auto")
        vae_slicing_api = _enable_locked_vae_slicing(pipe)
        components = _pipeline_component_state(pipe)
        hook_count = sum(bool(item["has_hf_hook"]) for item in components)
        if hook_count == 0:
            raise RuntimeError("No Accelerate CPU-offload hook was installed")
        load_seconds = time.perf_counter() - start
        torch_allocated = int(torch.cuda.memory_allocated())
        torch_reserved = int(torch.cuda.memory_reserved())
        state = {
            "pipeline_class": type(pipe).__name__,
            "offload_mode": "enable_model_cpu_offload",
            "attention_slicing": "auto",
            "vae_slicing": True,
            "vae_slicing_api": vae_slicing_api,
            "weight_serialization": "pinned SHA-256 PyTorch .bin files",
            "component_states": components,
            "offload_hook_count": hook_count,
            "load_seconds": load_seconds,
            "torch_cuda_allocated_bytes_after_load": torch_allocated,
            "torch_cuda_reserved_bytes_after_load": torch_reserved,
            "pipeline_invoked": False,
            "synthetic_pixels_generated": False,
            "passed": True,
        }
        del pipe
        gc.collect()
        torch.cuda.empty_cache()
    telemetry = {
        "samples": sampler.samples,
        "summary": sampler.summary(),
    }
    return state, telemetry


def _run_pipeline_load_worker(args: argparse.Namespace) -> Dict[str, object]:
    if args.worker_snapshot_manifest is None or args.worker_result is None:
        raise ValueError(
            "Pipeline worker requires snapshot manifest and result paths"
        )
    started_unix = time.time()
    try:
        snapshot = _read_json(args.worker_snapshot_manifest)
        if not isinstance(snapshot, dict):
            raise TypeError("Worker snapshot manifest must be a JSON object")
        if int(snapshot.get("total_bytes", -1)) != LOCKED_MODEL_REMOTE_BYTES:
            raise ValueError("Worker snapshot byte count differs from the lock")
        pipeline, telemetry = _load_pipeline_no_output(
            snapshot,
            allowed_process_ids=set(args.worker_allowed_pid),
        )
        result = {
            "schema": PIPELINE_WORKER_SCHEMA,
            "passed": True,
            "started_unix": started_unix,
            "finished_unix": time.time(),
            "pipeline": pipeline,
            "resource_telemetry": telemetry,
            "pipeline_invoked": False,
            "synthetic_pixels_generated": False,
        }
        _write_json(args.worker_result, result)
        return result
    except Exception as exc:
        failure = {
            "schema": PIPELINE_WORKER_SCHEMA,
            "passed": False,
            "started_unix": started_unix,
            "finished_unix": time.time(),
            "error_type": type(exc).__name__,
            "error": str(exc),
            "traceback": traceback.format_exc(),
            "pipeline_invoked": False,
            "synthetic_pixels_generated": False,
        }
        _write_json(args.worker_result, failure)
        raise


def _load_pipeline_no_output_isolated(
    snapshot_manifest: Mapping[str, object],
    output_dir: Path,
    *,
    timeout_seconds: float,
) -> Dict[str, object]:
    allowed_parent_pids = sorted({os.getpid(), os.getppid()})
    with tempfile.TemporaryDirectory(
        prefix=".saspa_f0_pipeline_worker_",
        dir=str(output_dir),
    ) as temporary_directory:
        temporary_root = Path(temporary_directory)
        snapshot_path = temporary_root / "snapshot_manifest.json"
        result_path = temporary_root / "worker_result.json"
        _write_json(snapshot_path, dict(snapshot_manifest))
        command = [
            sys.executable,
            "-m",
            "trkh.tools.audit_saspa_synthetic_a0_f0",
            "--pipeline-load-worker",
            "--worker-snapshot-manifest",
            str(snapshot_path),
            "--worker-result",
            str(result_path),
        ]
        for process_id in allowed_parent_pids:
            command.extend(["--worker-allowed-pid", str(process_id)])
        environment = os.environ.copy()
        existing_python_path = environment.get("PYTHONPATH")
        environment["PYTHONPATH"] = str(REPO_ROOT)
        if existing_python_path:
            environment["PYTHONPATH"] += os.pathsep + existing_python_path
        environment["HF_HUB_OFFLINE"] = "1"
        environment["TRANSFORMERS_OFFLINE"] = "1"
        environment["DIFFUSERS_OFFLINE"] = "1"
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
            return {
                "schema": PIPELINE_WORKER_SCHEMA,
                "passed": False,
                "worker_process": {
                    "mode": "isolated_subprocess",
                    "python_executable": sys.executable,
                    "allowed_parent_pids": allowed_parent_pids,
                    "timeout_seconds": timeout_seconds,
                },
                "error_type": type(exc).__name__,
                "error": "Pipeline load worker exceeded the locked timeout",
                "pipeline_invoked": False,
                "synthetic_pixels_generated": False,
            }
        payload = _read_json(result_path) if result_path.is_file() else {}
        if not isinstance(payload, dict):
            payload = {}
        worker_process = {
            "mode": "isolated_subprocess",
            "python_executable": sys.executable,
            "allowed_parent_pids": allowed_parent_pids,
            "timeout_seconds": timeout_seconds,
            "return_code": completed.returncode,
        }
        payload["worker_process"] = worker_process
        if payload.get("schema") != PIPELINE_WORKER_SCHEMA:
            payload["passed"] = False
            payload["error"] = "Pipeline load worker returned an invalid schema"
        if completed.returncode != 0 or not payload.get("passed"):
            payload["passed"] = False
            payload.setdefault(
                "error",
                "Pipeline load worker failed without a structured error",
            )
            payload["stdout_tail"] = completed.stdout[-4000:]
            payload["stderr_tail"] = completed.stderr[-4000:]
        return payload


def _artifact_manifest(output_dir: Path, filename: str) -> Dict[str, object]:
    rows = []
    for path in sorted(output_dir.iterdir(), key=lambda item: item.name.lower()):
        if not path.is_file() or path.name == filename:
            continue
        rows.append(
            {
                "path": path.name,
                "size_bytes": path.stat().st_size,
                "sha256": _sha256(path),
            }
        )
    return {
        "schema": "trkh_saspa_a0_artifact_manifest_v1",
        "files": rows,
        "files_sha256": _canonical_sha256(rows),
    }


def run_formal_f0(args: argparse.Namespace) -> Dict[str, object]:
    output_dir = Path(args.output_dir)
    if output_dir.exists():
        raise FileExistsError(
            "Formal F0 output already exists; never overwrite: {}".format(
                output_dir
            )
        )
    output_dir.mkdir(parents=True)
    start_time = time.time()
    resource_gate_context: Dict[str, object] = {}
    resource_snapshot_context: Dict[str, object] = {}
    failure_context: Dict[str, object] = {
        "stage": "repo_state",
        "resource_gates": resource_gate_context,
        "resource_snapshots": resource_snapshot_context,
    }
    try:
        repo = _repo_state()
        failure_context["repo_state"] = repo
        if not repo["passed"]:
            raise RuntimeError("Formal F0 requires clean pushed implementation state")
        failure_context["stage"] = "static_inputs"
        static = verify_static_inputs(args.protocol, args.requirements)
        failure_context["static_inputs_passed"] = static["passed"]
        protocol = _read_json(args.protocol)
        if not isinstance(protocol, dict):
            raise TypeError("Protocol must be a JSON object")
        failure_context["stage"] = "start_resource_gate"
        start_resources = resource_snapshot()
        start_gates = evaluate_f0_resource_gates(protocol, start_resources)
        resource_snapshot_context["start"] = start_resources
        resource_gate_context["start"] = start_gates
        if not start_gates["passed"]:
            raise RuntimeError(
                "F0 start resource gates failed: {}".format(start_gates["checks"])
            )
        failure_context["stage"] = "environment_manifest"
        environment = _installed_runtime_manifest(
            args.requirements,
            hash_distribution_files=True,
            import_runtime=False,
        )
        failure_context["environment"] = {
            "passed": environment["passed"],
            "package_count": len(environment["packages"]),
        }
        if not environment["passed"]:
            raise RuntimeError("Isolated runtime manifest failed")
        failure_context["stage"] = "source_selection"
        selection = select_tiny_cohort(protocol)
        failure_context["source_selection"] = {
            "row_count": selection["row_count"],
            "selection_sha256": selection["selection_sha256"],
            "eligible_counts": selection["population"]["eligible_counts"],
        }
        if tuple(selection["population"]["eligible_counts"]) != EXPECTED_ELIGIBLE_COUNTS:
            raise RuntimeError(
                "Eligible population changed: {}".format(
                    selection["population"]["eligible_counts"]
                )
            )
        failure_context["stage"] = "remote_model_metadata"
        remote = _remote_model_metadata(protocol)
        failure_context["remote_model"] = {
            "revision": remote["resolved_revision"],
            "total_bytes": remote["total_bytes"],
        }
        failure_context["stage"] = "pre_download_resource_gate"
        pre_download_resources = resource_snapshot()
        size_gates = evaluate_f0_resource_gates(
            protocol,
            pre_download_resources,
            model_snapshot_bytes=int(remote["total_bytes"]),
        )
        resource_snapshot_context["pre_download"] = pre_download_resources
        resource_gate_context["pre_download"] = size_gates
        if not size_gates["passed"]:
            raise RuntimeError("Pre-download F0 gates failed")
        failure_context["stage"] = "snapshot_manifest"
        snapshot = _download_and_hash_snapshot(
            protocol, Path(args.cache_root), remote
        )
        failure_context["snapshot"] = {
            "passed": snapshot["passed"],
            "total_bytes": snapshot["total_bytes"],
            "file_manifest_sha256": snapshot["file_manifest_sha256"],
        }
        if not snapshot["passed"]:
            raise RuntimeError("Pinned model snapshot manifest failed")
        failure_context["stage"] = "pre_load_resource_gate"
        pre_load_resources = resource_snapshot()
        pre_load_gates = evaluate_f0_resource_gates(
            protocol,
            pre_load_resources,
            model_snapshot_bytes=int(snapshot["total_bytes"]),
        )
        resource_snapshot_context["pre_load"] = pre_load_resources
        resource_gate_context["pre_load"] = pre_load_gates
        if not pre_load_gates["passed"]:
            raise RuntimeError("Pre-load F0 resource gates failed")
        failure_context["stage"] = "isolated_pipeline_load"
        pipeline_timeout_seconds = (
            float(
                protocol["phase_f0_no_output"]["resource_gates"][
                    "pipeline_load_minutes_max"
                ]
            )
            * 60.0
            + 60.0
        )
        worker_result = _load_pipeline_no_output_isolated(
            snapshot,
            output_dir,
            timeout_seconds=pipeline_timeout_seconds,
        )
        failure_context["pipeline_worker"] = worker_result
        if not worker_result.get("passed"):
            raise RuntimeError(
                "Isolated pipeline load worker failed: {}".format(
                    worker_result.get("error", "unknown error")
                )
            )
        if worker_result.get("pipeline_invoked") or worker_result.get(
            "synthetic_pixels_generated"
        ):
            raise RuntimeError("Pipeline worker violated the F0 no-output boundary")
        pipeline = worker_result.get("pipeline")
        telemetry = worker_result.get("resource_telemetry")
        if not isinstance(pipeline, dict) or not isinstance(telemetry, dict):
            raise TypeError("Pipeline worker returned an invalid payload")
        if pipeline.get("pipeline_invoked") or pipeline.get(
            "synthetic_pixels_generated"
        ):
            raise RuntimeError("Pipeline state violated the F0 no-output boundary")
        pipeline["worker_process"] = worker_result["worker_process"]
        failure_context["pipeline_worker"] = {
            "passed": True,
            "worker_process": worker_result["worker_process"],
            "pipeline": pipeline,
            "resource_telemetry": telemetry,
        }
        failure_context["stage"] = "final_resource_gate"
        end_resources = resource_snapshot()
        final_gates = evaluate_f0_resource_gates(
            protocol,
            end_resources,
            model_snapshot_bytes=int(snapshot["total_bytes"]),
            pipeline_load_seconds=float(pipeline["load_seconds"]),
        )
        telemetry_virtual = telemetry["summary"]["max_virtual_used_fraction"]
        if telemetry_virtual is not None:
            final_gates["checks"]["telemetry_virtual_memory_fraction"] = (
                float(telemetry_virtual)
                < float(
                    protocol["phase_f0_no_output"]["resource_gates"][
                        "host_virtual_memory_fraction_max"
                    ]
                )
            )
            final_gates["passed"] = all(final_gates["checks"].values())
        telemetry_external_pids = telemetry["summary"].get(
            "external_python_or_trtexec_pids", []
        )
        final_gates["checks"][
            "telemetry_no_unknown_python_or_trtexec"
        ] = not bool(telemetry_external_pids)
        final_gates["passed"] = all(final_gates["checks"].values())
        resource_snapshot_context["final"] = end_resources
        resource_gate_context["final"] = final_gates
        if not final_gates["passed"]:
            raise RuntimeError(
                "Final F0 resource gates failed: {}".format(
                    final_gates["checks"]
                )
            )

        failure_context["stage"] = "artifact_write"
        shutil.copy2(args.protocol, output_dir / "locked_protocol.json")
        _write_json(output_dir / "static_inputs.json", static)
        _write_json(output_dir / "environment_manifest.json", environment)
        _write_json(output_dir / "source_selection.json", selection)
        _write_source_csv(output_dir / "source_manifest.csv", selection["rows"])
        _write_json(output_dir / "model_remote_metadata.json", remote)
        _write_json(output_dir / "model_snapshot_manifest.json", snapshot)
        _write_json(output_dir / "pipeline_load.json", pipeline)
        _write_json(output_dir / "resource_telemetry.json", telemetry)
        summary = {
            "schema": SCHEMA,
            "method": METHOD,
            "phase": "f0_no_output",
            "status": "Pass",
            "passed": True,
            "started_unix": start_time,
            "finished_unix": time.time(),
            "repo_state": repo,
            "static_inputs": {
                "passed": static["passed"],
                "protocol_sha256": static["files"]["protocol_json"]["sha256"],
                "requirements_sha256": static["files"]["requirements"]["sha256"],
            },
            "environment": {
                "passed": environment["passed"],
                "runtime_versions": environment["runtime_versions"],
                "package_manifest_sha256": _canonical_sha256(
                    environment["packages"]
                ),
            },
            "source_selection": {
                "row_count": selection["row_count"],
                "eligible_counts": selection["population"]["eligible_counts"],
                "selection_sha256": selection["selection_sha256"],
                "all_leakage_groups_unique": selection[
                    "all_leakage_groups_unique"
                ],
            },
            "model": {
                "repository": remote["repository"],
                "revision": remote["resolved_revision"],
                "snapshot_total_bytes": snapshot["total_bytes"],
                "snapshot_manifest_sha256": snapshot["file_manifest_sha256"],
            },
            "pipeline": pipeline,
            "resource_gates": {
                "start": start_gates,
                "pre_download": size_gates,
                "pre_load": pre_load_gates,
                "final": final_gates,
            },
            "dataset_boundary": {
                "raw_dataset_modified": False,
                "train_pixels_used_for_generation": False,
                "validation_test_pixels_opened": False,
                "validation_test_labels_opened": False,
            },
            "pipeline_invoked": False,
            "synthetic_pixels_generated": False,
            "f1_authorized": True,
            "claims_authorized": [
                "Only F0 runtime, source-selection, model-snapshot, and pipeline-load feasibility passed."
            ],
            "claims_not_authorized": protocol["claims_not_authorized"],
            "current_best_command_updated": False,
        }
        _write_json(output_dir / "summary.json", summary)
        manifest = _artifact_manifest(output_dir, "manifest.json")
        _write_json(output_dir / "manifest.json", manifest)
        result = {
            "summary_path": str((output_dir / "summary.json").resolve()),
            "summary_sha256": _sha256(output_dir / "summary.json"),
            "manifest_path": str((output_dir / "manifest.json").resolve()),
            "manifest_sha256": _sha256(output_dir / "manifest.json"),
            "selection_sha256": selection["selection_sha256"],
            "passed": True,
            "synthetic_pixels_generated": False,
        }
        print(json.dumps(result, indent=2, sort_keys=True))
        return result
    except Exception as exc:
        failure = {
            "schema": "trkh_saspa_a0_f0_failure_v1",
            "method": METHOD,
            "phase": "f0_no_output",
            "status": "FailClosed",
            "passed": False,
            "started_unix": start_time,
            "finished_unix": time.time(),
            "error_type": type(exc).__name__,
            "error": str(exc),
            "traceback": traceback.format_exc(),
            "failure_stage": failure_context.get("stage"),
            "failure_context": failure_context,
            "pipeline_invoked": False,
            "synthetic_pixels_generated": False,
            "f1_authorized": False,
        }
        _write_json(output_dir / "failure.json", failure)
        raise


def replay_summary(summary_path: Path) -> Dict[str, object]:
    summary_path = Path(summary_path).resolve()
    output_dir = summary_path.parent
    summary = _read_json(summary_path)
    if not isinstance(summary, dict):
        raise TypeError("Summary must be a JSON object")
    manifest_path = output_dir / "manifest.json"
    manifest = _read_json(manifest_path)
    failures: List[str] = []
    if not isinstance(manifest, dict):
        failures.append("manifest_not_object")
        manifest = {"files": []}
    for row in manifest["files"]:
        path = output_dir / str(row["path"])
        if not path.is_file():
            failures.append("missing:{}".format(row["path"]))
            continue
        if path.stat().st_size != int(row["size_bytes"]):
            failures.append("size:{}".format(row["path"]))
        if _sha256(path) != str(row["sha256"]):
            failures.append("sha256:{}".format(row["path"]))
    if summary.get("schema") != SCHEMA or not summary.get("passed"):
        failures.append("summary_status")
    if summary.get("pipeline_invoked") or summary.get("synthetic_pixels_generated"):
        failures.append("f0_output_boundary")
    boundary = summary.get("dataset_boundary", {})
    if boundary.get("validation_test_pixels_opened") or boundary.get(
        "validation_test_labels_opened"
    ):
        failures.append("validation_test_boundary")
    static = verify_static_inputs(PROTOCOL_PATH, REQUIREMENTS_PATH)
    protocol = _read_json(PROTOCOL_PATH)
    selection = select_tiny_cohort(protocol)  # type: ignore[arg-type]
    expected_selection = summary["source_selection"]["selection_sha256"]
    if selection["selection_sha256"] != expected_selection:
        failures.append("source_selection")
    source_artifact = _read_json(output_dir / "source_selection.json")
    if (
        not isinstance(source_artifact, dict)
        or source_artifact.get("selection_sha256") != expected_selection
    ):
        failures.append("source_artifact")
    result = {
        "schema": "trkh_saspa_a0_f0_replay_v1",
        "summary_path": str(summary_path),
        "summary_sha256": _sha256(summary_path),
        "manifest_sha256": _sha256(manifest_path),
        "static_inputs_passed": static["passed"],
        "selection_sha256": selection["selection_sha256"],
        "failures": failures,
        "passed": not failures,
        "pipeline_invoked": False,
        "synthetic_pixels_generated": False,
        "validation_test_pixels_opened": False,
    }
    _write_json(output_dir / "replay.json", result)
    final_manifest = _artifact_manifest(output_dir, "final_manifest.json")
    _write_json(output_dir / "final_manifest.json", final_manifest)
    print(json.dumps(result, indent=2, sort_keys=True))
    if failures:
        raise RuntimeError("F0 replay failed: {}".format(failures))
    return result


def main(argv: Optional[Sequence[str]] = None) -> None:
    args = parse_args(argv)
    if args.preflight_only:
        result = verify_static_inputs(args.protocol, args.requirements)
        protocol = _read_json(args.protocol)
        selection = select_tiny_cohort(protocol)  # type: ignore[arg-type]
        result["source_selection"] = {
            "eligible_counts": selection["population"]["eligible_counts"],
            "row_count": selection["row_count"],
            "selection_sha256": selection["selection_sha256"],
        }
        result["synthetic_pixels_generated"] = False
        print(json.dumps(result, indent=2, sort_keys=True))
    elif args.runtime_check_only:
        result = _installed_runtime_manifest(
            args.requirements,
            hash_distribution_files=False,
            import_runtime=True,
        )
        display = {
            "schema": result["schema"],
            "passed": result["passed"],
            "python_executable": result["python_executable"],
            "runtime_versions": result["runtime_versions"],
            "runtime_import": result["runtime_import"],
            "isolation": result["isolation"],
            "requirements": result["requirements"],
            "package_count": len(result["packages"]),
            "pip_check": result["pip_check"],
            "mismatches": result["mismatches"],
        }
        print(json.dumps(display, indent=2, sort_keys=True))
        if not result["passed"]:
            raise RuntimeError("Runtime check failed")
    elif args.pipeline_load_worker:
        _run_pipeline_load_worker(args)
    elif args.formal_f0:
        run_formal_f0(args)
    else:
        replay_summary(args.replay_summary)


if __name__ == "__main__":
    main()
