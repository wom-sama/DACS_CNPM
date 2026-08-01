from __future__ import annotations

import argparse
import hashlib
import importlib.metadata
import json
import os
import re
import subprocess
import sys
import zipfile
from pathlib import Path, PurePosixPath
from typing import Iterable


SOURCE_COMMIT = "7f7f0883cbb71b6a5620fee86c15b996c400a813"
SOURCE_TREE_SHA256 = "7c8752efe6acb728ed913abe5db165b218c94177e3ef13f5b37ce1c62b19d965"
DINO_SHA256 = "2a1ec16ae28ffa07bc0ead0241ee7df9fc26451fe6f9f839b7b3afa0a906b040"
DINO_BYTES = 86_362_376
DINO_REVISION = "3bf4720a82ec2066db88137180ff1f83a675cef0"
DINOV3_LICENSE_SHA256 = "25d122eb8f5b880fd23c736fb6ea8018ee45c12237e00b8a86d14c653904999e"
TIMM_VERSION = "1.0.27"
PROTOCOL_ID = "TRKH_PRETRAINED_CLASSF_B2_TEMPERED_P05_20260731"
EXPERIMENT_KEY = "b2-tempered-p05"
NOTEBOOK_CONTRACT = "TRKH_KAGGLE_B2_T4_OFFLINE_V3_20260801"
V1_NOTEBOOK_SHA256 = "e154a00394cdbbffe19f98d542cd8d94a01df3d3ac25ffec81b7cb16055186ba"
RELEASE_BASENAME = "TRKH_KAGGLE_B2_T4_UPLOAD_BUNDLE_20260801.zip"
FIXED_ZIP_TIMESTAMP = (2026, 8, 1, 0, 0, 0)

FOCUSED_TESTS = (
    "tests/test_pretrained_semantic_branch.py",
    "tests/test_timm_classifier_model.py",
    "tests/test_canonical_classf_defaults.py",
    "tests/test_deploy_classification_folder.py",
    "tests/test_pretrained_classf_recipe.py",
    "tests/test_resume_weight_and_distillation_source.py",
    "tests/test_attention_viz_headless.py",
)


def sha256_bytes(payload: bytes) -> str:
    return hashlib.sha256(payload).hexdigest()


def sha256_file(path: Path, chunk_size: int = 1024 * 1024) -> str:
    digest = hashlib.sha256()
    with Path(path).open("rb") as handle:
        for chunk in iter(lambda: handle.read(chunk_size), b""):
            digest.update(chunk)
    return digest.hexdigest()


def run_git(repo: Path, *arguments: str) -> bytes:
    safe_directory = Path(repo).resolve().as_posix()
    return subprocess.run(
        ["git", "-c", f"safe.directory={safe_directory}", "-C", str(repo), *arguments],
        check=True,
        capture_output=True,
    ).stdout


def git_file(repo: Path, revision: str, relative_path: str) -> bytes:
    if not re.fullmatch(r"[0-9a-f]{40}", revision):
        raise RuntimeError(f"Unsafe git revision: {revision!r}")
    normalized = PurePosixPath(relative_path).as_posix()
    if normalized.startswith("/") or ".." in PurePosixPath(normalized).parts:
        raise RuntimeError(f"Unsafe git path: {relative_path!r}")
    return run_git(repo, "show", f"{revision}:{normalized}")


def exact_source_payload(pretrained_repo: Path) -> dict[str, bytes]:
    commit_type = run_git(pretrained_repo, "cat-file", "-t", SOURCE_COMMIT).decode().strip()
    if commit_type != "commit":
        raise RuntimeError(f"Missing locked commit {SOURCE_COMMIT}: {commit_type!r}")
    names = run_git(
        pretrained_repo,
        "ls-tree",
        "-r",
        "--name-only",
        "-z",
        SOURCE_COMMIT,
    ).decode("utf-8").split("\0")
    names = [name for name in names if name]
    selected = sorted(
        name
        for name in names
        if (name.startswith("trkh/") and name.endswith(".py"))
        or (name.startswith("configs/") and name.endswith(".yaml"))
        or name in FOCUSED_TESTS
        or name == "tests/conftest.py"
        or name == "pytest.ini"
    )
    missing_tests = sorted(set(FOCUSED_TESTS) - set(selected))
    if missing_tests:
        raise RuntimeError(f"Locked source commit misses focused tests: {missing_tests}")
    payload = {
        f"TRKH_pretrained/{name}": git_file(pretrained_repo, SOURCE_COMMIT, name)
        for name in selected
    }
    payload["TRKH_pretrained/SOURCE_COMMIT.txt"] = (SOURCE_COMMIT + "\n").encode("utf-8")

    digest = hashlib.sha256()
    source_names = sorted(
        name for name in selected if name.startswith("trkh/") and name.endswith(".py")
    ) + sorted(
        name for name in selected if name.startswith("configs/") and name.endswith(".yaml")
    )
    for name in source_names:
        data = payload[f"TRKH_pretrained/{name}"]
        digest.update(name.encode("utf-8") + b"\0")
        digest.update(data.replace(b"\r\n", b"\n"))
    observed = digest.hexdigest()
    if observed != SOURCE_TREE_SHA256:
        raise RuntimeError(
            "Locked source tree digest drifted: "
            f"observed={observed}, expected={SOURCE_TREE_SHA256}"
        )
    return payload


def vendored_timm_payload() -> tuple[dict[str, bytes], str]:
    distribution = importlib.metadata.distribution("timm")
    if distribution.version != TIMM_VERSION:
        raise RuntimeError(
            f"Release builder needs local timm {TIMM_VERSION}; observed {distribution.version}."
        )
    site_root = Path(distribution.locate_file("")).resolve()
    dist_info_prefix = f"timm-{TIMM_VERSION}.dist-info/"
    payload: dict[str, bytes] = {}
    for entry in distribution.files or ():
        relative = PurePosixPath(str(entry).replace("\\", "/"))
        relative_text = relative.as_posix()
        if not (
            relative_text.startswith("timm/")
            or relative_text.startswith(dist_info_prefix)
        ):
            continue
        if "__pycache__" in relative.parts or relative.suffix == ".pyc":
            continue
        source = Path(distribution.locate_file(entry)).resolve()
        try:
            source.relative_to(site_root)
        except ValueError as error:
            raise RuntimeError(f"Vendored timm file escapes site-packages: {source}") from error
        if not source.is_file():
            continue
        if source.suffix.lower() in {".so", ".pyd", ".dll", ".dylib"}:
            raise RuntimeError(f"timm is no longer pure Python: {relative_text}")
        payload[f"vendor/python/{relative_text}"] = source.read_bytes()
    required = {
        "vendor/python/timm/__init__.py",
        f"vendor/python/{dist_info_prefix}METADATA",
        f"vendor/python/{dist_info_prefix}licenses/LICENSE",
    }
    missing = sorted(required - set(payload))
    if missing:
        raise RuntimeError(f"Incomplete timm vendor payload: {missing}")
    tree_digest = hashlib.sha256()
    for name, data in sorted(payload.items()):
        tree_digest.update(name.encode("utf-8") + b"\0")
        tree_digest.update(data)
    return payload, tree_digest.hexdigest()


def dino_from_zip(zip_path: Path) -> bytes | None:
    try:
        with zipfile.ZipFile(zip_path, "r") as archive:
            candidates = [
                name
                for name in archive.namelist()
                if PurePosixPath(name).as_posix() == "weights/model.safetensors"
            ]
            if len(candidates) != 1:
                return None
            info = archive.getinfo(candidates[0])
            if info.file_size != DINO_BYTES:
                return None
            payload = archive.read(candidates[0])
    except (OSError, zipfile.BadZipFile, KeyError):
        return None
    return payload if sha256_bytes(payload) == DINO_SHA256 else None


def resolve_dino_weight(kaggle_dir: Path, explicit_path: Path | None) -> bytes:
    if explicit_path is not None:
        explicit_path = explicit_path.resolve()
        if not explicit_path.is_file():
            raise FileNotFoundError(explicit_path)
        if explicit_path.stat().st_size != DINO_BYTES:
            raise RuntimeError(f"Wrong DINO size: {explicit_path}")
        payload = explicit_path.read_bytes()
        if sha256_bytes(payload) != DINO_SHA256:
            raise RuntimeError(f"Wrong DINO hash: {explicit_path}")
        return payload

    cache_root = Path.home() / ".cache" / "huggingface" / "hub"
    cache_glob = (
        "models--timm--vit_small_patch16_dinov3.lvd1689m/"
        "snapshots/*/model.safetensors"
    )
    for candidate in sorted(cache_root.glob(cache_glob)) if cache_root.is_dir() else ():
        if candidate.stat().st_size == DINO_BYTES and sha256_file(candidate) == DINO_SHA256:
            return candidate.read_bytes()

    zip_candidates = sorted(
        kaggle_dir.glob("TRKH_KAGGLE_*UPLOAD_BUNDLE_*.zip"),
        key=lambda path: (path.stat().st_mtime_ns, path.name),
        reverse=True,
    )
    for candidate in zip_candidates:
        payload = dino_from_zip(candidate)
        if payload is not None:
            return payload
    raise FileNotFoundError(
        "Cannot locate the locked DINOv3 weight. Pass --dino-weight or retain one prior "
        "TRKH Kaggle asset ZIP until the new release is built."
    )


def runtime_target_document(timm_tree_sha256: str) -> bytes:
    document = {
        "schema_version": 1,
        "contract": NOTEBOOK_CONTRACT,
        "target": {
            "platform": "Kaggle Notebooks GPU",
            "official_gpu_release": "v170",
            "official_git_commit": "fc61d5cda7da39530055bae9bd0e92865f995cd9",
            "official_private_image_digest": "bdf9e0538555f90453619adefb49ba40cfa136db44a9c9be7a42ea715c0aa068",
            "python_minor": "3.12",
            "torch_family": "2.10.x+cu128",
            "torchvision_family": "0.25.x+cu128",
            "preferred_accelerator": "NvidiaTeslaT4",
            "minimum_compute_capability": [7, 5],
            "p100_supported": False,
            "fp16_grad_scaler_init_scale": 1024.0,
        },
        "dependency_policy": {
            "mutates_global_environment": False,
            "pip_install": False,
            "vendored_timm": TIMM_VERSION,
            "vendored_timm_tree_sha256": timm_tree_sha256,
            "uses_kaggle_torch_torchvision_cuda_stack": True,
            "uses_kaggle_tqdm_environment_override": True,
        },
        "input_layout": {
            "asset_modes": ["archive_file", "kaggle_mounted_expanded"],
            "dataset_modes": ["archive_file", "kaggle_mounted_expanded"],
            "separate_kaggle_dataset_mounts_required": True,
        },
        "sources": {
            "docker_release": (
                "https://github.com/Kaggle/docker-python/releases/tag/"
                "v170-GPU-bdf9e0538555f90453619adefb49ba40cfa136db44a9c9be7a42ea715c0aa068"
            ),
            "kaggle_accelerator_warning": (
                "https://github.com/Kaggle/kaggle-cli/blob/main/docs/kernels.md"
                "#kaggle-kernels-push"
            ),
        },
    }
    return (json.dumps(document, ensure_ascii=False, indent=2) + "\n").encode("utf-8")


def dino_provenance_document(license_sha256: str) -> bytes:
    document = {
        "schema_version": 1,
        "model": "timm/vit_small_patch16_dinov3.lvd1689m",
        "source_url": "https://huggingface.co/timm/vit_small_patch16_dinov3.lvd1689m",
        "revision": DINO_REVISION,
        "weight_path": "weights/model.safetensors",
        "weight_size_bytes": DINO_BYTES,
        "weight_sha256": DINO_SHA256,
        "license": "DINOv3 License",
        "license_path": "third_party/dinov3/LICENSE.md",
        "license_sha256": license_sha256,
        "distribution_note": (
            "The DINOv3 Agreement is included with the redistributed checkpoint. "
            "Research publications using the checkpoint must acknowledge DINOv3."
        ),
    }
    return (json.dumps(document, ensure_ascii=False, indent=2) + "\n").encode("utf-8")


def zip_info(name: str) -> zipfile.ZipInfo:
    normalized = PurePosixPath(name).as_posix()
    if normalized.startswith("/") or ".." in PurePosixPath(normalized).parts:
        raise RuntimeError(f"Unsafe release path: {name!r}")
    info = zipfile.ZipInfo(normalized, FIXED_ZIP_TIMESTAMP)
    info.compress_type = zipfile.ZIP_DEFLATED
    info.create_system = 3
    info.external_attr = (0o100644 & 0xFFFF) << 16
    return info


def file_inventory(payload: dict[str, bytes]) -> list[dict[str, object]]:
    return [
        {
            "path": name,
            "size_bytes": len(data),
            "sha256": sha256_bytes(data),
        }
        for name, data in sorted(payload.items())
    ]


def validate_built_zip(path: Path) -> dict[str, object]:
    with zipfile.ZipFile(path, "r") as archive:
        bad_crc = archive.testzip()
        if bad_crc is not None:
            raise RuntimeError(f"ZIP CRC failed: {bad_crc}")
        names = archive.namelist()
        if len(names) != len(set(name.casefold() for name in names)):
            raise RuntimeError("ZIP contains duplicate/case-colliding paths")
        manifest = json.loads(archive.read("TRKH_KAGGLE_B2_UPLOAD_MANIFEST.json"))
        declared = {entry["path"]: entry for entry in manifest["files"]}
        actual_names = set(names) - {"TRKH_KAGGLE_B2_UPLOAD_MANIFEST.json"}
        if set(declared) != actual_names:
            raise RuntimeError("ZIP manifest inventory does not match archive members")
        for name, entry in declared.items():
            data = archive.read(name)
            if len(data) != entry["size_bytes"] or sha256_bytes(data) != entry["sha256"]:
                raise RuntimeError(f"ZIP manifest digest failed: {name}")
    return {
        "path": str(path),
        "size_bytes": path.stat().st_size,
        "sha256": sha256_file(path),
        "members": len(names),
        "manifest_files": len(declared),
    }


def build_release(
    *,
    project_root: Path,
    pretrained_repo: Path,
    output_path: Path,
    dino_weight: Path | None,
    replace_existing: bool = False,
) -> dict[str, object]:
    kaggle_dir = project_root / "kagle"
    notebook_path = kaggle_dir / "TRKH_CLASSF_BEST_KAGGLE.ipynb"
    readme_path = kaggle_dir / "TRKH_CLASSF_BEST_KAGGLE_README.txt"
    license_path = kaggle_dir / "vendor" / "DINOV3_LICENSE.md"
    for required in (notebook_path, readme_path, license_path):
        if not required.is_file():
            raise FileNotFoundError(required)
    notebook_bytes = notebook_path.read_bytes()
    notebook = json.loads(notebook_bytes)
    cells = notebook.get("cells", [])
    code_cells = [cell for cell in cells if cell.get("cell_type") == "code"]
    if len(cells) != 12 or len(code_cells) != 12:
        raise RuntimeError(f"Notebook must contain exactly 12 code cells: {notebook_path}")
    for index, cell in enumerate(code_cells):
        compile("".join(cell.get("source", [])), f"{notebook_path.name}:cell-{index}", "exec")
    joined_source = "\n".join("".join(cell.get("source", [])) for cell in code_cells)
    required_notebook_tokens = (
        NOTEBOOK_CONTRACT,
        f'EXPECTED_SOURCE_COMMIT = "{SOURCE_COMMIT}"',
        f'EXPECTED_SOURCE_TREE_SHA256 = "{SOURCE_TREE_SHA256}"',
        "import math",
        "AMP_INIT_SCALE = 1024.0",
        "amp_init_scale=AMP_INIT_SCALE",
        "str(VENDORED_TIMM_ROOT) + os.pathsep + str(REPO_ROOT)",
        "key=lambda path: path.relative_to(REPO_ROOT).as_posix()",
        'SMOKE_STATE_CHECKPOINT = SMOKE_DIR / "checkpoints" / "last.pt"',
        '"train_optimizer_updates_successful"',
        '"optimizer_update_contract": SMOKE_OPTIMIZER_CONTRACT',
        "discover_expanded_asset_root",
        '"input_mode": "kaggle_mounted_expanded"',
        "Asset va dataset phai la hai Kaggle Dataset inputs rieng",
        'os.environ["TQDM_DISABLE"] = "1"',
        "TQDM_DISABLE=1 was not honored",
        '"epoch_metric_logs_retained": True',
        '"asset_input": ASSET_INPUT_CONTRACT',
        '"dataset_input": DATASET_INPUT_CONTRACT',
    )
    missing_notebook_tokens = [
        token for token in required_notebook_tokens if token not in joined_source
    ]
    if missing_notebook_tokens:
        raise RuntimeError(
            "Notebook does not satisfy the locked source/runtime/smoke contract: "
            f"missing={missing_notebook_tokens}"
        )
    forbidden_notebook_tokens = (
        "ensure_locked_packages(",
        '"-m", "pip", "install"',
        "Attach exactly one V2 asset ZIP",
    )
    retained_forbidden = [
        token for token in forbidden_notebook_tokens if token in joined_source
    ]
    if retained_forbidden:
        raise RuntimeError(f"Notebook retains forbidden dependency mutation: {retained_forbidden}")
    release_metadata = notebook.get("metadata", {}).get("trkh_release", {})
    if release_metadata.get("notebook_contract") != NOTEBOOK_CONTRACT:
        raise RuntimeError(f"Notebook release metadata contract drifted: {release_metadata}")
    if release_metadata.get("source_notebook_sha256") != V1_NOTEBOOK_SHA256:
        raise RuntimeError(f"Notebook V1 generator source drifted: {release_metadata}")

    license_bytes = license_path.read_bytes()
    license_hash = sha256_bytes(license_bytes)
    if license_hash != DINOV3_LICENSE_SHA256:
        raise RuntimeError(
            f"DINOv3 license drifted: observed={license_hash}, expected={DINOV3_LICENSE_SHA256}"
        )

    payload = exact_source_payload(pretrained_repo)
    locked_train_source = payload["TRKH_pretrained/trkh/training/train.py"].decode("utf-8")
    required_train_tokens = (
        "OPTIMIZER_TELEMETRY_HISTORY_FIELDS = (",
        "*OPTIMIZER_TELEMETRY_HISTORY_FIELDS,",
        'GradScaler(\n            "cuda",\n            init_scale=float(train_config.amp_init_scale),',
    )
    missing_train_tokens = [
        token for token in required_train_tokens if token not in locked_train_source
    ]
    if missing_train_tokens:
        raise RuntimeError(
            "Locked training source cannot satisfy the Kaggle optimizer evidence gate: "
            f"missing={missing_train_tokens}"
        )
    timm_payload, timm_tree_hash = vendored_timm_payload()
    payload.update(timm_payload)
    payload.update(
        {
            "TRKH_CLASSF_BEST_KAGGLE.ipynb": notebook_bytes,
            "TRKH_CLASSF_BEST_KAGGLE_README.txt": readme_path.read_bytes(),
            "weights/model.safetensors": resolve_dino_weight(kaggle_dir, dino_weight),
            "third_party/dinov3/LICENSE.md": license_bytes,
            "third_party/dinov3/PROVENANCE.json": dino_provenance_document(license_hash),
            "TRKH_KAGGLE_RUNTIME_TARGET.json": runtime_target_document(timm_tree_hash),
        }
    )
    if len(payload["weights/model.safetensors"]) != DINO_BYTES:
        raise RuntimeError("DINO weight size drifted while staging")
    if sha256_bytes(payload["weights/model.safetensors"]) != DINO_SHA256:
        raise RuntimeError("DINO weight digest drifted while staging")
    if len(payload) != len(set(name.casefold() for name in payload)):
        raise RuntimeError("Payload contains duplicate/case-colliding paths")

    manifest = {
        "schema_version": 3,
        "notebook_contract": NOTEBOOK_CONTRACT,
        "protocol": PROTOCOL_ID,
        "experiment": EXPERIMENT_KEY,
        "source_commit": SOURCE_COMMIT,
        "source_tree_sha256": SOURCE_TREE_SHA256,
        "source_snapshot": "minimal files read from exact git commit",
        "dino_sha256": DINO_SHA256,
        "dinov3_license_sha256": license_hash,
        "vendored_timm_version": TIMM_VERSION,
        "vendored_timm_tree_sha256": timm_tree_hash,
        "offline_dependency_bootstrap": True,
        "input_layout_contract": "separate_archive_or_kaggle_mounted_expanded_v1",
        "compact_progress_default": True,
        "kaggle_runtime_target": "v170 GPU / Python 3.12 / Torch 2.10 cu128 / T4",
        "files": file_inventory(payload),
    }
    manifest_bytes = (json.dumps(manifest, ensure_ascii=False, indent=2) + "\n").encode("utf-8")

    output_path = output_path.resolve()
    output_path.parent.mkdir(parents=True, exist_ok=True)
    temporary_path = output_path.with_name(output_path.name + ".partial")
    if temporary_path.exists():
        raise FileExistsError(
            f"Refusing to overwrite an incomplete prior build: {temporary_path}"
        )
    with zipfile.ZipFile(
        temporary_path,
        "x",
        compression=zipfile.ZIP_DEFLATED,
        compresslevel=6,
        allowZip64=True,
    ) as archive:
        for name, data in sorted(payload.items()):
            archive.writestr(zip_info(name), data)
        archive.writestr(zip_info("TRKH_KAGGLE_B2_UPLOAD_MANIFEST.json"), manifest_bytes)
    if output_path.exists() and not replace_existing:
        temporary_path.unlink(missing_ok=True)
        raise FileExistsError(f"Refusing to overwrite release: {output_path}")
    os.replace(temporary_path, output_path)
    return validate_built_zip(output_path)


def parse_args(argv: Iterable[str] | None = None) -> argparse.Namespace:
    script_path = Path(__file__).resolve()
    default_root = script_path.parents[1]
    parser = argparse.ArgumentParser(
        description="Build the deterministic offline Kaggle B2/T4 asset bundle."
    )
    parser.add_argument("--project-root", type=Path, default=default_root)
    parser.add_argument(
        "--pretrained-repo",
        type=Path,
        default=default_root.parent / "TRKH_pretrained",
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=default_root / "kagle" / RELEASE_BASENAME,
    )
    parser.add_argument("--dino-weight", type=Path, default=None)
    parser.add_argument(
        "--replace-existing",
        action="store_true",
        help="Atomically replace an existing generated release after external validation.",
    )
    return parser.parse_args(list(argv) if argv is not None else None)


def main(argv: Iterable[str] | None = None) -> int:
    args = parse_args(argv)
    result = build_release(
        project_root=args.project_root.resolve(),
        pretrained_repo=args.pretrained_repo.resolve(),
        output_path=args.output,
        dino_weight=args.dino_weight,
        replace_existing=bool(args.replace_existing),
    )
    print(json.dumps(result, ensure_ascii=False, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
