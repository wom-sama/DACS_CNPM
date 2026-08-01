#!/usr/bin/env python3
"""Upgrade the reviewed TRKH Kaggle B2/T4 V2 notebook to V3.

V3 accepts either an intact archive or Kaggle's mounted expanded tree for
each input, while retaining the same manifest/hash gates.  It also suppresses
TTY-style tqdm bars by default because Kaggle renders their carriage returns
as many notebook-output lines.
"""

from __future__ import annotations

import argparse
import copy
import hashlib
import json
import os
from pathlib import Path
from typing import Mapping


V2_CONTRACT = "TRKH_KAGGLE_B2_T4_OFFLINE_V2_20260731"
V3_CONTRACT = "TRKH_KAGGLE_B2_T4_OFFLINE_V3_20260801"
V3_RUN_TAG = "kaggle_b2_tempered_p05_v3"
EXPECTED_VENDORED_TIMM_TREE_SHA256 = (
    "c3f21c6f3ef4ac466d46e33494c726e4d80ce3329e73b537f68a47fe066dc856"
)
EXPECTED_DINOV3_LICENSE_SHA256 = (
    "25d122eb8f5b880fd23c736fb6ea8018ee45c12237e00b8a86d14c653904999e"
)


def _source(cell: Mapping[str, object]) -> str:
    value = cell.get("source", [])
    return value if isinstance(value, str) else "".join(str(part) for part in value)


def _set_source(cell: Mapping[str, object], source: str) -> dict[str, object]:
    updated = copy.deepcopy(dict(cell))
    updated["source"] = source.splitlines(keepends=True)
    return updated


def _replace_once(source: str, old: str, new: str, label: str) -> str:
    if source.count(old) != 1:
        raise ValueError(
            f"Expected exactly one {label} anchor, observed {source.count(old)}"
        )
    return source.replace(old, new, 1)


ASSET_RESOLUTION_V3 = r'''def discover_expanded_asset_root():
    manifests = sorted({
        path.resolve()
        for path in INPUT_ROOT.rglob(ASSET_MANIFEST_NAME)
        if path.is_file()
    })
    if len(manifests) > 1:
        raise RuntimeError(f"Can dung toi da 1 expanded asset manifest, tim thay: {manifests}")
    return manifests[0].parent if manifests else None

def kaggle_input_mount(path):
    root = INPUT_ROOT.resolve()
    resolved = Path(path).resolve()
    try:
        relative = resolved.relative_to(root)
    except ValueError as error:
        raise RuntimeError(f"Input path nam ngoai /kaggle/input: {resolved}") from error
    if not relative.parts:
        raise RuntimeError(f"Khong xac dinh duoc Kaggle Dataset mount cho: {resolved}")
    return (root / relative.parts[0]).resolve()

ASSET_ZIP = discover_optional_zip(ASSET_ZIP_GLOB, "asset bundle")
ASSET_EXPANDED_ROOT = discover_expanded_asset_root()
if ASSET_ZIP is not None and ASSET_EXPANDED_ROOT is not None:
    raise RuntimeError(
        "Asset input mo ho: dong thoi co ZIP va expanded manifest. "
        "Chi attach mot representation."
    )
DATASET_ZIP = discover_optional_zip(DATASET_ZIP_GLOB, "dataset")
if ASSET_ZIP is not None:
    ASSET_EXTRACT_ROOT, ASSET_ZIP_CONTRACT = safe_extract_zip(ASSET_ZIP, "assets")
    ASSET_ZIP_CONTRACT = {
        **ASSET_ZIP_CONTRACT,
        "input_mode": "archive_file",
    }
    ASSET_INPUT_ANCHOR = ASSET_ZIP
elif ASSET_EXPANDED_ROOT is not None:
    ASSET_EXTRACT_ROOT = ASSET_EXPANDED_ROOT.resolve()
    ASSET_ZIP_CONTRACT = {
        "schema_version": 1,
        "status": "resolved_pending_manifest",
        "label": "assets",
        "input_mode": "kaggle_mounted_expanded",
        "root": str(ASSET_EXTRACT_ROOT),
    }
    ASSET_INPUT_ANCHOR = ASSET_EXTRACT_ROOT / ASSET_MANIFEST_NAME
else:
    raise RuntimeError(
        f"Attach exactly one asset input: ZIP matching {ASSET_ZIP_GLOB} "
        f"or an expanded tree containing {ASSET_MANIFEST_NAME}"
    )
DATASET_EXTRACT_ROOT, DATASET_ZIP_CONTRACT = safe_extract_zip(
    DATASET_ZIP,
    "dataset",
    forbidden_components={"test"},
)
if DATASET_ZIP_CONTRACT is not None:
    DATASET_ZIP_CONTRACT = {
        **DATASET_ZIP_CONTRACT,
        "input_mode": "archive_file",
    }
ASSET_INPUT_MOUNT = kaggle_input_mount(ASSET_INPUT_ANCHOR)
'''


DATASET_EXPANDED_V3 = r'''
if any(str(key).casefold() == "test" for key in DATA_DOCUMENT):
    raise RuntimeError("Uploaded data.yaml khong duoc khai bao test")
'''


DATASET_INPUT_V3 = r'''
DATASET_INPUT_ANCHOR = DATASET_ZIP if DATASET_ZIP is not None else DATA_YAML
DATASET_INPUT_MOUNT = kaggle_input_mount(DATASET_INPUT_ANCHOR)
if DATASET_INPUT_MOUNT == ASSET_INPUT_MOUNT:
    raise RuntimeError(
        "Asset va dataset phai la hai Kaggle Dataset inputs rieng; "
        f"khong gop chung trong {ASSET_INPUT_MOUNT}"
    )
if DATASET_ZIP is None:
    invalid_dataset_entries = [
        str(path)
        for path in DATASET_INPUT_MOUNT.rglob("*")
        if path.is_symlink() or not (path.is_file() or path.is_dir())
    ]
    if invalid_dataset_entries:
        raise RuntimeError(
            f"Expanded dataset chua symlink/special entries: {invalid_dataset_entries[:20]}"
        )
    forbidden_test_paths = [
        str(path)
        for path in DATASET_INPUT_MOUNT.rglob("*")
        if any(part.casefold() == "test" for part in path.relative_to(DATASET_INPUT_MOUNT).parts)
    ]
    if forbidden_test_paths:
        raise RuntimeError(
            f"Expanded dataset khong duoc chua test component: {forbidden_test_paths[:20]}"
        )
    DATASET_ZIP_CONTRACT = {
        "schema_version": 1,
        "status": "passed",
        "label": "dataset",
        "input_mode": "kaggle_mounted_expanded",
        "root": str(DATASET_INPUT_MOUNT),
        "data_yaml": str(DATA_YAML),
        "data_yaml_sha256": sha256(DATA_YAML),
    }
DATASET_INPUT_CONTRACT = DATASET_ZIP_CONTRACT
DATASET_CONTENT_ROOT = (
    DATASET_EXTRACT_ROOT.resolve()
    if DATASET_EXTRACT_ROOT is not None
    else DATASET_INPUT_MOUNT
)
'''


def upgrade_notebook(notebook: Mapping[str, object]) -> dict[str, object]:
    upgraded = copy.deepcopy(dict(notebook))
    cells = upgraded.get("cells", [])
    if len(cells) != 12 or any(cell.get("cell_type") != "code" for cell in cells):
        raise ValueError("Reviewed V2 must contain exactly 12 code cells")

    config = _source(cells[0])
    if V3_CONTRACT in config:
        _validate_v3(upgraded)
        return upgraded
    if V2_CONTRACT not in config:
        raise ValueError("Input notebook is not the reviewed B2/T4 V2 contract")
    config = _replace_once(config, V2_CONTRACT, V3_CONTRACT, "notebook contract")
    config = _replace_once(
        config,
        'RUN_TAG = "kaggle_b2_tempered_p05_v2"',
        f'RUN_TAG = "{V3_RUN_TAG}"',
        "V3 run tag",
    )
    config = _replace_once(
        config,
        "CONFIRM_FULL = False  # Chi doi True sau khi doc metrics probe.\n",
        "CONFIRM_FULL = False  # Chi doi True sau khi doc metrics probe.\n"
        "KAGGLE_COMPACT_PROGRESS = True  # Tat tqdm dong; van giu metric log moi epoch.\n"
        "if KAGGLE_COMPACT_PROGRESS:\n"
        "    os.environ[\"TQDM_DISABLE\"] = \"1\"\n"
        "else:\n"
        "    os.environ.pop(\"TQDM_DISABLE\", None)\n",
        "compact progress switch",
    )
    config = _replace_once(
        config,
        "DINO_BYTES = 86362376\n",
        "DINO_BYTES = 86362376\n"
        f'EXPECTED_VENDORED_TIMM_TREE_SHA256 = "{EXPECTED_VENDORED_TIMM_TREE_SHA256}"\n'
        f'EXPECTED_DINOV3_LICENSE_SHA256 = "{EXPECTED_DINOV3_LICENSE_SHA256}"\n',
        "hardcoded redistributed-asset hashes",
    )
    config = _replace_once(
        config,
        'print(PROTOCOL_ID, "auto_resume=", AUTO_RESUME, "run_probe=", RUN_PROBE)\n',
        'print(PROTOCOL_ID, "auto_resume=", AUTO_RESUME, "run_probe=", RUN_PROBE, '
        '"compact_progress=", KAGGLE_COMPACT_PROGRESS)\n',
        "configuration telemetry",
    )

    asset = _source(cells[1])
    old_resolution = '''ASSET_ZIP = discover_optional_zip(ASSET_ZIP_GLOB, "asset bundle")
if ASSET_ZIP is None:
    raise RuntimeError(f"Attach exactly one V2 asset ZIP matching {ASSET_ZIP_GLOB}")
DATASET_ZIP = discover_optional_zip(DATASET_ZIP_GLOB, "dataset")
ASSET_EXTRACT_ROOT, ASSET_ZIP_CONTRACT = safe_extract_zip(ASSET_ZIP, "assets")
DATASET_EXTRACT_ROOT, DATASET_ZIP_CONTRACT = safe_extract_zip(
    DATASET_ZIP,
    "dataset",
    forbidden_components={"test"},
)
'''
    asset = _replace_once(asset, old_resolution, ASSET_RESOLUTION_V3, "input resolution")
    asset = _replace_once(
        asset,
        '        "vendored_timm_version": "1.0.27",\n',
        '        "vendored_timm_version": "1.0.27",\n'
        '        "vendored_timm_tree_sha256": EXPECTED_VENDORED_TIMM_TREE_SHA256,\n'
        '        "dinov3_license_sha256": EXPECTED_DINOV3_LICENSE_SHA256,\n',
        "manifest immutable asset hashes",
    )
    asset = _replace_once(
        asset,
        "    actual_files = {\n",
        "    invalid_asset_entries = [\n"
        "        str(path)\n"
        "        for path in ASSET_EXTRACT_ROOT.rglob(\"*\")\n"
        "        if path.is_symlink() or not (path.is_file() or path.is_dir())\n"
        "    ]\n"
        "    if invalid_asset_entries:\n"
        "        raise RuntimeError(\n"
        "            f\"Asset input chua symlink/special entries: {invalid_asset_entries[:20]}\"\n"
        "        )\n"
        "    actual_files = {\n",
        "expanded asset filesystem guard",
    )
    asset = _replace_once(
        asset,
        '        "license_sha256": asset_manifest.get("dinov3_license_sha256"),\n',
        '        "license_sha256": EXPECTED_DINOV3_LICENSE_SHA256,\n',
        "hardcoded DINO license provenance",
    )
    asset = _replace_once(
        asset,
        '    if sha256(license_path) != asset_manifest.get("dinov3_license_sha256"):\n'
        '        provenance_mismatches["license_file_sha256"] = sha256(license_path)\n',
        '    if sha256(license_path) != EXPECTED_DINOV3_LICENSE_SHA256:\n'
        '        provenance_mismatches["license_file_sha256"] = sha256(license_path)\n',
        "hardcoded DINO license file check",
    )
    asset = _replace_once(
        asset,
        '    if observed_timm_tree_sha256 != asset_manifest.get("vendored_timm_tree_sha256"):\n'
        '        raise RuntimeError(\n'
        '            "Vendored timm tree digest mismatch: "\n'
        '            f"{observed_timm_tree_sha256} != {asset_manifest.get(\'vendored_timm_tree_sha256\')}"\n'
        '        )\n',
        '    if observed_timm_tree_sha256 != EXPECTED_VENDORED_TIMM_TREE_SHA256:\n'
        '        raise RuntimeError(\n'
        '            "Vendored timm tree digest mismatch: "\n'
        '            f"{observed_timm_tree_sha256} != {EXPECTED_VENDORED_TIMM_TREE_SHA256}"\n'
        '        )\n'
        '    ASSET_ZIP_CONTRACT = {\n'
        '        **ASSET_ZIP_CONTRACT,\n'
        '        "status": "passed",\n'
        '        "manifest_path": str(manifest_path),\n'
        '        "manifest_sha256": sha256(manifest_path),\n'
        '        "manifest_files": len(declared_files),\n'
        '    }\n'
        '    ASSET_INPUT_CONTRACT = ASSET_ZIP_CONTRACT\n',
        "hardcoded timm tree and resolved asset contract",
    )

    runtime = _source(cells[2])
    runtime = _replace_once(
        runtime,
        'raise RuntimeError("B2 V2 requires the manifest-covered asset ZIP")',
        'raise RuntimeError("B2 V3 requires one manifest-covered asset input")',
        "runtime asset input message",
    )
    runtime = _replace_once(runtime, "import platform\n", "import io\nimport platform\n", "io import")
    runtime = _replace_once(
        runtime,
        '    "torchvision": "torchvision",\n',
        '    "torchvision": "torchvision",\n    "tqdm": "tqdm",\n',
        "tqdm required module",
    )
    runtime = _replace_once(
        runtime,
        "import timm\n",
        "import timm\nimport tqdm as tqdm_module\nfrom tqdm import tqdm as tqdm_progress\n",
        "tqdm imports",
    )
    runtime = _replace_once(
        runtime,
        'runtime_target = RUNTIME_TARGET["target"]\n',
        '_progress_sink = io.StringIO()\n'
        '_progress_probe = tqdm_progress(total=0, file=_progress_sink)\n'
        'if KAGGLE_COMPACT_PROGRESS and not bool(_progress_probe.disable):\n'
        '    raise RuntimeError("TQDM_DISABLE=1 was not honored by the Kaggle tqdm runtime")\n'
        '_progress_probe.close()\n'
        'if KAGGLE_COMPACT_PROGRESS and _progress_sink.getvalue():\n'
        '    raise RuntimeError("Compact progress probe unexpectedly emitted a dynamic bar")\n'
        'runtime_target = RUNTIME_TARGET["target"]\n',
        "compact progress runtime probe",
    )
    runtime = _replace_once(
        runtime,
        '        "pytest", "safetensors", "scikit-learn", "pandas", "onnx", "onnxruntime",\n',
        '        "pytest", "safetensors", "tqdm", "scikit-learn", "pandas", "onnx", "onnxruntime",\n',
        "tqdm distribution inventory",
    )
    runtime = _replace_once(
        runtime,
        '    "safetensors": str(Path(safetensors.__file__).resolve()),\n',
        '    "safetensors": str(Path(safetensors.__file__).resolve()),\n'
        '    "tqdm": str(Path(tqdm_module.__file__).resolve()),\n',
        "tqdm module inventory",
    )
    runtime = _replace_once(
        runtime,
        '    "dependency_install_performed": False,\n',
        '    "dependency_install_performed": False,\n'
        '    "progress": {\n'
        '        "compact": bool(KAGGLE_COMPACT_PROGRESS),\n'
        '        "tqdm_disable": os.environ.get("TQDM_DISABLE"),\n'
        '        "tqdm_probe_disabled": bool(_progress_probe.disable),\n'
        '        "epoch_metric_logs_retained": True,\n'
        '    },\n',
        "runtime progress evidence",
    )
    runtime = _replace_once(
        runtime,
        '        for key in ("KAGGLE_KERNEL_RUN_TYPE", "KAGGLE_URL_BASE", "CUDA_VISIBLE_DEVICES")\n',
        '        for key in ("KAGGLE_KERNEL_RUN_TYPE", "KAGGLE_URL_BASE", "CUDA_VISIBLE_DEVICES", "TQDM_DISABLE")\n',
        "progress environment inventory",
    )

    data = _source(cells[3])
    data = _replace_once(
        data,
        'if not data_document_compatible(DATA_DOCUMENT):\n'
        '    raise RuntimeError("DATA_YAML_OVERRIDE khong dung schema 5 lop/class order B2")\n',
        'if not data_document_compatible(DATA_DOCUMENT):\n'
        '    raise RuntimeError("DATA_YAML_OVERRIDE khong dung schema 5 lop/class order B2")\n'
        + DATASET_EXPANDED_V3,
        "general uploaded test-key guard",
    )
    data = _replace_once(
        data,
        '    if not DATA_FROM_DATASET_ZIP or DATA_YAML != zip_data_yamls[0]:\n'
        '        raise RuntimeError("Khi attach dataset ZIP, DATA_YAML phai nam trong ZIP do")\n',
        '    if not DATA_FROM_DATASET_ZIP or DATA_YAML != zip_data_yamls[0]:\n'
        '        raise RuntimeError("Khi attach dataset ZIP, DATA_YAML phai nam trong ZIP do")\n'
        + DATASET_INPUT_V3,
        "dataset input mode and mount isolation",
    )
    data = _replace_once(
        data,
        "DATA_ROOT = valid_roots[0]\n",
        "DATA_ROOT = valid_roots[0]\n"
        "for split_name in (\"train\", \"val\"):\n"
        "    resolved_split = split_path_for_root(DATA_ROOT, DATA_DOCUMENT[split_name])\n"
        "    if resolved_split is None or not resolved_split.is_relative_to(DATASET_CONTENT_ROOT):\n"
        "        raise RuntimeError(\n"
        "            f\"{split_name} escapes the selected dataset input: {resolved_split}\"\n"
        "        )\n",
        "dataset split containment",
    )
    data = data.replace("Asset ZIP layout/source selection mismatch", "Asset input layout/source selection mismatch")
    data = _replace_once(
        data,
        'print("asset_zip =", ASSET_ZIP, "extract_root =", ASSET_EXTRACT_ROOT)\n'
        'print("dataset_zip =", DATASET_ZIP, "extract_root =", DATASET_EXTRACT_ROOT)\n',
        'print("asset_input =", ASSET_INPUT_CONTRACT, "root =", ASSET_EXTRACT_ROOT)\n'
        'print("dataset_input =", DATASET_INPUT_CONTRACT, "content_root =", DATASET_CONTENT_ROOT)\n',
        "input provenance prints",
    )

    dataset = _source(cells[4])
    dataset = _replace_once(
        dataset,
        '    "input_mode": "dataset_zip" if DATA_FROM_DATASET_ZIP else "already_extracted",\n'
        '    "dataset_zip": DATASET_ZIP_CONTRACT,\n',
        '    "input_mode": DATASET_INPUT_CONTRACT["input_mode"],\n'
        '    "dataset_input": DATASET_INPUT_CONTRACT,\n',
        "dataset input contract",
    )

    training = _source(cells[6])
    training = _replace_once(
        training,
        '    result = subprocess.run(command, cwd=REPO_ROOT, env=os.environ.copy(), check=False)\n',
        '    child_env = os.environ.copy()\n'
        '    if KAGGLE_COMPACT_PROGRESS:\n'
        '        child_env["TQDM_DISABLE"] = "1"\n'
        '    result = subprocess.run(command, cwd=REPO_ROOT, env=child_env, check=False)\n',
        "compact subprocess progress environment",
    )

    for index in (6, 11):
        source = training if index == 6 else _source(cells[index])
        source = source.replace('"asset_zip": ASSET_ZIP_CONTRACT', '"asset_input": ASSET_INPUT_CONTRACT')
        source = source.replace('"dataset_zip": DATASET_ZIP_CONTRACT', '"dataset_input": DATASET_INPUT_CONTRACT')
        if index == 6:
            training = source
        else:
            cells[index] = _set_source(cells[index], source)

    cells[0] = _set_source(cells[0], config)
    cells[1] = _set_source(cells[1], asset)
    cells[2] = _set_source(cells[2], runtime)
    cells[3] = _set_source(cells[3], data)
    cells[4] = _set_source(cells[4], dataset)
    cells[6] = _set_source(cells[6], training)
    upgraded["cells"] = cells
    metadata = copy.deepcopy(upgraded.get("metadata", {}))
    release = copy.deepcopy(metadata.get("trkh_release", {}))
    release.update(
        {
            "notebook_contract": V3_CONTRACT,
            "input_layout_contract": "separate_archive_or_kaggle_mounted_expanded_v1",
            "compact_progress_default": True,
            "v3_upgrade_generator": Path(__file__).name,
        }
    )
    metadata["trkh_release"] = release
    upgraded["metadata"] = metadata
    _validate_v3(upgraded)
    return upgraded


def _validate_v3(notebook: Mapping[str, object]) -> None:
    cells = notebook.get("cells", [])
    if len(cells) != 12 or any(cell.get("cell_type") != "code" for cell in cells):
        raise ValueError("V3 notebook must contain exactly 12 code cells")
    joined = "\n".join(_source(cell) for cell in cells)
    required = (
        V3_CONTRACT,
        f'RUN_TAG = "{V3_RUN_TAG}"',
        "discover_expanded_asset_root",
        '"input_mode": "kaggle_mounted_expanded"',
        "Asset va dataset phai la hai Kaggle Dataset inputs rieng",
        "EXPECTED_VENDORED_TIMM_TREE_SHA256",
        "EXPECTED_DINOV3_LICENSE_SHA256",
        'os.environ["TQDM_DISABLE"] = "1"',
        "TQDM_DISABLE=1 was not honored",
        '"epoch_metric_logs_retained": True',
        '"asset_input": ASSET_INPUT_CONTRACT',
        '"dataset_input": DATASET_INPUT_CONTRACT',
    )
    missing = [token for token in required if token not in joined]
    if missing:
        raise AssertionError(f"V3 notebook misses required tokens: {missing}")
    forbidden = (
        "Attach exactly one V2 asset ZIP",
        '"input_mode": "dataset_zip" if DATA_FROM_DATASET_ZIP else "already_extracted"',
        'env=os.environ.copy(), check=False',
    )
    retained = [token for token in forbidden if token in joined]
    if retained:
        raise AssertionError(f"V3 notebook retains obsolete behavior: {retained}")
    for index, cell in enumerate(cells):
        compile(_source(cell), f"TRKH_CLASSF_BEST_KAGGLE.ipynb:cell-{index}", "exec")
    release = notebook.get("metadata", {}).get("trkh_release", {})
    if release.get("notebook_contract") != V3_CONTRACT:
        raise AssertionError(f"V3 release metadata drifted: {release}")


def parse_args() -> argparse.Namespace:
    repo_root = Path(__file__).resolve().parents[2]
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--input",
        type=Path,
        default=repo_root / "kagle" / "TRKH_CLASSF_BEST_KAGGLE.ipynb",
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=repo_root / "kagle" / "TRKH_CLASSF_BEST_KAGGLE.ipynb",
    )
    parser.add_argument("--force", action="store_true")
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    input_path = args.input.resolve()
    output_path = args.output.resolve()
    if not input_path.is_file():
        raise FileNotFoundError(input_path)
    if output_path.exists() and output_path != input_path and not args.force:
        raise FileExistsError(output_path)
    raw = input_path.read_bytes()
    notebook = json.loads(raw.decode("utf-8"))
    upgraded = upgrade_notebook(notebook)
    payload = (json.dumps(upgraded, ensure_ascii=False, indent=1) + "\n").encode("utf-8")
    temporary = output_path.with_name(output_path.name + ".v3.partial")
    if temporary.exists():
        raise FileExistsError(temporary)
    temporary.write_bytes(payload)
    os.replace(temporary, output_path)
    print(
        json.dumps(
            {
                "path": str(output_path),
                "bytes": len(payload),
                "sha256": hashlib.sha256(payload).hexdigest(),
                "code_cells": len(upgraded["cells"]),
                "contract": V3_CONTRACT,
            },
            indent=2,
        )
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
