#!/usr/bin/env python3
"""Upgrade the reviewed TRKH Kaggle B2/T4 V3 notebook to V4.

V4 understands both Kaggle's legacy ``/kaggle/input/<slug>`` layout and the
current ``/kaggle/input/datasets/<owner>/<slug>`` layout.  V3 treated the
shared ``datasets`` namespace as the resource mount, so two genuinely separate
dataset inputs were incorrectly reported as one input.
"""

from __future__ import annotations

import argparse
import copy
import hashlib
import json
import os
from pathlib import Path
from typing import Mapping


V3_CONTRACT = "TRKH_KAGGLE_B2_T4_OFFLINE_V3_20260801"
V4_CONTRACT = "TRKH_KAGGLE_B2_T4_OFFLINE_V4_20260801"
V3_RUN_TAG = "kaggle_b2_tempered_p05_v3"
V4_RUN_TAG = "kaggle_b2_tempered_p05_v4"


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


OLD_MOUNT_RESOLVER = '''def kaggle_input_mount(path):
    root = INPUT_ROOT.resolve()
    resolved = Path(path).resolve()
    try:
        relative = resolved.relative_to(root)
    except ValueError as error:
        raise RuntimeError(f"Input path nam ngoai /kaggle/input: {resolved}") from error
    if not relative.parts:
        raise RuntimeError(f"Khong xac dinh duoc Kaggle Dataset mount cho: {resolved}")
    return (root / relative.parts[0]).resolve()
'''


NEW_MOUNT_RESOLVER = '''KAGGLE_DATASET_NAMESPACE = "datasets"
KAGGLE_DATASET_RESOURCE_DEPTH = 3  # datasets/<owner>/<slug>

def kaggle_input_identity(path):
    root = INPUT_ROOT.resolve()
    resolved = Path(path).resolve()
    try:
        relative = resolved.relative_to(root)
    except ValueError as error:
        raise RuntimeError(f"Input path nam ngoai /kaggle/input: {resolved}") from error
    parts = relative.parts
    if not parts:
        raise RuntimeError(f"Khong xac dinh duoc Kaggle Dataset mount cho: {resolved}")
    if parts[0].casefold() == KAGGLE_DATASET_NAMESPACE:
        if len(parts) < KAGGLE_DATASET_RESOURCE_DEPTH:
            raise RuntimeError(
                "Kaggle dataset path thieu owner/slug: "
                f"{resolved}; expected /kaggle/input/datasets/<owner>/<slug>/..."
            )
        resource_parts = parts[:KAGGLE_DATASET_RESOURCE_DEPTH]
        layout = "namespaced_dataset_owner_slug"
    else:
        resource_parts = parts[:1]
        layout = "legacy_top_level_slug"
    resource_root = (root / Path(*resource_parts)).resolve()
    return {
        "layout": layout,
        "resource_root": str(resource_root),
        "resource_parts": list(resource_parts),
    }

def kaggle_input_mount(path):
    return Path(kaggle_input_identity(path)["resource_root"])
'''


def upgrade_notebook(notebook: Mapping[str, object]) -> dict[str, object]:
    upgraded = copy.deepcopy(dict(notebook))
    cells = upgraded.get("cells", [])
    if len(cells) != 12 or any(cell.get("cell_type") != "code" for cell in cells):
        raise ValueError("Reviewed V3 must contain exactly 12 code cells")

    config = _source(cells[0])
    if V4_CONTRACT in config:
        _validate_v4(upgraded)
        return upgraded
    if V3_CONTRACT not in config:
        raise ValueError("Input notebook is not the reviewed B2/T4 V3 contract")
    config = _replace_once(config, V3_CONTRACT, V4_CONTRACT, "notebook contract")
    config = _replace_once(config, V3_RUN_TAG, V4_RUN_TAG, "V4 run tag")

    asset = _source(cells[1])
    asset = _replace_once(
        asset,
        OLD_MOUNT_RESOLVER,
        NEW_MOUNT_RESOLVER,
        "Kaggle resource identity resolver",
    )
    asset = _replace_once(
        asset,
        "ASSET_INPUT_MOUNT = kaggle_input_mount(ASSET_INPUT_ANCHOR)\n",
        "ASSET_INPUT_IDENTITY = kaggle_input_identity(ASSET_INPUT_ANCHOR)\n"
        "ASSET_INPUT_MOUNT = Path(ASSET_INPUT_IDENTITY[\"resource_root\"])\n"
        "ASSET_ZIP_CONTRACT = {\n"
        "    **ASSET_ZIP_CONTRACT,\n"
        "    \"kaggle_input_identity\": ASSET_INPUT_IDENTITY,\n"
        "}\n",
        "asset input identity",
    )

    runtime = _source(cells[2])
    runtime = _replace_once(
        runtime,
        "B2 V3 requires one manifest-covered asset input",
        "B2 V4 requires one manifest-covered asset input",
        "runtime V4 message",
    )

    data = _source(cells[3])
    data = _replace_once(
        data,
        "DATASET_INPUT_MOUNT = kaggle_input_mount(DATASET_INPUT_ANCHOR)\n"
        "if DATASET_INPUT_MOUNT == ASSET_INPUT_MOUNT:\n",
        "DATASET_INPUT_IDENTITY = kaggle_input_identity(DATASET_INPUT_ANCHOR)\n"
        "DATASET_INPUT_MOUNT = Path(DATASET_INPUT_IDENTITY[\"resource_root\"])\n"
        "if DATASET_INPUT_MOUNT == ASSET_INPUT_MOUNT:\n",
        "dataset input identity",
    )
    data = _replace_once(
        data,
        "DATASET_INPUT_CONTRACT = DATASET_ZIP_CONTRACT\n",
        "DATASET_INPUT_CONTRACT = {\n"
        "    **DATASET_ZIP_CONTRACT,\n"
        "    \"kaggle_input_identity\": DATASET_INPUT_IDENTITY,\n"
        "}\n",
        "dataset identity contract",
    )

    cells[0] = _set_source(cells[0], config)
    cells[1] = _set_source(cells[1], asset)
    cells[2] = _set_source(cells[2], runtime)
    cells[3] = _set_source(cells[3], data)
    upgraded["cells"] = cells

    metadata = copy.deepcopy(upgraded.get("metadata", {}))
    release = copy.deepcopy(metadata.get("trkh_release", {}))
    release.update(
        {
            "notebook_contract": V4_CONTRACT,
            "input_layout_contract": (
                "separate_archive_or_kaggle_mounted_expanded_v2_owner_slug"
            ),
            "v4_upgrade_generator": Path(__file__).name,
        }
    )
    metadata["trkh_release"] = release
    upgraded["metadata"] = metadata
    _validate_v4(upgraded)
    return upgraded


def _validate_v4(notebook: Mapping[str, object]) -> None:
    cells = notebook.get("cells", [])
    if len(cells) != 12 or any(cell.get("cell_type") != "code" for cell in cells):
        raise ValueError("V4 notebook must contain exactly 12 code cells")
    joined = "\n".join(_source(cell) for cell in cells)
    required = (
        V4_CONTRACT,
        V4_RUN_TAG,
        "KAGGLE_DATASET_RESOURCE_DEPTH = 3",
        "namespaced_dataset_owner_slug",
        "legacy_top_level_slug",
        "ASSET_INPUT_IDENTITY = kaggle_input_identity(ASSET_INPUT_ANCHOR)",
        "DATASET_INPUT_IDENTITY = kaggle_input_identity(DATASET_INPUT_ANCHOR)",
        '"kaggle_input_identity": ASSET_INPUT_IDENTITY',
        '"kaggle_input_identity": DATASET_INPUT_IDENTITY',
    )
    missing = [token for token in required if token not in joined]
    if missing:
        raise AssertionError(f"V4 notebook misses required tokens: {missing}")
    if OLD_MOUNT_RESOLVER in joined:
        raise AssertionError("V4 notebook retains the V3 first-component resolver")
    for index, cell in enumerate(cells):
        compile(_source(cell), f"TRKH_CLASSF_BEST_KAGGLE.ipynb:cell-{index}", "exec")
    release = notebook.get("metadata", {}).get("trkh_release", {})
    if release.get("notebook_contract") != V4_CONTRACT:
        raise AssertionError(f"V4 release metadata drifted: {release}")


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
    temporary = output_path.with_name(output_path.name + ".v4.partial")
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
                "contract": V4_CONTRACT,
            },
            ensure_ascii=False,
            indent=2,
        )
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
