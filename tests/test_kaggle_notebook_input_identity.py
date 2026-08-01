from __future__ import annotations

import ast
import json
from pathlib import Path

import pytest


PROJECT_ROOT = Path(__file__).resolve().parents[1]
NOTEBOOK_PATH = PROJECT_ROOT / "kagle" / "TRKH_CLASSF_BEST_KAGGLE.ipynb"
V4_CONTRACT = "TRKH_KAGGLE_B2_T4_OFFLINE_V4_20260801"


def _notebook() -> dict[str, object]:
    return json.loads(NOTEBOOK_PATH.read_text(encoding="utf-8"))


def _resolver(input_root: Path):
    notebook = _notebook()
    cell_source = "".join(notebook["cells"][1]["source"])
    parsed = ast.parse(cell_source)
    functions = [
        node
        for node in parsed.body
        if isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef))
        and node.name in {"kaggle_input_identity", "kaggle_input_mount"}
    ]
    assert {node.name for node in functions} == {
        "kaggle_input_identity",
        "kaggle_input_mount",
    }
    module = ast.fix_missing_locations(ast.Module(body=functions, type_ignores=[]))
    namespace = {
        "Path": Path,
        "INPUT_ROOT": input_root,
        "KAGGLE_DATASET_NAMESPACE": "datasets",
        "KAGGLE_DATASET_RESOURCE_DEPTH": 3,
    }
    exec(compile(module, str(NOTEBOOK_PATH), "exec"), namespace)
    return namespace["kaggle_input_identity"], namespace["kaggle_input_mount"]


def test_notebook_v4_contract_and_all_cells_compile() -> None:
    notebook = _notebook()
    assert notebook["metadata"]["trkh_release"]["notebook_contract"] == V4_CONTRACT
    assert len(notebook["cells"]) == 12
    for index, cell in enumerate(notebook["cells"]):
        assert cell["cell_type"] == "code"
        compile("".join(cell["source"]), f"notebook-cell-{index}", "exec")


def test_current_namespaced_datasets_resolve_by_owner_and_slug(tmp_path: Path) -> None:
    input_root = tmp_path / "kaggle" / "input"
    identity, mount = _resolver(input_root)
    asset = input_root / "datasets" / "alice" / "code01" / "nested" / "manifest.json"
    data = input_root / "datasets" / "alice" / "data05" / "wrapper" / "data.yaml"

    assert mount(asset) == (input_root / "datasets" / "alice" / "code01").resolve()
    assert mount(data) == (input_root / "datasets" / "alice" / "data05").resolve()
    assert mount(asset) != mount(data)
    assert identity(asset)["layout"] == "namespaced_dataset_owner_slug"
    assert identity(asset)["resource_parts"] == ["datasets", "alice", "code01"]


def test_legacy_top_level_slug_remains_supported(tmp_path: Path) -> None:
    input_root = tmp_path / "kaggle" / "input"
    identity, mount = _resolver(input_root)
    anchor = input_root / "code01" / "nested" / "manifest.json"

    assert mount(anchor) == (input_root / "code01").resolve()
    assert identity(anchor)["layout"] == "legacy_top_level_slug"


def test_same_namespaced_dataset_is_still_detectable(tmp_path: Path) -> None:
    input_root = tmp_path / "kaggle" / "input"
    _, mount = _resolver(input_root)
    asset = input_root / "datasets" / "alice" / "combined" / "assets" / "manifest.json"
    data = input_root / "datasets" / "alice" / "combined" / "data" / "data.yaml"

    assert mount(asset) == mount(data)


def test_incomplete_or_external_paths_fail_closed(tmp_path: Path) -> None:
    input_root = tmp_path / "kaggle" / "input"
    identity, _ = _resolver(input_root)

    with pytest.raises(RuntimeError, match="thieu owner/slug"):
        identity(input_root / "datasets" / "alice")
    with pytest.raises(RuntimeError, match="nam ngoai"):
        identity(tmp_path / "outside" / "data.yaml")
