#!/usr/bin/env python3
"""Build the fail-closed TRKH Kaggle B2 V2 notebook from the reviewed V1.

This generator intentionally uses only the Python standard library.  It does
not build or mutate the upload bundle; the release packager must put the
generated notebook, the locked TRKH source, DINO weights, and the vendored
``timm-1.0.27`` tree into one manifest-covered asset ZIP.
"""

from __future__ import annotations

import argparse
import copy
import hashlib
import json
import os
from pathlib import Path
from textwrap import indent


V1_CONTRACT = "TRKH_KAGGLE_TWO_ZIP_B2_V1_20260731"
V2_CONTRACT = "TRKH_KAGGLE_B2_T4_OFFLINE_V2_20260731"
V2_NOTEBOOK_NAME = "TRKH_CLASSF_BEST_KAGGLE.ipynb"
V2_ASSET_MANIFEST_NAME = "TRKH_KAGGLE_B2_UPLOAD_MANIFEST.json"
LOCKED_SOURCE_COMMIT = "7f7f0883cbb71b6a5620fee86c15b996c400a813"
LOCKED_SOURCE_TREE_SHA256 = "7c8752efe6acb728ed913abe5db165b218c94177e3ef13f5b37ce1c62b19d965"


RUNTIME_CELL = r'''# Runtime contract: assets are already extracted and hash-verified above.
# Do not pip-install/downgrade Kaggle's live CUDA environment.
from importlib.metadata import PackageNotFoundError, version
import platform
import sysconfig

if ASSET_EXTRACT_ROOT is None:
    raise RuntimeError("B2 V2 requires the manifest-covered asset ZIP")
if platform.system() != "Linux":
    raise RuntimeError(f"Kaggle v170 contract requires Linux, observed {platform.system()}")
if sys.version_info[:2] != (3, 12):
    raise RuntimeError(
        "Kaggle v170 contract requires Python 3.12; "
        f"observed {sys.version.split()[0]}. Select a current Kaggle GPU image."
    )
if not INPUT_ROOT.is_dir() or not WORK_ROOT.is_dir():
    raise RuntimeError(f"Missing Kaggle roots: input={INPUT_ROOT}, working={WORK_ROOT}")

VENDORED_TIMM_ROOT = (ASSET_EXTRACT_ROOT / "vendor" / "python").resolve()
VENDORED_TIMM_PACKAGE = VENDORED_TIMM_ROOT / "timm"
if not (VENDORED_TIMM_PACKAGE / "__init__.py").is_file():
    raise RuntimeError(f"Asset ZIP is missing vendored timm 1.0.27: {VENDORED_TIMM_PACKAGE}")
sys.path.insert(0, str(VENDORED_TIMM_ROOT))

required_modules = {
    "numpy": "numpy",
    "PIL": "Pillow",
    "yaml": "PyYAML",
    "matplotlib": "matplotlib",
    "pytest": "pytest",
    "safetensors": "safetensors",
    "torch": "torch",
    "torchvision": "torchvision",
}
missing_modules = [
    f"{distribution} ({module})"
    for module, distribution in required_modules.items()
    if importlib.util.find_spec(module) is None
]
if missing_modules:
    raise RuntimeError(
        "Kaggle v170 base image is missing required modules; do not mutate torch/CUDA in-place: "
        + ", ".join(missing_modules)
    )

import numpy as np
import PIL
import matplotlib
import pytest
import safetensors
import torch
import torchvision
from torchvision.ops import nms
import yaml
import timm

observed_timm_root = Path(timm.__file__).resolve()
if not observed_timm_root.is_relative_to(VENDORED_TIMM_ROOT):
    raise RuntimeError(f"Imported non-vendored timm: {observed_timm_root}")
if str(getattr(timm, "__version__", "")) != "1.0.27":
    raise RuntimeError(f"Vendored timm version drift: {getattr(timm, '__version__', None)!r}")
runtime_target = RUNTIME_TARGET["target"]
if not str(torch.__version__).startswith("2.10."):
    raise RuntimeError(f"Kaggle v170 torch family drift: {torch.__version__!r}")
if not str(torchvision.__version__).startswith("0.25."):
    raise RuntimeError(f"Kaggle v170 torchvision family drift: {torchvision.__version__!r}")
if str(torch.version.cuda) != "12.8":
    raise RuntimeError(f"Kaggle v170 CUDA family drift: {torch.version.cuda!r}")
if not torch.cuda.is_available():
    raise RuntimeError("Enable the T4 GPU accelerator in Kaggle Settings")
if torch.cuda.device_count() < 1:
    raise RuntimeError("CUDA reported available but exposed no device")

gpu_inventory = []
for device_index in range(torch.cuda.device_count()):
    device_properties = torch.cuda.get_device_properties(device_index)
    gpu_inventory.append({
        "index": device_index,
        "name": device_properties.name,
        "compute_capability": [device_properties.major, device_properties.minor],
        "total_memory_bytes": int(device_properties.total_memory),
        "multi_processor_count": int(device_properties.multi_processor_count),
    })
primary_gpu = gpu_inventory[0]
if "T4" not in primary_gpu["name"].upper():
    raise RuntimeError(
        "This release is validated only for Kaggle NVIDIA T4; "
        f"observed {primary_gpu['name']!r}. Select the T4 accelerator."
    )
if tuple(primary_gpu["compute_capability"]) < (7, 5):
    raise RuntimeError(
        "T4 FP16 contract requires compute capability >= 7.5; "
        f"observed {primary_gpu['compute_capability']}"
    )

# Exercise a real FP16 CUDA matmul, synchronize, and compare with FP32 CPU.
torch.manual_seed(170)
cuda_generator = torch.Generator(device="cuda")
cuda_generator.manual_seed(170)
left = torch.randn((257, 193), device="cuda", dtype=torch.float16, generator=cuda_generator)
right = torch.randn((193, 131), device="cuda", dtype=torch.float16, generator=cuda_generator)
product_fp16 = left @ right
torch.cuda.synchronize(0)
if product_fp16.device.type != "cuda" or product_fp16.dtype != torch.float16:
    raise RuntimeError("FP16 CUDA matmul did not return a CUDA float16 tensor")
if not bool(torch.isfinite(product_fp16).all().item()):
    raise RuntimeError("FP16 CUDA matmul produced non-finite values")
reference_fp32 = left.float().cpu() @ right.float().cpu()
max_abs_error = float((product_fp16.float().cpu() - reference_fp32).abs().max().item())
reference_scale = max(1.0, float(reference_fp32.abs().max().item()))
relative_max_error = max_abs_error / reference_scale
if relative_max_error > 0.01:
    raise RuntimeError(f"FP16 CUDA matmul parity failed: relative_max_error={relative_max_error}")

# Exercise the torchvision C++/CUDA operator that commonly exposes torch ABI mismatches.
nms_boxes = torch.tensor(
    [[0.0, 0.0, 10.0, 10.0], [1.0, 1.0, 9.0, 9.0], [20.0, 20.0, 30.0, 30.0]],
    device="cuda",
)
nms_scores = torch.tensor([0.9, 0.8, 0.7], device="cuda")
nms_keep = nms(nms_boxes, nms_scores, 0.5)
torch.cuda.synchronize(0)
if nms_keep.cpu().tolist() != [0, 2]:
    raise RuntimeError(f"torchvision CUDA NMS smoke failed: {nms_keep.cpu().tolist()}")

def distribution_version(name):
    try:
        return version(name)
    except PackageNotFoundError:
        return None

runtime_distributions = {
    name: distribution_version(name)
    for name in (
        "torch", "torchvision", "numpy", "Pillow", "PyYAML", "matplotlib",
        "pytest", "safetensors", "scikit-learn", "pandas", "onnx", "onnxruntime",
    )
}
runtime_modules = {
    "torch": str(Path(torch.__file__).resolve()),
    "torchvision": str(Path(torchvision.__file__).resolve()),
    "timm": str(observed_timm_root),
    "numpy": str(Path(np.__file__).resolve()),
    "Pillow": str(Path(PIL.__file__).resolve()),
    "PyYAML": str(Path(yaml.__file__).resolve()),
    "matplotlib": str(Path(matplotlib.__file__).resolve()),
    "pytest": str(Path(pytest.__file__).resolve()),
    "safetensors": str(Path(safetensors.__file__).resolve()),
}
RUNTIME_CONTRACT = {
    "schema_version": 2,
    "status": "passed",
    "notebook_contract": KAGGLE_NOTEBOOK_CONTRACT,
    "kaggle_runtime_release_reference": KAGGLE_RUNTIME_RELEASE,
    "kaggle_runtime_source": KAGGLE_RUNTIME_SOURCE,
    "asset_runtime_target": RUNTIME_TARGET,
    "python_required": "3.12.x",
    "python": sys.version,
    "python_cache_tag": sys.implementation.cache_tag,
    "python_soabi": sysconfig.get_config_var("SOABI"),
    "platform": platform.platform(),
    "libc": list(platform.libc_ver()),
    "torch": torch.__version__,
    "torchvision": torchvision.__version__,
    "torch_cuda": torch.version.cuda,
    "cudnn": torch.backends.cudnn.version(),
    "cuda_arch_list": list(torch.cuda.get_arch_list()),
    "visible_gpu_count": torch.cuda.device_count(),
    "gpus": gpu_inventory,
    "single_gpu_training_device": 0,
    "amp_dtype_required": "fp16",
    "dataloader_worker_policy": {
        "train_workers": 2,
        "eval_workers": 2,
        "manual_safe_fallback": 0,
        "automatic_retry_after_partial_run": False,
    },
    "fp16_cuda_matmul": {
        "status": "passed",
        "shape": [257, 193, 131],
        "max_abs_error": max_abs_error,
        "relative_max_error": relative_max_error,
        "checksum": float(product_fp16.float().sum().item()),
    },
    "torchvision_cuda_nms": {
        "status": "passed",
        "keep": nms_keep.cpu().tolist(),
    },
    "vendored_timm": {
        "version": timm.__version__,
        "root": str(VENDORED_TIMM_ROOT),
        "module": str(observed_timm_root),
        "installed_distribution_ignored": distribution_version("timm"),
    },
    "distributions": runtime_distributions,
    "modules": runtime_modules,
    "selected_environment": {
        key: os.environ.get(key)
        for key in ("KAGGLE_KERNEL_RUN_TYPE", "KAGGLE_URL_BASE", "CUDA_VISIBLE_DEVICES")
    },
    "dependency_install_performed": False,
}
runtime_fingerprint_payload = json.dumps(
    RUNTIME_CONTRACT,
    ensure_ascii=False,
    sort_keys=True,
    separators=(",", ":"),
).encode("utf-8")
RUNTIME_CONTRACT["compatibility_fingerprint_sha256"] = hashlib.sha256(
    runtime_fingerprint_payload
).hexdigest()
RUNTIME_CONTRACT_PATH = WORK_ROOT / "kaggle_v170_runtime_contract.json"
RUNTIME_CONTRACT_PATH.write_text(
    json.dumps(RUNTIME_CONTRACT, ensure_ascii=False, indent=2) + "\n",
    encoding="utf-8",
)
del left, right, product_fp16, reference_fp32, nms_boxes, nms_scores, nms_keep
torch.cuda.empty_cache()
print(json.dumps(RUNTIME_CONTRACT, ensure_ascii=False, indent=2))
'''


TRAINING_FLOW = r'''SMOKE_TAG = f"{RUN_TAG}_smoke"
PROBE_TAG = f"{RUN_TAG}_probe"
FULL_TAG = f"{RUN_TAG}_full"

def validate_completed_stage(stage, run_tag, marker_path, auto_resume=False):
    marker = json.loads(marker_path.read_text(encoding="utf-8"))
    lineage = (
        marker.get("dataset_image_tree_sha256"),
        marker.get("source_tree_sha256"),
        marker.get("dino_sha256"),
    )
    expected_lineage = (DEVELOPMENT_IMAGE_TREE_SHA256, SOURCE_TREE_SHA256, DINO_SHA256)
    if lineage != expected_lineage:
        raise RuntimeError(f"{stage} marker lineage mismatch: {lineage}")
    preflight_path = Path(str(marker.get("preflight_manifest", "")))
    if not preflight_path.is_file():
        raise RuntimeError(f"{stage} marker is missing its preflight manifest")
    preflight = json.loads(preflight_path.read_text(encoding="utf-8"))
    if preflight.get("train_args") != stage_train_args(stage, run_tag, auto_resume):
        raise RuntimeError(f"{stage} marker train contract mismatch; use a new RUN_TAG")
    if marker.get("returncode") != 0:
        raise RuntimeError(f"{stage} marker records a failed process: {marker}")
    return marker, preflight_path

SMOKE_DIR = RUNS_ROOT / run_name("smoke", SMOKE_TAG, experiment=EXPERIMENT_KEY)
smoke_train_args = stage_train_args("smoke", SMOKE_TAG)
def train_argument_value(arguments, flag):
    try:
        return arguments[arguments.index(flag) + 1]
    except (ValueError, IndexError) as error:
        raise RuntimeError(f"Training contract is missing {flag}") from error

smoke_batch_contract = {
    "max_train_batches": train_argument_value(smoke_train_args, "--max-train-batches"),
    "max_val_batches": train_argument_value(smoke_train_args, "--max-val-batches"),
    "epochs": train_argument_value(smoke_train_args, "--epochs"),
}
if smoke_batch_contract != {
    "max_train_batches": "4",
    "max_val_batches": "2",
    "epochs": "1",
}:
    raise RuntimeError(f"Smoke must be exactly 4 train / 2 val batches: {smoke_batch_contract}")
smoke_marker_path = SMOKE_DIR / "b2_stage_complete.json"
if not smoke_marker_path.is_file():
    _, _, smoke_marker_path = run_stage("smoke", SMOKE_TAG)
smoke_marker, SMOKE_PREFLIGHT_MANIFEST = validate_completed_stage(
    "smoke", SMOKE_TAG, smoke_marker_path
)
SMOKE_CHECKPOINT = SMOKE_DIR / "checkpoints" / "best.pt"
if not SMOKE_CHECKPOINT.is_file():
    SMOKE_CHECKPOINT = SMOKE_DIR / "checkpoints" / "last.pt"
if not SMOKE_CHECKPOINT.is_file():
    raise FileNotFoundError(f"Smoke stage did not produce a checkpoint: {SMOKE_DIR}")

# A successful process/checkpoint reload is insufficient when GradScaler skipped
# every optimizer step. Require explicit epoch telemetry and non-empty AdamW state.
SMOKE_HISTORY = SMOKE_DIR / "history.csv"
if not SMOKE_HISTORY.is_file():
    raise FileNotFoundError(f"Smoke stage did not write history.csv: {SMOKE_DIR}")
with SMOKE_HISTORY.open("r", encoding="utf-8-sig", newline="") as handle:
    smoke_history_rows = list(csv.DictReader(handle))
if len(smoke_history_rows) != 1:
    raise RuntimeError(f"Smoke must write exactly one epoch row, observed {len(smoke_history_rows)}")
smoke_history_row = smoke_history_rows[0]
smoke_optimizer_telemetry = {
    key: float(smoke_history_row.get(key, "nan"))
    for key in (
        "train_optimizer_step_attempts",
        "train_optimizer_updates_successful",
        "train_optimizer_steps_skipped_nonfinite",
        "train_nonfinite_loss_batches",
        "train_amp_optimizer_steps_skipped",
    )
}
if not all(math.isfinite(value) for value in smoke_optimizer_telemetry.values()):
    raise RuntimeError(f"Smoke optimizer telemetry is missing/non-finite: {smoke_optimizer_telemetry}")
if smoke_optimizer_telemetry["train_optimizer_step_attempts"] < 1:
    raise RuntimeError(f"Smoke made no optimizer attempt: {smoke_optimizer_telemetry}")
if smoke_optimizer_telemetry["train_optimizer_updates_successful"] < 1:
    raise RuntimeError(f"Smoke made no real optimizer update: {smoke_optimizer_telemetry}")
for key in (
    "train_optimizer_steps_skipped_nonfinite",
    "train_nonfinite_loss_batches",
    "train_amp_optimizer_steps_skipped",
):
    if smoke_optimizer_telemetry[key] != 0:
        raise RuntimeError(f"Smoke numerical-stability gate failed: {smoke_optimizer_telemetry}")

SMOKE_STATE_CHECKPOINT = SMOKE_DIR / "checkpoints" / "last.pt"
if not SMOKE_STATE_CHECKPOINT.is_file():
    raise FileNotFoundError(f"Smoke stage did not write last.pt state checkpoint: {SMOKE_DIR}")
smoke_checkpoint_payload = torch.load(
    SMOKE_STATE_CHECKPOINT,
    map_location="cpu",
    weights_only=False,
)
smoke_optimizer_state_entries = len(
    smoke_checkpoint_payload.get("optimizer_state", {}).get("state", {})
)
smoke_scaler_state = smoke_checkpoint_payload.get("scaler_state", {})
smoke_checkpoint_scale = float(smoke_scaler_state.get("scale", float("nan")))
del smoke_checkpoint_payload
if smoke_optimizer_state_entries < 1:
    raise RuntimeError("Smoke checkpoint has empty optimizer state despite a successful process")
if not math.isfinite(smoke_checkpoint_scale) or smoke_checkpoint_scale != AMP_INIT_SCALE:
    raise RuntimeError(
        "Smoke GradScaler drift/overflow: "
        f"checkpoint_scale={smoke_checkpoint_scale}, expected={AMP_INIT_SCALE}"
    )
SMOKE_OPTIMIZER_CONTRACT = {
    "status": "passed",
    "history": str(SMOKE_HISTORY),
    "history_sha256": sha256(SMOKE_HISTORY),
    "state_checkpoint": str(SMOKE_STATE_CHECKPOINT),
    "state_checkpoint_sha256": sha256(SMOKE_STATE_CHECKPOINT),
    "amp_init_scale": AMP_INIT_SCALE,
    "checkpoint_scale": smoke_checkpoint_scale,
    "optimizer_state_entries": smoke_optimizer_state_entries,
    **smoke_optimizer_telemetry,
}

# A separate evaluator process must reload the checkpoint and complete one val batch.
smoke_checkpoint_sha256 = sha256(SMOKE_CHECKPOINT)
SMOKE_RELOAD_DIR = SMOKE_DIR / f"reload_eval_1batch_{smoke_checkpoint_sha256[:12]}"
SMOKE_RELOAD_MARKER_PATH = SMOKE_RELOAD_DIR / "checkpoint_reload_contract.json"
if not SMOKE_RELOAD_MARKER_PATH.is_file():
    run_checked([
        sys.executable, "-m", "trkh.evaluation.evaluate",
        "--checkpoint", SMOKE_CHECKPOINT,
        "--data", DEV_YAML,
        "--class-name-mode", "raw",
        "--expected-num-classes", "5",
        "--split", "val",
        "--batch-size", str(EVAL_BATCH),
        "--num-workers", "2",
        "--amp",
        "--max-batches", "1",
        "--paper-name", "TRKH-DINOv3-ClassF-B2-Kaggle-Smoke-Reload",
        "--family", "TRKH-Pretrained-Engineering-Smoke",
        "--output-dir", SMOKE_RELOAD_DIR,
    ])
    smoke_predictions = SMOKE_RELOAD_DIR / "predictions_detailed.csv"
    if not smoke_predictions.is_file():
        smoke_predictions = SMOKE_RELOAD_DIR / "predictions.csv"
    smoke_metrics = SMOKE_RELOAD_DIR / "metrics_detailed.json"
    if not smoke_metrics.is_file():
        smoke_metrics = SMOKE_RELOAD_DIR / "metrics.json"
    if not smoke_predictions.is_file() or not smoke_metrics.is_file():
        raise FileNotFoundError("Independent smoke checkpoint reload did not write metrics/predictions")
    with smoke_predictions.open("r", encoding="utf-8-sig", newline="") as handle:
        smoke_prediction_rows = sum(1 for _ in csv.DictReader(handle))
    if smoke_prediction_rows < 1 or smoke_prediction_rows > EVAL_BATCH:
        raise RuntimeError(f"Unexpected one-batch prediction rows: {smoke_prediction_rows}")
    SMOKE_RELOAD_CONTRACT = {
        "schema_version": 1,
        "status": "passed",
        "separate_process_checkpoint_reload": True,
        "max_val_batches": 1,
        "checkpoint": str(SMOKE_CHECKPOINT),
        "checkpoint_sha256": smoke_checkpoint_sha256,
        "metrics": str(smoke_metrics),
        "metrics_sha256": sha256(smoke_metrics),
        "predictions": str(smoke_predictions),
        "predictions_sha256": sha256(smoke_predictions),
        "prediction_rows": smoke_prediction_rows,
        "optimizer_update_contract": SMOKE_OPTIMIZER_CONTRACT,
        "runtime_fingerprint_sha256": RUNTIME_CONTRACT["compatibility_fingerprint_sha256"],
    }
    SMOKE_RELOAD_DIR.mkdir(parents=True, exist_ok=True)
    SMOKE_RELOAD_MARKER_PATH.write_text(
        json.dumps(SMOKE_RELOAD_CONTRACT, ensure_ascii=False, indent=2) + "\n",
        encoding="utf-8",
    )
else:
    SMOKE_RELOAD_CONTRACT = json.loads(SMOKE_RELOAD_MARKER_PATH.read_text(encoding="utf-8"))
    expected_reload = {
        "status": "passed",
        "separate_process_checkpoint_reload": True,
        "max_val_batches": 1,
        "checkpoint_sha256": smoke_checkpoint_sha256,
        "optimizer_update_contract": SMOKE_OPTIMIZER_CONTRACT,
        "runtime_fingerprint_sha256": RUNTIME_CONTRACT["compatibility_fingerprint_sha256"],
    }
    reload_mismatches = {
        key: {"observed": SMOKE_RELOAD_CONTRACT.get(key), "expected": expected}
        for key, expected in expected_reload.items()
        if SMOKE_RELOAD_CONTRACT.get(key) != expected
    }
    for artifact_key, digest_key in (("metrics", "metrics_sha256"), ("predictions", "predictions_sha256")):
        artifact_path = Path(str(SMOKE_RELOAD_CONTRACT.get(artifact_key, "")))
        if not artifact_path.is_file() or sha256(artifact_path) != SMOKE_RELOAD_CONTRACT.get(digest_key):
            reload_mismatches[artifact_key] = "missing_or_digest_mismatch"
    if reload_mismatches:
        raise RuntimeError(f"Smoke reload contract mismatch: {reload_mismatches}")
print("Smoke 4-train/2-val plus independent one-batch reload passed:", SMOKE_RELOAD_MARKER_PATH)

PROBE_DIR = RUNS_ROOT / run_name("probe", PROBE_TAG, experiment=EXPERIMENT_KEY)
PROBE_COMPLETED = False
if RUN_PROBE:
    probe_marker_path = PROBE_DIR / "b2_stage_complete.json"
    if not probe_marker_path.is_file():
        _, _, probe_marker_path = run_stage("probe", PROBE_TAG)
    probe_marker, PROBE_PREFLIGHT_MANIFEST = validate_completed_stage(
        "probe", PROBE_TAG, probe_marker_path
    )
    probe_metrics = probe_marker.get("metrics", {})
    if float(probe_metrics.get("best_macro_f1") or 0) < PROBE_MIN_MACRO_F1:
        raise RuntimeError(f"Probe macro-F1 gate failed: {probe_metrics}")
    if float(probe_metrics.get("best_class1_f1") or 0) < PROBE_MIN_CLASS1_F1:
        raise RuntimeError(f"Probe class-1 gate failed: {probe_metrics}")
    PROBE_COMPLETED = True
    print("Probe gate passed:", probe_metrics)
else:
    print("Probe execution skipped only because AUTO_RESUME=True; resume validation remains fail-closed.")

FULL_REQUESTED = bool(CONFIRM_FULL or AUTO_RESUME)
FULL_COMPLETED = False
RUN_DIR = None
FULL_PREFLIGHT_MANIFEST = None
BEST_CHECKPOINT = None
if FULL_REQUESTED:
    if not (PROBE_COMPLETED or AUTO_RESUME):
        raise RuntimeError("Full requires a completed probe or provenance-checked AUTO_RESUME")
    full_dir = RUNS_ROOT / run_name("full", FULL_TAG, experiment=EXPERIMENT_KEY)
    full_marker_path = full_dir / "b2_stage_complete.json"
    if full_marker_path.is_file():
        _, FULL_PREFLIGHT_MANIFEST = validate_completed_stage(
            "full", FULL_TAG, full_marker_path, AUTO_RESUME
        )
        RUN_DIR = full_dir
    else:
        RUN_DIR, FULL_PREFLIGHT_MANIFEST, full_marker_path = run_stage(
            "full", FULL_TAG, AUTO_RESUME
        )
    BEST_CHECKPOINT = RUN_DIR / "checkpoints" / "best.pt"
    if not BEST_CHECKPOINT.is_file() or not FULL_PREFLIGHT_MANIFEST.is_file():
        raise FileNotFoundError(f"Missing best/preflight: {BEST_CHECKPOINT}, {FULL_PREFLIGHT_MANIFEST}")
    FULL_COMPLETED = True
    print("Best checkpoint:", BEST_CHECKPOINT)
else:
    print(
        "Probe is complete. No exception is raised and no audit is attempted. "
        "Review probe evidence, set CONFIRM_FULL=True, then rerun from the configuration cell."
    )
'''


SKIP_SESSION_CELL = r'''if not FULL_COMPLETED:
    readiness_manifest = {
        "schema_version": 2,
        "status": "probe_ready_full_not_requested",
        "protocol": PROTOCOL_ID,
        "experiment": EXPERIMENT_KEY,
        "kaggle_notebook_contract": KAGGLE_NOTEBOOK_CONTRACT,
        "runtime_contract": str(RUNTIME_CONTRACT_PATH),
        "runtime_fingerprint_sha256": RUNTIME_CONTRACT["compatibility_fingerprint_sha256"],
        "smoke_stage_marker": str(SMOKE_DIR / "b2_stage_complete.json"),
        "smoke_checkpoint_reload_contract": str(SMOKE_RELOAD_MARKER_PATH),
        "probe_completed": PROBE_COMPLETED,
        "probe_stage_marker": str(PROBE_DIR / "b2_stage_complete.json") if PROBE_COMPLETED else None,
        "full_requested": FULL_REQUESTED,
        "full_completed": False,
        "audits_run": False,
        "next_action": "Review probe, set CONFIRM_FULL=True, rerun from cell 0.",
        "test_split_content_opened_or_hashed": False,
        "test_metrics_read": False,
    }
    SESSION_MANIFEST = WORK_ROOT / f"TRKH_CLASSF_B2_{RUN_TAG}_PROBE_READINESS.json"
    SESSION_MANIFEST.write_text(
        json.dumps(readiness_manifest, ensure_ascii=False, indent=2) + "\n",
        encoding="utf-8",
    )
    print("Full was not requested; audit/package cells skipped cleanly.")
    print("Readiness manifest:", SESSION_MANIFEST)
else:
{full_manifest_body}
'''


def _source(cell: dict) -> str:
    return "".join(cell.get("source", []))


def _set_source(cell: dict, source: str) -> dict:
    result = copy.deepcopy(cell)
    result["source"] = source.splitlines(keepends=True)
    if source and not source.endswith("\n"):
        result["source"].append("\n")
    result["execution_count"] = None
    result["outputs"] = []
    return result


def _new_code_cell(source: str) -> dict:
    return {
        "cell_type": "code",
        "execution_count": None,
        "metadata": {},
        "outputs": [],
        "source": source.splitlines(keepends=True),
    }


def _replace_once(source: str, old: str, new: str, label: str) -> str:
    count = source.count(old)
    if count != 1:
        raise ValueError(f"Expected exactly one {label}; found {count}")
    return source.replace(old, new, 1)


def _guard_full(source: str, label: str) -> str:
    return (
        "if not FULL_COMPLETED:\n"
        f"    print(\"SKIP {label}: full train has not completed.\")\n"
        "else:\n"
        + indent(source.rstrip() + "\n", "    ")
    )


def build_notebook(input_path: Path, output_path: Path) -> dict:
    # Canonicalize the reviewed Git blob so generation is byte-stable even
    # when a Windows archive/export materializes text files with CRLF.
    raw = input_path.read_bytes().replace(b"\r\n", b"\n")
    notebook = json.loads(raw.decode("utf-8"))
    cells = notebook.get("cells", [])
    if len(cells) != 11 or any(cell.get("cell_type") != "code" for cell in cells):
        raise ValueError("Reviewed V1 must contain exactly 11 code cells")

    config = _source(cells[0])
    if V1_CONTRACT not in config:
        raise ValueError("Input notebook is not the reviewed B2 V1 contract")
    config = _replace_once(
        config,
        "import json\n",
        "import json\nimport math\n",
        "math import for numerical smoke gates",
    )
    config = _replace_once(config, V1_CONTRACT, V2_CONTRACT, "notebook contract")
    config = _replace_once(
        config,
        'RUN_TAG = "kaggle_b2_tempered_p05_v1"',
        'RUN_TAG = "kaggle_b2_tempered_p05_v2"',
        "run tag",
    )
    config = _replace_once(
        config,
        'EXPECTED_SOURCE_COMMIT = "f1d79def090e347c0586efae4003b2579d368a28"',
        f'EXPECTED_SOURCE_COMMIT = "{LOCKED_SOURCE_COMMIT}"',
        "locked runtime-port source commit",
    )
    config = _replace_once(
        config,
        'EXPECTED_SOURCE_TREE_SHA256 = "61b860a5549ac172d06c64ee1c5f9fe035cc9853074312f2fe5a2de80390e747"',
        f'EXPECTED_SOURCE_TREE_SHA256 = "{LOCKED_SOURCE_TREE_SHA256}"',
        "locked runtime-port source tree",
    )
    config = _replace_once(
        config,
        'KAGGLE_NOTEBOOK_CONTRACT = "' + V2_CONTRACT + '"\n',
        'KAGGLE_NOTEBOOK_CONTRACT = "' + V2_CONTRACT + '"\n'
        'KAGGLE_RUNTIME_RELEASE = "v170"\n'
        'KAGGLE_RUNTIME_SOURCE = "https://github.com/Kaggle/docker-python/releases/tag/v170-GPU-bdf9e0538555f90453619adefb49ba40cfa136db44a9c9be7a42ea715c0aa068"\n'
        'sys.dont_write_bytecode = True\n'
        'os.environ["PYTHONDONTWRITEBYTECODE"] = "1"\n',
        "runtime release insertion",
    )

    old_asset_setup = _source(cells[2])
    split_marker = "def ordered_names(document):\n"
    if old_asset_setup.count(split_marker) != 1:
        raise ValueError("Could not split V1 extraction from YAML/data setup")
    asset_part, data_part = old_asset_setup.split(split_marker, 1)
    asset_part = asset_part.replace("import yaml\n", "")
    asset_part = _replace_once(
        asset_part,
        'ASSET_ZIP_GLOB = "TRKH_KAGGLE_B2_UPLOAD_BUNDLE_*.zip"',
        'ASSET_ZIP_GLOB = "TRKH_KAGGLE_B2_T4_UPLOAD_BUNDLE_*.zip"',
        "asset ZIP glob",
    )
    expected_manifest_assignment = f'ASSET_MANIFEST_NAME = "{V2_ASSET_MANIFEST_NAME}"'
    if asset_part.count(expected_manifest_assignment) != 1:
        raise ValueError("Reviewed V1 asset-manifest name drifted")
    asset_part = _replace_once(
        asset_part,
        'ASSET_ZIP = discover_optional_zip(ASSET_ZIP_GLOB, "asset bundle")\n',
        'ASSET_ZIP = discover_optional_zip(ASSET_ZIP_GLOB, "asset bundle")\n'
        'if ASSET_ZIP is None:\n'
        '    raise RuntimeError(f"Attach exactly one V2 asset ZIP matching {ASSET_ZIP_GLOB}")\n',
        "required asset ZIP",
    )
    asset_part = _replace_once(
        asset_part,
        '        "dino_sha256": DINO_SHA256,\n',
        '        "dino_sha256": DINO_SHA256,\n'
        '        "vendored_timm_version": "1.0.27",\n'
        '        "kaggle_runtime_target": "v170 GPU / Python 3.12 / Torch 2.10 cu128 / T4",\n',
        "manifest runtime fields",
    )
    asset_part = _replace_once(
        asset_part,
        '        ASSET_EXTRACT_ROOT / "TRKH_CLASSF_BEST_KAGGLE.ipynb",\n',
        '        ASSET_EXTRACT_ROOT / "TRKH_CLASSF_BEST_KAGGLE.ipynb",\n'
        '        ASSET_EXTRACT_ROOT / "vendor" / "python" / "timm" / "__init__.py",\n'
        '        ASSET_EXTRACT_ROOT / "third_party" / "dinov3" / "LICENSE.md",\n'
        '        ASSET_EXTRACT_ROOT / "third_party" / "dinov3" / "PROVENANCE.json",\n'
        '        ASSET_EXTRACT_ROOT / "TRKH_KAGGLE_RUNTIME_TARGET.json",\n',
        "required V2 notebook/vendor paths",
    )
    asset_validation_anchor = '''    if missing_required_assets:
        raise RuntimeError(f"Asset bundle thieu file bat buoc: {missing_required_assets}")
'''
    asset_validation_extension = asset_validation_anchor + r'''
    runtime_target_path = ASSET_EXTRACT_ROOT / "TRKH_KAGGLE_RUNTIME_TARGET.json"
    provenance_path = ASSET_EXTRACT_ROOT / "third_party" / "dinov3" / "PROVENANCE.json"
    license_path = ASSET_EXTRACT_ROOT / "third_party" / "dinov3" / "LICENSE.md"
    RUNTIME_TARGET = json.loads(runtime_target_path.read_text(encoding="utf-8"))
    DINO_PROVENANCE = json.loads(provenance_path.read_text(encoding="utf-8"))
    runtime_target_required = {
        "contract": KAGGLE_NOTEBOOK_CONTRACT,
    }
    target_required = {
        "official_gpu_release": KAGGLE_RUNTIME_RELEASE,
        "python_minor": "3.12",
        "torch_family": "2.10.x+cu128",
        "torchvision_family": "0.25.x+cu128",
        "preferred_accelerator": "NvidiaTeslaT4",
        "minimum_compute_capability": [7, 5],
        "p100_supported": False,
        "fp16_grad_scaler_init_scale": 1024.0,
    }
    target_mismatches = {
        key: {"observed": RUNTIME_TARGET.get(key), "expected": expected}
        for key, expected in runtime_target_required.items()
        if RUNTIME_TARGET.get(key) != expected
    }
    observed_target = RUNTIME_TARGET.get("target", {})
    target_mismatches.update({
        f"target.{key}": {"observed": observed_target.get(key), "expected": expected}
        for key, expected in target_required.items()
        if observed_target.get(key) != expected
    })
    dependency_policy = RUNTIME_TARGET.get("dependency_policy", {})
    if dependency_policy.get("mutates_global_environment") is not False:
        target_mismatches["dependency_policy.mutates_global_environment"] = dependency_policy.get("mutates_global_environment")
    if dependency_policy.get("pip_install") is not False:
        target_mismatches["dependency_policy.pip_install"] = dependency_policy.get("pip_install")
    if dependency_policy.get("vendored_timm") != "1.0.27":
        target_mismatches["dependency_policy.vendored_timm"] = dependency_policy.get("vendored_timm")
    if target_mismatches:
        raise RuntimeError(f"Kaggle runtime target mismatch: {target_mismatches}")

    provenance_required = {
        "revision": "3bf4720a82ec2066db88137180ff1f83a675cef0",
        "weight_path": "weights/model.safetensors",
        "weight_size_bytes": DINO_BYTES,
        "weight_sha256": DINO_SHA256,
        "license_path": "third_party/dinov3/LICENSE.md",
        "license_sha256": asset_manifest.get("dinov3_license_sha256"),
    }
    provenance_mismatches = {
        key: {"observed": DINO_PROVENANCE.get(key), "expected": expected}
        for key, expected in provenance_required.items()
        if DINO_PROVENANCE.get(key) != expected
    }
    if sha256(license_path) != asset_manifest.get("dinov3_license_sha256"):
        provenance_mismatches["license_file_sha256"] = sha256(license_path)
    if provenance_mismatches:
        raise RuntimeError(f"DINO provenance/license mismatch: {provenance_mismatches}")

    vendored_timm_digest = hashlib.sha256()
    vendor_root = ASSET_EXTRACT_ROOT / "vendor" / "python"
    vendor_files = [item for item in vendor_root.rglob("*") if item.is_file()]
    for path in sorted(
        vendor_files,
        key=lambda item: item.relative_to(ASSET_EXTRACT_ROOT).as_posix(),
    ):
        relative = path.relative_to(ASSET_EXTRACT_ROOT).as_posix()
        vendored_timm_digest.update(relative.encode("utf-8") + b"\0")
        vendored_timm_digest.update(path.read_bytes())
    observed_timm_tree_sha256 = vendored_timm_digest.hexdigest()
    if observed_timm_tree_sha256 != asset_manifest.get("vendored_timm_tree_sha256"):
        raise RuntimeError(
            "Vendored timm tree digest mismatch: "
            f"{observed_timm_tree_sha256} != {asset_manifest.get('vendored_timm_tree_sha256')}"
        )
'''
    asset_part = _replace_once(
        asset_part,
        asset_validation_anchor,
        asset_validation_extension,
        "asset runtime/provenance validation",
    )
    data_part = _replace_once(
        data_part,
        'os.environ["PYTHONPATH"] = str(REPO_ROOT)\n',
        'os.environ["PYTHONPATH"] = (\n'
        '    str(VENDORED_TIMM_ROOT) + os.pathsep + str(REPO_ROOT)\n'
        ')\n',
        "vendored timm subprocess path",
    )
    data_part = _replace_once(
        data_part,
        'source_files = sorted((REPO_ROOT / "trkh").rglob("*.py")) + sorted((REPO_ROOT / "configs").rglob("*.yaml"))\n',
        'source_files = (\n'
        '    sorted(\n'
        '        (REPO_ROOT / "trkh").rglob("*.py"),\n'
        '        key=lambda path: path.relative_to(REPO_ROOT).as_posix(),\n'
        '    )\n'
        '    + sorted(\n'
        '        (REPO_ROOT / "configs").rglob("*.yaml"),\n'
        '        key=lambda path: path.relative_to(REPO_ROOT).as_posix(),\n'
        '    )\n'
        ')\n',
        "platform-stable source-tree ordering",
    )
    data_setup = "import yaml\n\n" + split_marker + data_part

    hardware = _source(cells[4])
    hardware_start = hardware.index("if not torch.cuda.is_available():")
    hardware_end_marker = 'os.environ.setdefault("OMP_NUM_THREADS", "4")\n'
    hardware_end = hardware.index(hardware_end_marker, hardware_start) + len(hardware_end_marker)
    hardware_contract = '''if RUNTIME_CONTRACT.get("status") != "passed":
    raise RuntimeError("Kaggle runtime contract was not established")
properties = torch.cuda.get_device_properties(0)
vram_gib = properties.total_memory / (1024 ** 3)
MICRO_BATCH = 24 if vram_gib >= 14.5 else 16
GRAD_ACCUM = {24: 2, 16: 3}[MICRO_BATCH]
AMP_DTYPE = "fp16"
AMP_INIT_SCALE = 1024.0
EVAL_BATCH = 64 if vram_gib >= 14.5 else 32
NUM_WORKERS = 2
EVAL_NUM_WORKERS = 2
os.environ["TRKH_AMP_DTYPE"] = AMP_DTYPE
os.environ.setdefault("OMP_NUM_THREADS", "4")
'''
    hardware = hardware[:hardware_start] + hardware_contract + hardware[hardware_end:]
    hardware = _replace_once(
        hardware,
        '        num_workers=4,\n        eval_num_workers=2,\n',
        '        num_workers=NUM_WORKERS,\n        eval_num_workers=EVAL_NUM_WORKERS,\n',
        "Kaggle DataLoader worker policy",
    )
    hardware = _replace_once(
        hardware,
        '        seed=42,\n        auto_resume=auto_resume,\n',
        '        seed=42,\n'
        '        amp_init_scale=AMP_INIT_SCALE,\n'
        '        auto_resume=auto_resume,\n',
        "Kaggle FP16 GradScaler initial scale",
    )

    training = _source(cells[5])
    flow_start = training.index("PROBE_DIR =")
    training_prefix = training[:flow_start]
    training_prefix = _replace_once(
        training_prefix,
        '    REPO_ROOT / "tests" / "test_mobile_onnx_quantization.py",\n',
        "",
        "non-bundled optional ONNX test",
    )
    training_prefix = _replace_once(
        training_prefix,
        '    test_returncode = run_checked([sys.executable, "-m", "pytest", "-q", *focused_tests])\n',
        '    test_returncode = run_checked([\n'
        '        sys.executable, "-m", "pytest", "-q", "-p", "no:cacheprovider", *focused_tests\n'
        '    ])\n',
        "immutable extracted-source pytest invocation",
    )
    training_prefix = _replace_once(
        training_prefix,
        '        "runtime": {\n',
        '        "runtime_contract": RUNTIME_CONTRACT,\n'
        '        "runtime_contract_path": str(RUNTIME_CONTRACT_PATH),\n'
        '        "smoke_reload_contract": globals().get("SMOKE_RELOAD_CONTRACT"),\n'
        '        "runtime": {\n',
        "preflight runtime contract",
    )
    training_prefix = _replace_once(
        training_prefix,
        '            "effective_batch": MICRO_BATCH * GRAD_ACCUM,\n',
        '            "effective_batch": MICRO_BATCH * GRAD_ACCUM,\n'
        '            "amp_init_scale": AMP_INIT_SCALE,\n'
        '            "num_workers": NUM_WORKERS,\n'
        '            "eval_num_workers": EVAL_NUM_WORKERS,\n',
        "preflight worker telemetry",
    )
    training_prefix = _replace_once(
        training_prefix,
        '        raise RuntimeError(f"B2 {stage} failed: {marker_path}")\n',
        '        raise RuntimeError(\n'
        '            f"B2 {stage} failed: {marker_path}. "\n'
        '            "If Kaggle reports a DataLoader worker-start failure, set "\n'
        '            "NUM_WORKERS=0 and EVAL_NUM_WORKERS=0 before retrying with a new RUN_TAG; "\n'
        '            "the notebook will not auto-retry a possibly partial run."\n'
        '        )\n',
        "DataLoader failure guidance",
    )
    training = training_prefix + TRAINING_FLOW

    evaluation = (
        "if not FULL_COMPLETED:\n"
        "    EVAL_DIR = PREDICTIONS = METRICS_DETAIL = None\n"
        "    print(\"SKIP validation evaluation: full train has not completed.\")\n"
        "else:\n"
        + indent(_source(cells[6]).rstrip() + "\n", "    ")
    )
    confusion_audits = _guard_full(_source(cells[7]), "class-confusion/forensics audits")
    xai_audits = _guard_full(_source(cells[8]), "XAI/robustness audits")

    export_body = _source(cells[9])
    install_start = export_body.index("# DINOv3 B2 la research teacher")
    export_prefix = export_body[:install_start]
    deploy_body = r'''# DINOv3 B2 is a research teacher, not the mobile deliverable.
deploy_code = None
if importlib.util.find_spec("onnx") is None:
    print("SKIP optional ONNX export: onnx is absent from the immutable Kaggle runtime.")
else:
    deploy_code = run_checked([
        sys.executable, "-m", "trkh.inference.deploy",
        "--checkpoint", BEST_CHECKPOINT,
        "--data", DEV_YAML,
        "--class-name-mode", "raw",
        "--expected-num-classes", "5",
        "--output-dir", RUN_DIR / "deploy_teacher_fp32",
        "--batch-size", EVAL_BATCH,
        "--num-workers", "2",
        "--benchmark-batch-size", "1",
        "--benchmark-warmup", "2",
        "--benchmark-runs", "2",
        "--split", "val",
        "--skip-accuracy",
        "--skip-benchmark",
        "--skip-trt-engine",
    ], required=False)
    if deploy_code:
        print("WARNING: optional ONNX export failed; checkpoint/audits remain valid:", deploy_code)
'''
    export_cell = (
        "deploy_code = None\n"
        "if not FULL_COMPLETED:\n"
        "    print(\"SKIP architecture trace/optional export: full train has not completed.\")\n"
        "else:\n"
        + indent((export_prefix + deploy_body).rstrip() + "\n", "    ")
    )

    manifest_body = _source(cells[10])
    manifest_body = _replace_once(
        manifest_body,
        'shutil.copy2(FULL_PREFLIGHT_MANIFEST, RUN_DIR / "b2_full_preflight_manifest.json")\n',
        'shutil.copy2(FULL_PREFLIGHT_MANIFEST, RUN_DIR / "b2_full_preflight_manifest.json")\n'
        'shutil.copy2(RUNTIME_CONTRACT_PATH, RUN_DIR / RUNTIME_CONTRACT_PATH.name)\n'
        'shutil.copy2(SMOKE_RELOAD_MARKER_PATH, RUN_DIR / "smoke_checkpoint_reload_contract.json")\n',
        "runtime evidence copies",
    )
    manifest_body = _replace_once(
        manifest_body,
        '    RUN_DIR / "b2_full_preflight_manifest.json",\n',
        '    RUN_DIR / "b2_full_preflight_manifest.json",\n'
        '    RUN_DIR / RUNTIME_CONTRACT_PATH.name,\n'
        '    RUN_DIR / "smoke_checkpoint_reload_contract.json",\n',
        "runtime evidence inventory",
    )
    manifest_body = _replace_once(
        manifest_body,
        '    "python": sys.version,\n',
        '    "python": sys.version,\n'
        '    "runtime_contract": RUNTIME_CONTRACT,\n'
        '    "runtime_contract_sha256": sha256(RUN_DIR / RUNTIME_CONTRACT_PATH.name),\n'
        '    "smoke_checkpoint_reload_contract_sha256": sha256(RUN_DIR / "smoke_checkpoint_reload_contract.json"),\n',
        "result runtime contract",
    )
    session_or_manifest = SKIP_SESSION_CELL.replace(
        "{full_manifest_body}", indent(manifest_body.rstrip() + "\n", "    ")
    )

    notebook["cells"] = [
        _set_source(cells[0], config),
        _new_code_cell(asset_part),
        _new_code_cell(RUNTIME_CELL),
        _new_code_cell(data_setup),
        _set_source(cells[3], _source(cells[3])),
        _set_source(cells[4], hardware),
        _set_source(cells[5], training),
        _set_source(cells[6], evaluation),
        _set_source(cells[7], confusion_audits),
        _set_source(cells[8], xai_audits),
        _set_source(cells[9], export_cell),
        _set_source(cells[10], session_or_manifest),
    ]
    metadata = copy.deepcopy(notebook.get("metadata", {}))
    metadata["trkh_release"] = {
        "generator": Path(__file__).name,
        "source_notebook": input_path.name,
        "source_notebook_sha256": hashlib.sha256(raw).hexdigest(),
        "notebook_contract": V2_CONTRACT,
        "kaggle_runtime_release": "v170",
        "python_runtime": "3.12.x",
        "gpu_contract": "NVIDIA T4, compute capability >= 7.5, FP16",
        "dependency_policy": "immutable Kaggle base plus manifest-covered vendored timm 1.0.27",
    }
    notebook["metadata"] = metadata

    generated_source = "\n".join(_source(cell) for cell in notebook["cells"])
    forbidden = ("pip install", "ensure_locked_packages", "Da dung sau probe")
    found = [token for token in forbidden if token in generated_source]
    if found:
        raise AssertionError(f"Generated V2 retains forbidden runtime behavior: {found}")
    required = (
        "safe_extract_zip(ASSET_ZIP",
        '"vendor" / "python"',
        "FP16 CUDA matmul",
        "torchvision CUDA NMS",
        'run_stage("smoke", SMOKE_TAG)',
        '"--max-batches", "1"',
        'str(VENDORED_TIMM_ROOT) + os.pathsep + str(REPO_ROOT)',
        "if not FULL_COMPLETED:",
    )
    missing = [token for token in required if token not in generated_source]
    if missing:
        raise AssertionError(f"Generated V2 is missing contract tokens: {missing}")
    return notebook


def parse_args() -> argparse.Namespace:
    repo_root = Path(__file__).resolve().parents[2]
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--input",
        type=Path,
        required=True,
        help="Reviewed 11-cell V1 notebook exported from the locked Git revision.",
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=repo_root / "kagle" / V2_NOTEBOOK_NAME,
    )
    parser.add_argument("--force", action="store_true", help="Replace an existing V2 notebook")
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    input_path = args.input.resolve()
    output_path = args.output.resolve()
    if not input_path.is_file():
        raise FileNotFoundError(input_path)
    if output_path.exists() and not args.force:
        raise FileExistsError(f"Refusing to replace {output_path}; pass --force explicitly")
    notebook = build_notebook(input_path, output_path)
    payload = (json.dumps(notebook, ensure_ascii=False, indent=1) + "\n").encode("utf-8")
    output_path.parent.mkdir(parents=True, exist_ok=True)
    temporary_path = output_path.with_name(output_path.name + ".v2.partial")
    if temporary_path.exists():
        raise FileExistsError(f"Refusing to overwrite incomplete generation: {temporary_path}")
    temporary_path.write_bytes(payload)
    os.replace(temporary_path, output_path)
    print(f"Wrote {output_path}")
    print(f"sha256={hashlib.sha256(payload).hexdigest()} code_cells={len(notebook['cells'])}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
