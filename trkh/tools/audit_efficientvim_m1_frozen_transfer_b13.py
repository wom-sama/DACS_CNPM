from __future__ import annotations

import argparse
import hashlib
import importlib.metadata
import json
import os
import platform
import subprocess
import sys
import tempfile
import time
import warnings
from collections.abc import Mapping, Sequence
from pathlib import Path
from typing import Any

os.environ.setdefault("CUBLAS_WORKSPACE_CONFIG", ":4096:8")
os.environ.setdefault("HF_HUB_OFFLINE", "1")

import numpy as np  # noqa: E402
import onnx  # noqa: E402
import onnxruntime as ort  # noqa: E402
import timm  # noqa: E402
import torch  # noqa: E402
from huggingface_hub import try_to_load_from_cache  # noqa: E402
from PIL import Image, ImageOps  # noqa: E402
from safetensors.torch import load_file  # noqa: E402
from sklearn.exceptions import ConvergenceWarning  # noqa: E402
from sklearn.linear_model import LogisticRegression  # noqa: E402
from sklearn.metrics import (  # noqa: E402
    accuracy_score,
    confusion_matrix,
    precision_recall_fscore_support,
    roc_auc_score,
)
from sklearn.preprocessing import StandardScaler  # noqa: E402
from torch import Tensor, nn  # noqa: E402
from torch.utils.data import DataLoader, Dataset  # noqa: E402
from torchvision.transforms import InterpolationMode  # noqa: E402
from torchvision.transforms import v2  # noqa: E402

from trkh.models.efficientvim_m1 import (  # noqa: E402
    OFFICIAL_M1_E450_SHA256,
    EfficientViMM1,
    load_official_efficientvim_m1_weights,
)
from trkh.tools.audit_dinov3_depth_trajectory_b12 import (  # noqa: E402
    _atomic_json,
    _atomic_npz,
)
from trkh.tools.precheck_dinov3_xcnorm_a0_sourcefold import (  # noqa: E402
    EXPECTED_CLASSES,
    EXPECTED_DATA_SHA256,
    EXPECTED_TRAIN_SAMPLES,
    _data_root_from_yaml,
    _sha256,
    assert_train_only_paths,
)
from trkh.tools.probe_embedding_prototypes import _resolve_device  # noqa: E402
from trkh.tools.screen_dinov3_cgaer_b11_sourcefold import (  # noqa: E402
    read_locked_assignment,
    validate_a0_cache,
)


PROTOCOL_ID = "TRKH_PRETRAINED_CLASSF_B13_EFFICIENTVIM_M1_FROZEN_TRANSFER_20260803"
SEED = 20260803
FOLDS = 5
CLASSES = 5
CLASS1 = 1
RIVALS = (0, 2, 4)
EXPECTED_CLASS_COUNTS = (1987, 497, 1326, 2080, 2388)
EXPECTED_ASSIGNMENT_SHA256 = (
    "afde4fec6b344b867d31e0e01b89f16c5f4be7fd551fdc1fdfd0ea34219d725f"
)
DINO_MODEL = "vit_small_patch16_dinov3.lvd1689m"
DINO_HF_REPO = "timm/vit_small_patch16_dinov3.lvd1689m"
DINO_WEIGHT_SHA256 = (
    "2a1ec16ae28ffa07bc0ead0241ee7df9fc26451fe6f9f839b7b3afa0a906b040"
)
DINO_IMAGE_SIZE = 256
DINO_PREFIX_TOKENS = 5
DINO_FEATURE_DIM = 384
EFFICIENTVIM_IMAGE_SIZE = 224
EFFICIENTVIM_FEATURE_DIM = 320
EFFICIENTVIM_1000_CLASS_PARAMS = 6_679_458
EFFICIENTVIM_5_CLASS_PARAMS = 5_720_278
DINO_5_CLASS_PARAMS = 21_588_869
BATCH_SIZE = 32
WORKERS = 0
LOGISTIC_C = 1.0
LOGISTIC_TOL = 1e-8
LOGISTIC_MAX_ITER = 2_000
BOOTSTRAP_REPLICATES = 5_000
ORT_THREADS = 4
ORT_WARMUPS = 15
ORT_TRIALS = 5
ORT_ITERATIONS = 100
OFFICIAL_SOURCE_REVISION = "304340cb9c339b61669250d058525c9cdadd5e93"
OFFICIAL_SOURCE_HASHES = {
    "EfficientViM.py": "e8717ca0fe3a3d5a883afe590e572a6706ee0004399230a865e68b62a04b15bd",
    "utils.py": "91431b918a5f00bd47b19a70676c5c6d993f4f651d020bcd180f357e6978c945",
    "LICENSE": "c5749700fe7c07b7edb3082c234e46f1e140311f84dcda2268ff8a0392a671fb",
}


def _parse_args(argv: Sequence[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Locked B13 TRAIN-only frozen EfficientViM-M1 versus raw DINOv3 "
            "representation screen. Validation/test are never constructed."
        )
    )
    parser.add_argument("--data", type=Path, required=True)
    parser.add_argument("--cache-dir", type=Path, required=True)
    parser.add_argument("--assignment-csv", type=Path, required=True)
    parser.add_argument("--efficientvim-checkpoint", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--dino-weight", type=Path, default=None)
    parser.add_argument("--official-source-root", type=Path, default=None)
    parser.add_argument("--preflight-only", action="store_true")
    parser.add_argument("--preflight-artifact", type=Path, default=None)
    parser.add_argument("--preflight-sha256", type=str, default="")
    parser.add_argument("--batch-size", type=int, default=BATCH_SIZE)
    parser.add_argument("--workers", type=int, default=WORKERS)
    parser.add_argument("--device", type=str, default="")
    parser.add_argument("--torch-threads", type=int, default=4)
    return parser.parse_args(argv)


def _repository_root() -> Path:
    return Path(__file__).resolve().parents[2]


def _source_paths() -> dict[str, Path]:
    root = _repository_root()
    return {
        "runner": Path(__file__).resolve(),
        "protocol": root
        / "docs"
        / "TRKH_PRETRAINED_CLASSF_B13_EFFICIENTVIM_FROZEN_TRANSFER_PROTOCOL_20260803.md",
        "efficientvim_model": root / "trkh" / "models" / "efficientvim_m1.py",
        "efficientvim_test": root / "tests" / "test_efficientvim_m1.py",
        "b13_test": root
        / "tests"
        / "test_audit_efficientvim_m1_frozen_transfer_b13.py",
        "a0_precheck": root
        / "trkh"
        / "tools"
        / "precheck_dinov3_xcnorm_a0_sourcefold.py",
        "cache_guard": root
        / "trkh"
        / "tools"
        / "screen_dinov3_cgaer_b11_sourcefold.py",
    }


def _source_hashes() -> dict[str, str]:
    paths = _source_paths()
    missing = [name for name, path in paths.items() if not path.is_file()]
    if missing:
        raise FileNotFoundError(f"B13 bound source files are missing: {missing}")
    root = _repository_root()
    for name, path in paths.items():
        relative = path.relative_to(root)
        tracked = subprocess.run(
            ["git", "ls-files", "--error-unmatch", str(relative)],
            cwd=root,
            capture_output=True,
            text=True,
        )
        if tracked.returncode != 0:
            raise RuntimeError(f"B13 bound source is not tracked: {name}={relative}")
    return {name: _sha256(path) for name, path in paths.items()}


def _git_contract() -> dict[str, object]:
    root = _repository_root()
    head = subprocess.run(
        ["git", "rev-parse", "HEAD"],
        cwd=root,
        check=True,
        capture_output=True,
        text=True,
    ).stdout.strip()
    branch = subprocess.run(
        ["git", "branch", "--show-current"],
        cwd=root,
        check=True,
        capture_output=True,
        text=True,
    ).stdout.strip()
    dirty = subprocess.run(
        ["git", "status", "--porcelain"],
        cwd=root,
        check=True,
        capture_output=True,
        text=True,
    ).stdout.strip()
    common = subprocess.run(
        ["git", "rev-parse", "--git-common-dir"],
        cwd=root,
        check=True,
        capture_output=True,
        text=True,
    ).stdout.strip()
    return {
        "head": head,
        "branch": branch,
        "common_git_dir": common,
        "tracked_worktree_clean": not bool(dirty),
        "dirty": dirty,
    }


def _dependency_contract() -> dict[str, object]:
    onnxruntime_distribution: dict[str, str] | None = None
    for distribution in ("onnxruntime", "onnxruntime-gpu"):
        try:
            version = importlib.metadata.version(distribution)
        except importlib.metadata.PackageNotFoundError:
            continue
        onnxruntime_distribution = {
            "distribution": distribution,
            "version": version,
        }
        break
    if onnxruntime_distribution is None:
        raise importlib.metadata.PackageNotFoundError(
            "Neither onnxruntime nor onnxruntime-gpu is installed"
        )
    if onnxruntime_distribution["version"] != str(ort.__version__):
        raise RuntimeError("ONNX Runtime module and distribution versions differ")
    return {
        "python": platform.python_version(),
        "python_executable": str(Path(sys.executable).resolve()),
        "numpy": importlib.metadata.version("numpy"),
        "onnx": importlib.metadata.version("onnx"),
        "onnxruntime": onnxruntime_distribution,
        "pillow": importlib.metadata.version("pillow"),
        "safetensors": importlib.metadata.version("safetensors"),
        "scikit_learn": importlib.metadata.version("scikit-learn"),
        "timm": importlib.metadata.version("timm"),
        "torch": str(torch.__version__),
        "torchvision": importlib.metadata.version("torchvision"),
        "torch_cuda_runtime": str(torch.version.cuda or ""),
        "cudnn": int(torch.backends.cudnn.version() or 0),
    }


def _resolve_dino_weight(explicit: Path | None) -> Path:
    if explicit is not None:
        path = explicit.expanduser().resolve()
    else:
        cached = try_to_load_from_cache(DINO_HF_REPO, "model.safetensors")
        if not isinstance(cached, str):
            raise FileNotFoundError(
                "Locked DINOv3 safetensors is absent from the offline HF cache"
            )
        path = Path(cached).resolve()
    if not path.is_file() or _sha256(path) != DINO_WEIGHT_SHA256:
        raise ValueError("DINOv3 weight is missing or its SHA-256 changed")
    return path


def _build_dino(weight_path: Path, *, num_classes: int = 0) -> nn.Module:
    model = timm.create_model(
        DINO_MODEL,
        pretrained=False,
        num_classes=int(num_classes),
        img_size=DINO_IMAGE_SIZE,
    )
    state = load_file(str(weight_path), device="cpu")
    if int(num_classes) == 0:
        model.load_state_dict(state, strict=True)
    else:
        result = model.load_state_dict(state, strict=False)
        if set(result.missing_keys) != {"head.weight", "head.bias"} or result.unexpected_keys:
            raise ValueError(f"DINO 5-class skeleton state mismatch: {result}")
    observed = {
        "features": int(getattr(model, "num_features", -1)),
        "prefix_tokens": int(getattr(model, "num_prefix_tokens", -1)),
        "grid": tuple(int(value) for value in model.patch_embed.grid_size),
    }
    expected = {
        "features": DINO_FEATURE_DIM,
        "prefix_tokens": DINO_PREFIX_TOKENS,
        "grid": (16, 16),
    }
    if observed != expected:
        raise ValueError(f"DINO architecture drifted: {observed} != {expected}")
    return model


def _copy_efficientvim_backbone(source: EfficientViMM1) -> EfficientViMM1:
    torch.manual_seed(SEED + 17)
    target = EfficientViMM1(num_classes=CLASSES)
    backbone = {
        key: value
        for key, value in source.state_dict().items()
        if not key.startswith("heads.")
    }
    result = target.load_state_dict(backbone, strict=False)
    expected_missing = {
        f"heads.{index}.{field}"
        for index in range(4)
        for field in ("weight", "bias")
    }
    if set(result.missing_keys) != expected_missing or result.unexpected_keys:
        raise ValueError(f"EfficientViM 5-class skeleton state mismatch: {result}")
    return target


def _build_efficientvim(checkpoint: Path) -> tuple[EfficientViMM1, str]:
    model = EfficientViMM1(num_classes=1000)
    digest = load_official_efficientvim_m1_weights(model, checkpoint)
    count = sum(parameter.numel() for parameter in model.parameters())
    if count != EFFICIENTVIM_1000_CLASS_PARAMS:
        raise ValueError(f"EfficientViM parameter count changed: {count}")
    return model, digest


def _device_contract(requested: str) -> dict[str, object]:
    device = _resolve_device(requested)
    payload: dict[str, object] = {
        "requested": requested,
        "resolved": str(device),
        "cuda_available": bool(torch.cuda.is_available()),
    }
    if device.type == "cuda":
        index = int(device.index if device.index is not None else torch.cuda.current_device())
        properties = torch.cuda.get_device_properties(index)
        payload.update(
            {
                "index": index,
                "name": str(properties.name),
                "total_memory": int(properties.total_memory),
                "capability": list(torch.cuda.get_device_capability(index)),
            }
        )
    return payload


def _validate_official_source(root: Path | None) -> dict[str, object]:
    if root is None:
        raise ValueError("B13 preflight requires --official-source-root")
    source = root.expanduser().resolve()
    models = source / "classification" / "models"
    files = {
        "EfficientViM.py": models / "EfficientViM.py",
        "utils.py": models / "utils.py",
        "LICENSE": source / "LICENSE",
    }
    observed = {name: _sha256(path) for name, path in files.items() if path.is_file()}
    if observed != OFFICIAL_SOURCE_HASHES:
        raise ValueError(
            f"Official EfficientViM source hashes changed: {observed}"
        )
    revision = subprocess.run(
        ["git", "rev-parse", "HEAD"],
        cwd=source,
        check=True,
        capture_output=True,
        text=True,
    ).stdout.strip()
    if revision != OFFICIAL_SOURCE_REVISION:
        raise ValueError(f"Official EfficientViM revision changed: {revision}")
    return {
        "root": str(source),
        "revision": revision,
        "sha256": observed,
        "license": "MIT",
    }


def _run_model_focused_tests(
    official_source_root: Path, efficientvim_checkpoint: Path
) -> dict[str, object]:
    command = [
        sys.executable,
        "-m",
        "pytest",
        "tests/test_efficientvim_m1.py",
        "tests/test_audit_efficientvim_m1_frozen_transfer_b13.py",
        "-q",
    ]
    environment = os.environ.copy()
    environment["TRKH_EFFICIENTVIM_OFFICIAL_ROOT"] = str(official_source_root)
    environment["TRKH_EFFICIENTVIM_CHECKPOINT"] = str(efficientvim_checkpoint)
    result = subprocess.run(
        command,
        cwd=_repository_root(),
        capture_output=True,
        text=True,
        timeout=180,
        env=environment,
    )
    combined = (result.stdout + "\n" + result.stderr).strip()
    passed = result.returncode == 0 and "skipped" not in combined.casefold()
    if not passed:
        raise RuntimeError(f"EfficientViM focused parity tests failed:\n{combined}")
    return {"command": command, "returncode": result.returncode, "output": combined}


class _SingleInputExportWrapper(nn.Module):
    def __init__(self, model: nn.Module) -> None:
        super().__init__()
        self.model = model

    def forward(self, images: Tensor) -> Tensor:
        return self.model(images)


def _export_onnx(model: nn.Module, sample: Tensor, path: Path) -> dict[str, object]:
    model.cpu().eval()
    export_model = _SingleInputExportWrapper(model).eval()
    with warnings.catch_warnings():
        warnings.filterwarnings("ignore", category=torch.jit.TracerWarning)
        torch.onnx.export(
            export_model,
            sample,
            str(path),
            input_names=["images"],
            output_names=["logits"],
            opset_version=17,
            do_constant_folding=True,
            dynamic_axes=None,
        )
    graph = onnx.load(str(path))
    onnx.checker.check_model(graph)
    domains = sorted({str(node.domain) for node in graph.graph.node})
    if any(domain not in {"", "ai.onnx"} for domain in domains):
        raise ValueError(f"ONNX graph uses a custom operator domain: {domains}")
    return {
        "bytes": int(path.stat().st_size),
        "sha256": _sha256(path),
        "operator_domains": domains,
        "operators": sorted({str(node.op_type) for node in graph.graph.node}),
        "export_signature": "single_tensor_forward_wrapper",
    }


def _ort_session(path: Path) -> ort.InferenceSession:
    options = ort.SessionOptions()
    options.intra_op_num_threads = ORT_THREADS
    options.inter_op_num_threads = 1
    options.execution_mode = ort.ExecutionMode.ORT_SEQUENTIAL
    options.graph_optimization_level = ort.GraphOptimizationLevel.ORT_ENABLE_ALL
    return ort.InferenceSession(
        str(path), sess_options=options, providers=["CPUExecutionProvider"]
    )


def _timed_ort(session: ort.InferenceSession, array: np.ndarray, iterations: int) -> list[float]:
    values: list[float] = []
    for _ in range(int(iterations)):
        start = time.perf_counter_ns()
        session.run(None, {"images": array})
        values.append((time.perf_counter_ns() - start) / 1e6)
    return values


def _mobile_preflight(
    dino_weight: Path, efficientvim_checkpoint: Path
) -> dict[str, object]:
    dino = _build_dino(dino_weight, num_classes=CLASSES).cpu().eval()
    efficientvim_source, _ = _build_efficientvim(efficientvim_checkpoint)
    efficientvim = _copy_efficientvim_backbone(efficientvim_source).cpu().eval()
    if sum(parameter.numel() for parameter in dino.parameters()) != DINO_5_CLASS_PARAMS:
        raise ValueError("DINO 5-class parameter count drifted")
    if sum(parameter.numel() for parameter in efficientvim.parameters()) != EFFICIENTVIM_5_CLASS_PARAMS:
        raise ValueError("EfficientViM 5-class parameter count drifted")

    torch.manual_seed(SEED + 29)
    dino_input = torch.randn(1, 3, DINO_IMAGE_SIZE, DINO_IMAGE_SIZE)
    efficientvim_input = torch.randn(
        1, 3, EFFICIENTVIM_IMAGE_SIZE, EFFICIENTVIM_IMAGE_SIZE
    )
    with torch.inference_mode():
        dino_torch = dino(dino_input).numpy()
        efficientvim_torch = efficientvim(efficientvim_input).numpy()
    if not np.isfinite(dino_torch).all() or not np.isfinite(efficientvim_torch).all():
        raise FloatingPointError("Synthetic PyTorch logits are non-finite")

    with tempfile.TemporaryDirectory(prefix="trkh_b13_onnx_") as temporary:
        temporary_root = Path(temporary)
        paths = {
            "dino": temporary_root / "dino.onnx",
            "efficientvim": temporary_root / "efficientvim.onnx",
        }
        exports = {
            "dino": _export_onnx(dino, dino_input, paths["dino"]),
            "efficientvim": _export_onnx(
                efficientvim, efficientvim_input, paths["efficientvim"]
            ),
        }
        sessions = {name: _ort_session(path) for name, path in paths.items()}
        arrays = {
            "dino": dino_input.numpy(),
            "efficientvim": efficientvim_input.numpy(),
        }
        torch_outputs = {"dino": dino_torch, "efficientvim": efficientvim_torch}
        parity: dict[str, object] = {}
        for name in ("dino", "efficientvim"):
            output = sessions[name].run(None, {"images": arrays[name]})[0]
            maximum = float(np.max(np.abs(output - torch_outputs[name])))
            parity[name] = {
                "max_abs": maximum,
                "argmax_equal": bool(
                    np.array_equal(output.argmax(axis=1), torch_outputs[name].argmax(axis=1))
                ),
            }
        for _ in range(ORT_WARMUPS):
            for name in ("dino", "efficientvim"):
                sessions[name].run(None, {"images": arrays[name]})
        timings = {"dino": [], "efficientvim": []}
        trial_means = {"dino": [], "efficientvim": []}
        for trial in range(ORT_TRIALS):
            order = ("efficientvim", "dino") if trial % 2 == 0 else ("dino", "efficientvim")
            for name in order:
                samples = _timed_ort(sessions[name], arrays[name], ORT_ITERATIONS)
                timings[name].extend(samples)
                trial_means[name].append(float(np.mean(samples)))
        latency = {
            name: {
                "median_ms": float(np.median(timings[name])),
                "p95_ms": float(np.quantile(timings[name], 0.95)),
                "trial_mean_ms": trial_means[name],
            }
            for name in timings
        }

    median_ratio = latency["efficientvim"]["median_ms"] / latency["dino"]["median_ms"]
    p95_ratio = latency["efficientvim"]["p95_ms"] / latency["dino"]["p95_ms"]
    parameter_ratio = EFFICIENTVIM_5_CLASS_PARAMS / DINO_5_CLASS_PARAMS
    checks = {
        "efficientvim_5class_params_exact": True,
        "parameter_ratio_lte_0p30": parameter_ratio <= 0.30,
        "onnx_standard_domains_only": all(
            all(domain in {"", "ai.onnx"} for domain in exports[name]["operator_domains"])
            for name in exports
        ),
        "onnx_parity_max_abs_lte_1e_5": all(
            float(parity[name]["max_abs"]) <= 1e-5 for name in parity
        ),
        "onnx_argmax_exact": all(bool(parity[name]["argmax_equal"]) for name in parity),
        "median_latency_ratio_lte_0p25": median_ratio <= 0.25,
        "p95_latency_ratio_lte_0p30": p95_ratio <= 0.30,
    }
    return {
        "settings": {
            "runtime": "onnxruntime_cpu",
            "threads": ORT_THREADS,
            "batch": 1,
            "warmups": ORT_WARMUPS,
            "trials": ORT_TRIALS,
            "iterations_per_trial": ORT_ITERATIONS,
            "dino_input": [1, 3, DINO_IMAGE_SIZE, DINO_IMAGE_SIZE],
            "efficientvim_input": [1, 3, EFFICIENTVIM_IMAGE_SIZE, EFFICIENTVIM_IMAGE_SIZE],
        },
        "parameters": {
            "dino_5class": DINO_5_CLASS_PARAMS,
            "efficientvim_5class": EFFICIENTVIM_5_CLASS_PARAMS,
            "ratio": parameter_ratio,
        },
        "exports": exports,
        "parity": parity,
        "latency": latency,
        "ratios": {"median": median_ratio, "p95": p95_ratio},
        "checks": checks,
        "passed": all(checks.values()),
    }


def _prepare_ledger(args: argparse.Namespace) -> dict[str, object]:
    if int(args.batch_size) != BATCH_SIZE or int(args.workers) != WORKERS:
        raise ValueError("B13 is locked to batch-size=32 and workers=0")
    data_yaml = args.data.expanduser().resolve()
    if _sha256(data_yaml) != EXPECTED_DATA_SHA256:
        raise ValueError("Canonical class_f data YAML hash changed")
    data_root = _data_root_from_yaml(data_yaml)
    cache = validate_a0_cache(args.cache_dir)
    labels = np.asarray(cache["labels"], dtype=np.int64)
    paths = list(cache["paths"])
    counts = tuple(np.bincount(labels, minlength=CLASSES).astype(int).tolist())
    if labels.size != EXPECTED_TRAIN_SAMPLES or counts != EXPECTED_CLASS_COUNTS:
        raise ValueError(f"B13 TRAIN support changed: rows={labels.size}, counts={counts}")
    assignment = read_locked_assignment(args.assignment_csv, paths, labels)
    if assignment["csv_sha256"] != EXPECTED_ASSIGNMENT_SHA256:
        raise ValueError("B13 assignment CSV hash changed")
    absolute_paths = [(data_root / path).resolve() for path in paths]
    assert_train_only_paths(absolute_paths, data_root / "train")
    if any(not path.is_file() for path in absolute_paths):
        raise FileNotFoundError("B13 TRAIN ledger references a missing image")
    train_content = _train_content_contract(paths, absolute_paths)
    return {
        "data_yaml": data_yaml,
        "data_root": data_root,
        "cache": cache,
        "labels": labels,
        "paths": paths,
        "absolute_paths": absolute_paths,
        "assignment": assignment,
        "train_content": train_content,
    }


def _train_content_contract(
    relative_paths: Sequence[str], absolute_paths: Sequence[Path]
) -> dict[str, object]:
    if len(relative_paths) != len(absolute_paths) or not relative_paths:
        raise ValueError("TRAIN content ledger is empty or misaligned")
    digest = hashlib.sha256()
    total_bytes = 0
    for relative, absolute in zip(relative_paths, absolute_paths):
        path = absolute.resolve()
        before = path.stat()
        file_digest = _sha256(path)
        after = path.stat()
        identity_before = (before.st_size, before.st_mtime_ns)
        identity_after = (after.st_size, after.st_mtime_ns)
        if identity_before != identity_after:
            raise RuntimeError(f"TRAIN image changed while hashing: {path}")
        encoded = str(relative).encode("utf-8")
        digest.update(len(encoded).to_bytes(8, "little"))
        digest.update(encoded)
        digest.update(int(before.st_size).to_bytes(8, "little"))
        digest.update(bytes.fromhex(file_digest))
        total_bytes += int(before.st_size)
    return {
        "algorithm": "ordered_path_length_file_sha256_v1",
        "files": len(relative_paths),
        "bytes": total_bytes,
        "sha256": digest.hexdigest(),
        "train_only": True,
    }


def _assert_output_outside_data_root(output_dir: Path, data_root: Path) -> None:
    output = output_dir.expanduser().resolve()
    immutable = data_root.expanduser().resolve()
    try:
        output.relative_to(immutable)
    except ValueError:
        return
    raise ValueError(f"B13 output must stay outside immutable dataset root: {output}")


def build_preflight(args: argparse.Namespace) -> dict[str, object]:
    ledger = _prepare_ledger(args)
    _assert_output_outside_data_root(args.output_dir, ledger["data_root"])
    git = _git_contract()
    if git["branch"] != "research/pretrained-classf-b1" or not git["tracked_worktree_clean"]:
        raise RuntimeError(f"B13 preflight requires the clean canonical branch: {git}")
    dino_weight = _resolve_dino_weight(args.dino_weight)
    efficientvim_checkpoint = args.efficientvim_checkpoint.expanduser().resolve()
    efficientvim, efficientvim_digest = _build_efficientvim(efficientvim_checkpoint)
    dino = _build_dino(dino_weight)
    dino_data_config = timm.data.resolve_model_data_config(dino)
    torch.manual_seed(SEED)
    with torch.inference_mode():
        dino_feature = _dino_descriptor(
            dino, torch.randn(1, 3, DINO_IMAGE_SIZE, DINO_IMAGE_SIZE)
        )
        efficientvim_feature = efficientvim.forward_final_features(
            torch.randn(1, 3, EFFICIENTVIM_IMAGE_SIZE, EFFICIENTVIM_IMAGE_SIZE)
        )
    official_source = _validate_official_source(args.official_source_root)
    focused_tests = _run_model_focused_tests(
        Path(official_source["root"]), efficientvim_checkpoint
    )
    mobile = _mobile_preflight(dino_weight, efficientvim_checkpoint)
    checks = {
        "git_clean_canonical_branch": True,
        "train_ledger_exact": True,
        "validation_constructed": False,
        "test_constructed": False,
        "dino_strict_weight": True,
        "efficientvim_strict_ema_weight": efficientvim_digest == OFFICIAL_M1_E450_SHA256,
        "descriptor_shapes_exact": tuple(dino_feature.shape) == (1, DINO_FEATURE_DIM)
        and tuple(efficientvim_feature.shape) == (1, EFFICIENTVIM_FEATURE_DIM),
        "official_parity_tests": True,
        "mobile_precondition": bool(mobile["passed"]),
    }
    payload = {
        "schema_version": 1,
        "protocol_id": PROTOCOL_ID,
        "scientific_scope": "raw_frozen_train_only_mobile_representation_screen",
        "created_at_unix": time.time(),
        "source_hashes": _source_hashes(),
        "git": git,
        "dependencies": _dependency_contract(),
        "device": _device_contract(args.device),
        "dataset": {
            "data_yaml": str(ledger["data_yaml"]),
            "data_yaml_sha256": EXPECTED_DATA_SHA256,
            "train_rows": EXPECTED_TRAIN_SAMPLES,
            "class_names": list(EXPECTED_CLASSES),
            "class_counts": list(EXPECTED_CLASS_COUNTS),
            "train_split_used": True,
            "validation_split_used": False,
            "test_split_used": False,
            "train_content": ledger["train_content"],
        },
        "cache_hashes": ledger["cache"]["hashes"],
        "assignment": {
            key: ledger["assignment"][key]
            for key in (
                "csv_sha256",
                "assignment_int64_sha256",
                "path_fold_sha256",
                "group_vector_int64_sha256",
                "fold_rows",
            )
        },
        "weights": {
            "dino": {"path": str(dino_weight), "sha256": DINO_WEIGHT_SHA256},
            "efficientvim": {
                "path": str(efficientvim_checkpoint),
                "sha256": efficientvim_digest,
                "state_key": "model_ema",
            },
        },
        "preprocessing": {
            "dino": {
                "resize": [DINO_IMAGE_SIZE, DINO_IMAGE_SIZE],
                "interpolation": "bicubic",
                "crop": None,
                "mean": [float(value) for value in dino_data_config["mean"]],
                "std": [float(value) for value in dino_data_config["std"]],
                "descriptor": "mean_final_postnorm_patch_tokens_after_5_prefix_tokens",
                "dimension": DINO_FEATURE_DIM,
            },
            "efficientvim": {
                "resize_short_side": 256,
                "center_crop": EFFICIENTVIM_IMAGE_SIZE,
                "interpolation": "bicubic",
                "mean": [0.485, 0.456, 0.406],
                "std": [0.229, 0.224, 0.225],
                "descriptor": "final_norm3_global_average_pool",
                "dimension": EFFICIENTVIM_FEATURE_DIM,
            },
        },
        "official_source": official_source,
        "focused_tests": focused_tests,
        "mobile": mobile,
        "checks": checks,
        "passed": all(checks.values()),
    }
    return payload


def _validate_preflight(
    path: Path,
    expected_sha256: str,
    *,
    ledger: Mapping[str, object],
    dino_weight: Path,
    efficientvim_checkpoint: Path,
    requested_device: str,
) -> dict[str, object]:
    artifact = path.expanduser().resolve()
    digest = _sha256(artifact)
    if digest != expected_sha256.strip().lower() or len(digest) != 64:
        raise ValueError("B13 preflight SHA-256 mismatch")
    payload = json.loads(artifact.read_text(encoding="utf-8"))
    if not isinstance(payload, Mapping) or payload.get("protocol_id") != PROTOCOL_ID:
        raise ValueError("B13 preflight protocol mismatch")
    current_git = _git_contract()
    expected_dataset = {
        "data_yaml": str(ledger["data_yaml"]),
        "data_yaml_sha256": EXPECTED_DATA_SHA256,
        "train_rows": EXPECTED_TRAIN_SAMPLES,
        "class_names": list(EXPECTED_CLASSES),
        "class_counts": list(EXPECTED_CLASS_COUNTS),
        "train_split_used": True,
        "validation_split_used": False,
        "test_split_used": False,
        "train_content": ledger["train_content"],
    }
    expected_assignment = {
        key: ledger["assignment"][key]
        for key in (
            "csv_sha256",
            "assignment_int64_sha256",
            "path_fold_sha256",
            "group_vector_int64_sha256",
            "fold_rows",
        )
    }
    payload_checks = payload.get("checks", {})
    mobile = payload.get("mobile", {})
    mobile_checks = mobile.get("checks", {}) if isinstance(mobile, Mapping) else {}
    weights = payload.get("weights", {})
    official_source = payload.get("official_source", {})
    checks = {
        "preflight_passed": payload.get("passed") is True,
        "preflight_checks_all_true": isinstance(payload_checks, Mapping)
        and bool(payload_checks)
        and all(value is True for value in payload_checks.values()),
        "source_hashes": payload.get("source_hashes") == _source_hashes(),
        "git_head": payload.get("git", {}).get("head") == current_git["head"],
        "git_clean": current_git["tracked_worktree_clean"] is True,
        "branch": current_git["branch"] == "research/pretrained-classf-b1",
        "dependencies_exact": payload.get("dependencies") == _dependency_contract(),
        "device_exact": payload.get("device") == _device_contract(requested_device),
        "dataset_exact_train_only": payload.get("dataset") == expected_dataset,
        "cache_hashes_exact": payload.get("cache_hashes") == ledger["cache"]["hashes"],
        "assignment_exact": payload.get("assignment") == expected_assignment,
        "dino_weight_exact": isinstance(weights, Mapping)
        and weights.get("dino", {}).get("sha256") == DINO_WEIGHT_SHA256
        and _sha256(dino_weight) == DINO_WEIGHT_SHA256,
        "efficientvim_weight_exact": isinstance(weights, Mapping)
        and weights.get("efficientvim", {}).get("sha256") == OFFICIAL_M1_E450_SHA256
        and weights.get("efficientvim", {}).get("state_key") == "model_ema"
        and _sha256(efficientvim_checkpoint) == OFFICIAL_M1_E450_SHA256,
        "official_source_exact": isinstance(official_source, Mapping)
        and official_source.get("revision") == OFFICIAL_SOURCE_REVISION
        and official_source.get("sha256") == OFFICIAL_SOURCE_HASHES
        and official_source.get("license") == "MIT",
        "focused_tests_passed": payload.get("focused_tests", {}).get("returncode") == 0,
        "mobile_passed": isinstance(mobile, Mapping)
        and mobile.get("passed") is True
        and isinstance(mobile_checks, Mapping)
        and bool(mobile_checks)
        and all(value is True for value in mobile_checks.values()),
    }
    if not all(checks.values()):
        raise ValueError(f"B13 preflight validation failed: {checks}")
    return {"artifact": str(artifact), "sha256": digest, "checks": checks, "payload": payload}


class _TrainLedgerDataset(Dataset[tuple[Tensor, int, int]]):
    def __init__(
        self,
        paths: Sequence[Path],
        labels: np.ndarray,
        transform: Any,
    ) -> None:
        self.paths = list(paths)
        self.labels = np.asarray(labels, dtype=np.int64)
        self.transform = transform
        if len(self.paths) != self.labels.size:
            raise ValueError("TRAIN ledger paths/labels are not aligned")

    def __len__(self) -> int:
        return len(self.paths)

    def __getitem__(self, index: int) -> tuple[Tensor, int, int]:
        with Image.open(self.paths[index]) as image:
            rgb = ImageOps.exif_transpose(image).convert("RGB")
            tensor = self.transform(rgb)
        return tensor, int(self.labels[index]), int(index)


def _dino_transform(model: nn.Module) -> Any:
    config = timm.data.resolve_model_data_config(model)
    return v2.Compose(
        (
            v2.Resize(
                (DINO_IMAGE_SIZE, DINO_IMAGE_SIZE),
                interpolation=InterpolationMode.BICUBIC,
                antialias=True,
            ),
            v2.ToImage(),
            v2.ToDtype(torch.float32, scale=True),
            v2.Normalize(
                mean=tuple(float(value) for value in config["mean"]),
                std=tuple(float(value) for value in config["std"]),
            ),
        )
    )


def _efficientvim_transform() -> Any:
    return v2.Compose(
        (
            v2.Resize(
                256,
                interpolation=InterpolationMode.BICUBIC,
                antialias=True,
            ),
            v2.CenterCrop(EFFICIENTVIM_IMAGE_SIZE),
            v2.ToImage(),
            v2.ToDtype(torch.float32, scale=True),
            v2.Normalize(
                mean=(0.485, 0.456, 0.406),
                std=(0.229, 0.224, 0.225),
            ),
        )
    )


def _dino_descriptor(model: nn.Module, images: Tensor) -> Tensor:
    tokens = model.forward_features(images)
    if not isinstance(tokens, Tensor) or tuple(tokens.shape[1:]) != (261, DINO_FEATURE_DIM):
        raise ValueError(f"DINO token geometry changed: {getattr(tokens, 'shape', None)}")
    return tokens[:, DINO_PREFIX_TOKENS:].mean(dim=1).float()


def _extract_descriptors(
    *,
    name: str,
    model: nn.Module,
    dataset: Dataset[tuple[Tensor, int, int]],
    feature_dim: int,
    device: torch.device,
    batch_size: int,
    workers: int,
) -> np.ndarray:
    loader = DataLoader(
        dataset,
        batch_size=int(batch_size),
        shuffle=False,
        num_workers=int(workers),
        pin_memory=device.type == "cuda",
        drop_last=False,
    )
    output = np.full((len(dataset), feature_dim), np.nan, dtype=np.float32)
    expected_index = 0
    total_batches = len(loader)
    next_report = 10
    model.to(device).eval()
    with torch.inference_mode():
        for batch_index, (images, _labels, indices) in enumerate(loader, start=1):
            index_array = indices.numpy().astype(np.int64, copy=False)
            expected = np.arange(
                expected_index, expected_index + index_array.size, dtype=np.int64
            )
            if not np.array_equal(index_array, expected):
                raise ValueError(f"{name} DataLoader order changed")
            images = images.to(device=device, dtype=torch.float32, non_blocking=True)
            if name == "dino":
                features = _dino_descriptor(model, images)
            else:
                if not isinstance(model, EfficientViMM1):
                    raise TypeError("EfficientViM descriptor received the wrong model")
                features = model.forward_final_features(images).float()
            values = features.detach().cpu().numpy().astype(np.float32, copy=False)
            if values.shape != (index_array.size, feature_dim):
                raise ValueError(f"{name} descriptor shape changed: {values.shape}")
            output[index_array] = values
            expected_index += index_array.size
            percent = int(100 * batch_index / max(total_batches, 1))
            if percent >= next_report:
                print(
                    f"{name}: {percent}% ({expected_index}/{len(dataset)})",
                    flush=True,
                )
                next_report += 10
    model.cpu()
    if expected_index != len(dataset) or not np.isfinite(output).all():
        raise RuntimeError(f"{name} descriptor extraction is incomplete/non-finite")
    return output


def _atomic_npy(path: Path, array: np.ndarray) -> str:
    path.parent.mkdir(parents=True, exist_ok=True)
    partial = path.with_suffix(path.suffix + ".partial")
    if path.exists() or partial.exists():
        raise RuntimeError(f"refuse to overwrite descriptor artifact: {path}")
    with partial.open("wb") as handle:
        np.save(handle, array, allow_pickle=False)
    digest = _sha256(partial)
    partial.replace(path)
    if _sha256(path) != digest:
        raise RuntimeError("descriptor digest changed during atomic promotion")
    return digest


def _softmax(values: np.ndarray) -> np.ndarray:
    shifted = values - np.max(values, axis=1, keepdims=True)
    exponent = np.exp(shifted)
    return exponent / exponent.sum(axis=1, keepdims=True)


def fit_oof_readout(
    features: np.ndarray,
    labels: np.ndarray,
    folds: np.ndarray,
) -> dict[str, object]:
    features = np.asarray(features, dtype=np.float32)
    labels = np.asarray(labels, dtype=np.int64)
    folds = np.asarray(folds, dtype=np.int64)
    if features.ndim != 2 or features.shape[0] != labels.size or labels.shape != folds.shape:
        raise ValueError("B13 readout arrays are not aligned")
    scores = np.full((labels.size, CLASSES), np.nan, dtype=np.float64)
    coefficients = np.full((FOLDS, CLASSES, features.shape[1]), np.nan, dtype=np.float64)
    intercepts = np.full((FOLDS, CLASSES), np.nan, dtype=np.float64)
    scaler_means = np.full((FOLDS, features.shape[1]), np.nan, dtype=np.float64)
    scaler_scales = np.full((FOLDS, features.shape[1]), np.nan, dtype=np.float64)
    fold_records: list[dict[str, object]] = []
    converged = True
    for fold in range(FOLDS):
        fit = folds != fold
        held = folds == fold
        scaler = StandardScaler()
        fit_values = scaler.fit_transform(features[fit].astype(np.float64, copy=False))
        held_values = scaler.transform(features[held].astype(np.float64, copy=False))
        classifier = LogisticRegression(
            C=LOGISTIC_C,
            class_weight="balanced",
            solver="lbfgs",
            tol=LOGISTIC_TOL,
            max_iter=LOGISTIC_MAX_ITER,
            random_state=SEED,
        )
        with warnings.catch_warnings(record=True) as caught:
            warnings.simplefilter("always", ConvergenceWarning)
            classifier.fit(fit_values, labels[fit])
        fold_converged = not any(
            issubclass(record.category, ConvergenceWarning) for record in caught
        ) and int(np.max(classifier.n_iter_)) < LOGISTIC_MAX_ITER
        converged = converged and fold_converged
        held_scores = classifier.decision_function(held_values)
        if held_scores.shape != (int(held.sum()), CLASSES):
            raise ValueError("B13 multinomial decision score shape changed")
        scores[held] = held_scores
        coefficients[fold] = classifier.coef_
        intercepts[fold] = classifier.intercept_
        scaler_means[fold] = scaler.mean_
        scaler_scales[fold] = scaler.scale_
        fold_records.append(
            {
                "fold": fold,
                "fit_rows": int(fit.sum()),
                "held_rows": int(held.sum()),
                "n_iter": classifier.n_iter_.astype(int).tolist(),
                "converged": fold_converged,
            }
        )
    if not np.isfinite(scores).all():
        raise FloatingPointError("B13 OOF scores are incomplete/non-finite")
    return {
        "scores": scores,
        "probabilities": _softmax(scores),
        "coefficients": coefficients,
        "intercepts": intercepts,
        "scaler_means": scaler_means,
        "scaler_scales": scaler_scales,
        "fold_records": fold_records,
        "converged": converged,
    }


def classification_summary(
    labels: np.ndarray,
    scores: np.ndarray,
    folds: np.ndarray | None = None,
) -> dict[str, object]:
    labels = np.asarray(labels, dtype=np.int64)
    scores = np.asarray(scores, dtype=np.float64)
    predictions = scores.argmax(axis=1).astype(np.int64)
    matrix = confusion_matrix(labels, predictions, labels=np.arange(CLASSES))
    precision, recall, f1, support = precision_recall_fscore_support(
        labels,
        predictions,
        labels=np.arange(CLASSES),
        zero_division=0,
    )
    pairs: dict[str, object] = {}
    for rival in RIVALS:
        mask = (labels == CLASS1) | (labels == rival)
        pair_labels = (labels[mask] == CLASS1).astype(np.int64)
        margin = scores[mask, CLASS1] - scores[mask, rival]
        pairs[str(rival)] = {
            "auroc": float(roc_auc_score(pair_labels, margin)),
            "rows": int(mask.sum()),
        }
    restricted_support = int(np.isin(labels, RIVALS).sum())
    restricted_fp = int(sum(matrix[rival, CLASS1] for rival in RIVALS))
    payload: dict[str, object] = {
        "accuracy": float(accuracy_score(labels, predictions)),
        "macro_f1": float(np.mean(f1)),
        "class1_precision": float(precision[CLASS1]),
        "class1_recall": float(recall[CLASS1]),
        "class1_f1": float(f1[CLASS1]),
        "class1_tp": int(matrix[CLASS1, CLASS1]),
        "class1_fn": int(support[CLASS1] - matrix[CLASS1, CLASS1]),
        "restricted_fp": restricted_fp,
        "restricted_support": restricted_support,
        "restricted_fp_rate": restricted_fp / max(restricted_support, 1),
        "confusion_matrix": matrix.astype(int).tolist(),
        "per_class": [
            {
                "class": index,
                "precision": float(precision[index]),
                "recall": float(recall[index]),
                "f1": float(f1[index]),
                "support": int(support[index]),
            }
            for index in range(CLASSES)
        ],
        "pairs": pairs,
        "mean_pair_auroc": float(
            np.mean([float(pairs[str(rival)]["auroc"]) for rival in RIVALS])
        ),
    }
    if folds is not None:
        payload["folds"] = [
            {"fold": fold, **classification_summary(labels[folds == fold], scores[folds == fold])}
            for fold in range(FOLDS)
        ]
    return payload


def _fast_f1_from_predictions(labels: np.ndarray, predictions: np.ndarray) -> tuple[float, float]:
    matrix = np.bincount(
        labels.astype(np.int64) * CLASSES + predictions.astype(np.int64),
        minlength=CLASSES * CLASSES,
    ).reshape(CLASSES, CLASSES)
    tp = np.diag(matrix).astype(np.float64)
    fp = matrix.sum(axis=0) - tp
    fn = matrix.sum(axis=1) - tp
    denominator = 2 * tp + fp + fn
    f1 = np.divide(2 * tp, denominator, out=np.zeros_like(tp), where=denominator > 0)
    return float(np.mean(f1)), float(f1[CLASS1])


def paired_component_bootstrap(
    *,
    labels: np.ndarray,
    folds: np.ndarray,
    groups: np.ndarray,
    dino_scores: np.ndarray,
    candidate_scores: np.ndarray,
    replicates: int = BOOTSTRAP_REPLICATES,
    seed: int = SEED,
) -> dict[str, object]:
    if int(replicates) <= 0:
        raise ValueError("B13 bootstrap replicate count must be positive")
    labels = np.asarray(labels, dtype=np.int64)
    folds = np.asarray(folds, dtype=np.int64)
    groups = np.asarray(groups, dtype=np.int64)
    dino_scores = np.asarray(dino_scores, dtype=np.float64)
    candidate_scores = np.asarray(candidate_scores, dtype=np.float64)
    if (
        labels.ndim != 1
        or labels.shape != folds.shape
        or labels.shape != groups.shape
        or dino_scores.shape != (labels.size, CLASSES)
        or candidate_scores.shape != (labels.size, CLASSES)
    ):
        raise ValueError("B13 bootstrap arrays are not aligned")
    if (
        set(np.unique(folds).tolist()) != set(range(FOLDS))
        or bool((groups < 0).any())
        or bool((labels < 0).any())
        or bool((labels >= CLASSES).any())
        or not np.isfinite(dino_scores).all()
        or not np.isfinite(candidate_scores).all()
    ):
        raise ValueError("B13 bootstrap fold/group/score domain is invalid")
    members: dict[int, np.ndarray] = {}
    fold_groups: dict[int, list[int]] = {fold: [] for fold in range(FOLDS)}
    for group in np.unique(groups):
        positions = np.flatnonzero(groups == group)
        observed = np.unique(folds[positions])
        if observed.size != 1:
            raise ValueError("B13 bootstrap component crosses folds")
        members[int(group)] = positions
        fold_groups[int(observed[0])].append(int(group))
    values = {
        "macro_f1_delta": np.empty(int(replicates), dtype=np.float64),
        "class1_f1_delta": np.empty(int(replicates), dtype=np.float64),
        "mean_pair_auroc_delta": np.empty(int(replicates), dtype=np.float64),
    }
    predictions = {
        "dino": dino_scores.argmax(axis=1),
        "candidate": candidate_scores.argmax(axis=1),
    }
    rng = np.random.default_rng(int(seed))
    draw_hash = hashlib.sha256()
    for replicate in range(int(replicates)):
        chunks: list[np.ndarray] = []
        for fold in range(FOLDS):
            available = fold_groups[fold]
            if not available:
                raise ValueError(f"B13 bootstrap fold {fold} has no union component")
            draw = rng.integers(0, len(available), size=len(available), dtype=np.int64)
            draw_hash.update(np.asarray((replicate, fold), dtype="<i8").tobytes())
            draw_hash.update(draw.astype("<i8").tobytes())
            chunks.extend(members[available[int(position)]] for position in draw)
        selected = np.concatenate(chunks)
        dino_macro, dino_class1 = _fast_f1_from_predictions(
            labels[selected], predictions["dino"][selected]
        )
        candidate_macro, candidate_class1 = _fast_f1_from_predictions(
            labels[selected], predictions["candidate"][selected]
        )
        values["macro_f1_delta"][replicate] = candidate_macro - dino_macro
        values["class1_f1_delta"][replicate] = candidate_class1 - dino_class1
        aucs: dict[str, list[float]] = {"dino": [], "candidate": []}
        for rival in RIVALS:
            mask = (labels[selected] == CLASS1) | (labels[selected] == rival)
            binary = (labels[selected][mask] == CLASS1).astype(np.int64)
            for name, scores in (("dino", dino_scores), ("candidate", candidate_scores)):
                margin = scores[selected][mask, CLASS1] - scores[selected][mask, rival]
                aucs[name].append(float(roc_auc_score(binary, margin)))
        values["mean_pair_auroc_delta"][replicate] = float(
            np.mean(aucs["candidate"]) - np.mean(aucs["dino"])
        )
    intervals = {
        name: {
            "lower": float(np.quantile(sample, 0.025)),
            "upper": float(np.quantile(sample, 0.975)),
        }
        for name, sample in values.items()
    }
    return {
        "method": "paired_fold_stratified_union_component_percentile_bootstrap",
        "replicates": int(replicates),
        "seed": int(seed),
        "draws_int64_sha256": draw_hash.hexdigest(),
        "intervals": intervals,
    }


def assess_gate(
    *,
    dino: Mapping[str, object],
    candidate: Mapping[str, object],
    bootstrap: Mapping[str, object],
    integrity_complete: bool,
    readouts_converged: bool,
) -> dict[str, object]:
    pair_delta = float(candidate["mean_pair_auroc"]) - float(dino["mean_pair_auroc"])
    macro_delta = float(candidate["macro_f1"]) - float(dino["macro_f1"])
    class1_delta = float(candidate["class1_f1"]) - float(dino["class1_f1"])
    intervals = bootstrap["intervals"]
    dino_recall = float(dino["class1_recall"])
    dino_fp_rate = float(dino["restricted_fp_rate"])
    candidate_fp_rate = float(candidate["restricted_fp_rate"])
    dino_folds = {int(row["fold"]): row for row in dino["folds"]}
    candidate_folds = {int(row["fold"]): row for row in candidate["folds"]}
    fold_deltas = [
        float(candidate_folds[fold]["class1_f1"])
        - float(dino_folds[fold]["class1_f1"])
        for fold in range(FOLDS)
    ]
    checks = {
        "pair_auroc_noninferiority": pair_delta >= -0.005
        and float(intervals["mean_pair_auroc_delta"]["lower"]) >= -0.010,
        "macro_f1_noninferiority": macro_delta >= -0.025
        and float(intervals["macro_f1_delta"]["lower"]) >= -0.040,
        "class1_f1_noninferiority": class1_delta >= -0.035
        and float(intervals["class1_f1_delta"]["lower"]) >= -0.060,
        "class1_recall_retention": float(candidate["class1_recall"]) >= 0.90 * dino_recall,
        "restricted_fp_rate_control": candidate_fp_rate <= 1.25 * dino_fp_rate,
        "fold_class1_noninferiority": sum(delta >= -0.050 for delta in fold_deltas) >= 3,
        "readouts_converged": bool(readouts_converged),
        "integrity_complete_train_only": bool(integrity_complete),
    }
    passed = all(checks.values())
    return {
        "signal_gate_passed": passed,
        "fine_tune_protocol_permission": passed,
        "validation_permission": False,
        "test_permission": False,
        "full_train_permission": False,
        "exact_m1_final_feature_route_closed": not passed,
        "checks": checks,
        "failed_checks": [name for name, value in checks.items() if not value],
        "point_deltas": {
            "mean_pair_auroc": pair_delta,
            "macro_f1": macro_delta,
            "class1_f1": class1_delta,
            "class1_recall_ratio": float(candidate["class1_recall"]) / max(dino_recall, 1e-12),
            "restricted_fp_rate_ratio": candidate_fp_rate / max(dino_fp_rate, 1e-12),
            "class1_f1_by_fold": fold_deltas,
        },
    }


def _run_formal(args: argparse.Namespace) -> dict[str, object]:
    if args.preflight_artifact is None or not args.preflight_sha256:
        raise ValueError("Formal B13 requires accepted preflight path and SHA-256")
    output_dir = args.output_dir.expanduser().resolve()
    if output_dir.exists():
        raise RuntimeError(f"refuse to overwrite B13 output directory: {output_dir}")
    ledger = _prepare_ledger(args)
    _assert_output_outside_data_root(output_dir, ledger["data_root"])
    dino_weight = _resolve_dino_weight(args.dino_weight)
    efficientvim_checkpoint = args.efficientvim_checkpoint.expanduser().resolve()
    if _sha256(efficientvim_checkpoint) != OFFICIAL_M1_E450_SHA256:
        raise ValueError("EfficientViM checkpoint changed after preflight")
    accepted = _validate_preflight(
        args.preflight_artifact,
        args.preflight_sha256,
        ledger=ledger,
        dino_weight=dino_weight,
        efficientvim_checkpoint=efficientvim_checkpoint,
        requested_device=args.device,
    )
    start_source_hashes = _source_hashes()
    start_git = _git_contract()
    start_dependencies = _dependency_contract()
    start_device = _device_contract(args.device)
    torch.set_num_threads(int(args.torch_threads))
    torch.manual_seed(SEED)
    np.random.seed(SEED)
    torch.use_deterministic_algorithms(True)
    torch.backends.cudnn.benchmark = False
    device = _resolve_device(args.device)
    dino = _build_dino(dino_weight)
    efficientvim, _ = _build_efficientvim(efficientvim_checkpoint)
    dino_dataset = _TrainLedgerDataset(
        ledger["absolute_paths"], ledger["labels"], _dino_transform(dino)
    )
    efficientvim_dataset = _TrainLedgerDataset(
        ledger["absolute_paths"], ledger["labels"], _efficientvim_transform()
    )
    started = time.perf_counter()
    dino_features = _extract_descriptors(
        name="dino",
        model=dino,
        dataset=dino_dataset,
        feature_dim=DINO_FEATURE_DIM,
        device=device,
        batch_size=args.batch_size,
        workers=args.workers,
    )
    if device.type == "cuda":
        torch.cuda.empty_cache()
    efficientvim_features = _extract_descriptors(
        name="efficientvim",
        model=efficientvim,
        dataset=efficientvim_dataset,
        feature_dim=EFFICIENTVIM_FEATURE_DIM,
        device=device,
        batch_size=args.batch_size,
        workers=args.workers,
    )
    output_dir.mkdir(parents=True, exist_ok=False)
    descriptor_hashes = {
        "dino": _atomic_npy(output_dir / "train_dino_final_f32.npy", dino_features),
        "efficientvim": _atomic_npy(
            output_dir / "train_efficientvim_final_f32.npy", efficientvim_features
        ),
    }
    _atomic_json(output_dir / "train_paths.json", ledger["paths"])
    paths_sha256 = _sha256(output_dir / "train_paths.json")
    folds = np.asarray(ledger["assignment"]["folds"], dtype=np.int64)
    groups = np.asarray(ledger["assignment"]["groups"], dtype=np.int64)
    labels = np.asarray(ledger["labels"], dtype=np.int64)
    readouts = {
        "dino": fit_oof_readout(dino_features, labels, folds),
        "efficientvim": fit_oof_readout(efficientvim_features, labels, folds),
    }
    summaries = {
        name: classification_summary(labels, result["scores"], folds)
        for name, result in readouts.items()
    }
    bootstrap = paired_component_bootstrap(
        labels=labels,
        folds=folds,
        groups=groups,
        dino_scores=readouts["dino"]["scores"],
        candidate_scores=readouts["efficientvim"]["scores"],
    )
    end_train_content = _train_content_contract(
        ledger["paths"], ledger["absolute_paths"]
    )
    end_source_hashes = _source_hashes()
    end_git = _git_contract()
    integrity_complete = bool(
        np.isfinite(dino_features).all()
        and np.isfinite(efficientvim_features).all()
        and all(np.isfinite(result["scores"]).all() for result in readouts.values())
        and ledger["assignment"]["csv_sha256"] == EXPECTED_ASSIGNMENT_SHA256
        and end_source_hashes == start_source_hashes
        and end_git == start_git
        and end_git["tracked_worktree_clean"] is True
        and end_train_content == ledger["train_content"]
    )
    gate = assess_gate(
        dino=summaries["dino"],
        candidate=summaries["efficientvim"],
        bootstrap=bootstrap,
        integrity_complete=integrity_complete,
        readouts_converged=all(bool(result["converged"]) for result in readouts.values()),
    )
    oof_sha = _atomic_npz(
        output_dir / "train_oof_readouts.npz",
        labels=labels,
        folds=folds,
        groups=groups,
        dino_scores=np.asarray(readouts["dino"]["scores"], dtype=np.float64),
        efficientvim_scores=np.asarray(
            readouts["efficientvim"]["scores"], dtype=np.float64
        ),
        dino_coefficients=np.asarray(readouts["dino"]["coefficients"]),
        dino_intercepts=np.asarray(readouts["dino"]["intercepts"]),
        dino_scaler_means=np.asarray(readouts["dino"]["scaler_means"]),
        dino_scaler_scales=np.asarray(readouts["dino"]["scaler_scales"]),
        efficientvim_coefficients=np.asarray(
            readouts["efficientvim"]["coefficients"]
        ),
        efficientvim_intercepts=np.asarray(readouts["efficientvim"]["intercepts"]),
        efficientvim_scaler_means=np.asarray(readouts["efficientvim"]["scaler_means"]),
        efficientvim_scaler_scales=np.asarray(readouts["efficientvim"]["scaler_scales"]),
    )
    summary = {
        "schema_version": 1,
        "protocol_id": PROTOCOL_ID,
        "mode": "formal_train_only_frozen_representation_screen",
        "source_hashes": start_source_hashes,
        "git": {"start": start_git, "end": end_git},
        "dependencies": start_dependencies,
        "device": start_device,
        "weights": accepted["payload"]["weights"],
        "preprocessing": accepted["payload"]["preprocessing"],
        "accepted_preflight": {
            "artifact": accepted["artifact"],
            "sha256": accepted["sha256"],
        },
        "dataset": {
            "data_yaml": str(ledger["data_yaml"]),
            "data_yaml_sha256": EXPECTED_DATA_SHA256,
            "train_rows": EXPECTED_TRAIN_SAMPLES,
            "class_counts": list(EXPECTED_CLASS_COUNTS),
            "train_split_used": True,
            "validation_split_used": False,
            "test_split_used": False,
            "train_content": {
                "start": ledger["train_content"],
                "end": end_train_content,
            },
        },
        "assignment": {
            key: ledger["assignment"][key]
            for key in (
                "csv_sha256",
                "assignment_int64_sha256",
                "path_fold_sha256",
                "group_vector_int64_sha256",
                "fold_rows",
            )
        },
        "descriptor_hashes": descriptor_hashes,
        "paths_sha256": paths_sha256,
        "oof_sha256": oof_sha,
        "readout": {
            "C": LOGISTIC_C,
            "class_weight": "balanced",
            "solver": "lbfgs",
            "tol": LOGISTIC_TOL,
            "max_iter": LOGISTIC_MAX_ITER,
            "fold_records": {
                name: result["fold_records"] for name, result in readouts.items()
            },
        },
        "metrics": summaries,
        "bootstrap": bootstrap,
        "gate": gate,
        "integrity_complete": integrity_complete,
        "wall_seconds": time.perf_counter() - started,
    }
    _atomic_json(output_dir / "summary.json", summary)
    return summary


def main(argv: Sequence[str] | None = None) -> int:
    args = _parse_args(argv)
    if args.preflight_only:
        if args.preflight_artifact is not None or args.preflight_sha256:
            raise ValueError("Preflight mode cannot consume a prior preflight")
        output_dir = args.output_dir.expanduser().resolve()
        if output_dir.exists():
            raise RuntimeError(f"refuse to overwrite B13 preflight directory: {output_dir}")
        payload = build_preflight(args)
        output_dir.mkdir(parents=True, exist_ok=False)
        _atomic_json(output_dir / "preflight.json", payload)
        print(json.dumps({"passed": payload["passed"], "output": str(output_dir)}, indent=2))
        return 0
    summary = _run_formal(args)
    print(json.dumps({"gate": summary["gate"], "output": str(args.output_dir)}, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
