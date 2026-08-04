from __future__ import annotations

import argparse
import hashlib
import json
import os
import subprocess
import sys
import tempfile
import time
import urllib.request
from collections.abc import Mapping, Sequence
from pathlib import Path, PurePosixPath
from typing import Any

os.environ.setdefault("CUBLAS_WORKSPACE_CONFIG", ":4096:8")
os.environ.setdefault("HF_HUB_OFFLINE", "1")

import numpy as np  # noqa: E402
import torch  # noqa: E402
from torch import Tensor, nn  # noqa: E402
from torch.utils.data import DataLoader, Dataset  # noqa: E402
from torchvision.transforms import InterpolationMode  # noqa: E402
from torchvision.transforms import v2  # noqa: E402

from trkh.models.iformer_s_official import (  # noqa: E402
    DESCRIPTOR_DIM,
    OFFICIAL_CHECKPOINT_ASSET_ID,
    OFFICIAL_CHECKPOINT_BYTES,
    OFFICIAL_CHECKPOINT_SHA256,
    OFFICIAL_CHECKPOINT_URL,
    OFFICIAL_LICENSE_SHA256,
    OFFICIAL_REVISION,
    OFFICIAL_SOURCE_SHA256,
    OFFICIAL_TAG,
    PARAMETERS_5,
    build_official_iformer_s_5class,
    load_official_checkpoint,
    load_official_iformer_s_1000,
    validate_official_source,
)
from trkh.tools.audit_dinov3_depth_trajectory_b12 import (  # noqa: E402
    _atomic_json,
    _atomic_npz,
)
from trkh.tools.audit_efficientvim_m1_frozen_transfer_b13 import (  # noqa: E402
    BATCH_SIZE,
    BOOTSTRAP_REPLICATES,
    CLASSES,
    DINO_5_CLASS_PARAMS,
    DINO_IMAGE_SIZE,
    DINO_WEIGHT_SHA256,
    FOLDS,
    ORT_ITERATIONS,
    ORT_THREADS,
    ORT_TRIALS,
    ORT_WARMUPS,
    SEED,
    WORKERS,
    _TrainLedgerDataset,
    _atomic_npy,
    _build_dino,
    _dependency_contract,
    _device_contract,
    _export_onnx,
    _git_contract,
    _ort_session,
    _resolve_device,
    _resolve_dino_weight,
    _sha256,
    _timed_ort,
    _train_content_contract,
    assess_gate,
    classification_summary,
    fit_oof_readout,
    paired_component_bootstrap,
)
from trkh.tools.precheck_dinov3_xcnorm_a0_sourcefold import (  # noqa: E402
    EXPECTED_CLASSES,
)


PROTOCOL_ID = "TRKH_PRETRAINED_CLASSF_B14_IFORMER_S_FROZEN_TRANSFER_20260804"
EXPECTED_BRANCH = "research/pretrained-classf-b1"
EXPECTED_DATA_YAML_SHA256 = (
    "312eb376e89023f42e9df24fea8723b64a6b885c15c24f85d8e319d1ff4f1fa8"
)
EXPECTED_TRAIN_CONTENT_SHA256 = (
    "e9319e6fbddd03382050fadc42bcf156195c590926700b3a7287adc38d4373c8"
)
EXPECTED_TRAIN_ROWS = 8_278
EXPECTED_CLASS_COUNTS = (1_987, 497, 1_326, 2_080, 2_388)
EXPECTED_ASSIGNMENT_SHA256 = (
    "afde4fec6b344b867d31e0e01b89f16c5f4be7fd551fdc1fdfd0ea34219d725f"
)
EXPECTED_FOLD_VECTOR_SHA256 = (
    "fca50be670fac7da20dd23cb382f9096c3c071d1a1e0dbaa1fd123e65c8aeb4b"
)
EXPECTED_GROUP_VECTOR_SHA256 = (
    "1243b61c91497459fa12036d1b9f63770ba4f1f11bb132ca17fb3b6013f1ade2"
)
EXPECTED_BOOTSTRAP_DRAW_SHA256 = (
    "d8967cbc13b17b78dd225841bd82a1a3074a7f57c3728fc6dd600aba646ec469"
)
B13_PROTOCOL_ID = "TRKH_PRETRAINED_CLASSF_B13_EFFICIENTVIM_M1_FROZEN_TRANSFER_20260803"
B13_HEAD = "f229649fed92b56d3ec43f13bd1c23919d956abf"
B13_RUNNER_SHA256 = "641791100aa626379637a2d939f7c36c7971779ab4b723ce40898d2a457fa7cb"
B13_SUMMARY_SHA256 = "b017d34955f1c9126a611cf064cb919b6696ed9a19bf0f2b676838ad97955a75"
B13_OOF_SHA256 = "82bcef6d293223b819f559122bffee822b7f2a2c80a200113dd6c0b7843def39"
B13_PATHS_SHA256 = "a472608c3d0b1ce1b941a1fa4bfa72b9f3032140d7f4c2792cb9bdd357b1c0fb"
EXPECTED_DINO_METRICS = {
    "accuracy": 0.8581783039381493,
    "macro_f1": 0.8020758755136912,
    "class1_precision": 0.4675324675324675,
    "class1_recall": 0.579476861167002,
    "class1_f1": 0.5175202156334232,
    "restricted_fp_rate": 0.05560427995088581,
    "mean_pair_auroc": 0.9461030773728475,
}
OFFICIAL_CONFIG_SHA256 = "754ea8f96b0d480c7ef435cb1ba056b927f1850fa63c975f7792a1db581b37d7"
OFFICIAL_DATASETS_SHA256 = "a046faac72944294051fe6fd202e02939b25b67cb86ee5751f7ce9a70524abc4"
OFFICIAL_REMOTE = "https://github.com/ChuanyangZheng/iFormer.git"
RELEASE_API_URL = "https://api.github.com/repos/ChuanyangZheng/iFormer/releases/tags/v0.9"
EXPECTED_RUNTIME = {
    "python": "3.9.11",
    "numpy": "1.26.4",
    "onnx": "1.19.1",
    "scikit_learn": "1.6.1",
    "timm": "1.0.27",
    "torch": "2.6.0+cu124",
    "torchvision": "0.21.0+cu124",
}
WALL_BUDGET_SECONDS = 15 * 60
VRAM_BUDGET_BYTES = 6 * 1024**3
ARTIFACT_BUDGET_BYTES = 100 * 1024**2


def _repository_root() -> Path:
    return Path(__file__).resolve().parents[2]


def _defaults() -> dict[str, Path]:
    root = _repository_root()
    return {
        "data_yaml": Path(r"D:\DataAI\AIEx\newdataset\class_f\data.yaml"),
        "data_root": Path(r"D:\DataAI\AIEx\newdataset\class_f"),
        "b13_root": root
        / "runs"
        / "pretrained_efficientvim_classf_b13_frozen_transfer_f229649_r1",
        "official_source_root": Path(
            r"D:\DataAI\external_sources\official\iformer_iclr2025"
        ),
        "iformer_checkpoint": root
        / "runs"
        / "pretrained_assets"
        / "iformer_s"
        / "iFormer_s.pth",
    }


def _parse_args(argv: Sequence[str] | None = None) -> argparse.Namespace:
    defaults = _defaults()
    parser = argparse.ArgumentParser(
        description="Locked B14 TRAIN-only iFormer-S frozen representation screen"
    )
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--data-yaml", type=Path, default=defaults["data_yaml"])
    parser.add_argument("--data-root", type=Path, default=defaults["data_root"])
    parser.add_argument("--b13-root", type=Path, default=defaults["b13_root"])
    parser.add_argument(
        "--official-source-root", type=Path, default=defaults["official_source_root"]
    )
    parser.add_argument(
        "--iformer-checkpoint", type=Path, default=defaults["iformer_checkpoint"]
    )
    parser.add_argument("--dino-weight", type=Path, default=None)
    parser.add_argument("--preflight-only", action="store_true")
    parser.add_argument("--preflight-artifact", type=Path, default=None)
    parser.add_argument("--preflight-sha256", type=str, default="")
    parser.add_argument("--batch-size", type=int, default=BATCH_SIZE)
    parser.add_argument("--workers", type=int, default=WORKERS)
    parser.add_argument("--device", type=str, default="cuda")
    parser.add_argument("--torch-threads", type=int, default=4)
    return parser.parse_args(argv)


def _assert_locked_locations(args: argparse.Namespace, *, include_data: bool) -> None:
    defaults = _defaults()
    names = ["official_source_root", "iformer_checkpoint"]
    if include_data:
        names.extend(("data_yaml", "data_root", "b13_root"))
    mismatches = {}
    for name in names:
        observed = Path(getattr(args, name)).expanduser().resolve()
        expected = defaults[name].expanduser().resolve()
        if observed != expected:
            mismatches[name] = {"observed": str(observed), "expected": str(expected)}
    if mismatches:
        raise ValueError(f"B14 locked artifact locations changed: {mismatches}")


def _source_paths() -> dict[str, Path]:
    root = _repository_root()
    return {
        "runner": Path(__file__).resolve(),
        "runner_test": root / "tests" / "test_audit_iformer_s_frozen_transfer_b14.py",
        "protocol": root
        / "docs"
        / "TRKH_PRETRAINED_CLASSF_B14_IFORMER_S_FROZEN_TRANSFER_PROTOCOL_20260804.md",
        "adapter": root / "trkh" / "models" / "iformer_s_official.py",
        "adapter_test": root / "tests" / "test_iformer_s_official.py",
        "b13_runner": root
        / "trkh"
        / "tools"
        / "audit_efficientvim_m1_frozen_transfer_b13.py",
    }


def _source_hashes() -> dict[str, str]:
    root = _repository_root()
    paths = _source_paths()
    for name, path in paths.items():
        if not path.is_file():
            raise FileNotFoundError(f"B14 bound source is missing: {name}={path}")
        result = subprocess.run(
            ["git", "ls-files", "--error-unmatch", str(path.relative_to(root))],
            cwd=root,
            capture_output=True,
            text=True,
        )
        if result.returncode != 0:
            raise RuntimeError(f"B14 bound source is not tracked: {name}={path}")
    hashes = {name: _sha256(path) for name, path in paths.items()}
    if hashes["b13_runner"] != B13_RUNNER_SHA256:
        raise ValueError("Locked B13 helper source changed")
    return hashes


def _runtime_contract() -> dict[str, object]:
    observed = _dependency_contract()
    mismatches = {
        key: (observed.get(key), expected)
        for key, expected in EXPECTED_RUNTIME.items()
        if observed.get(key) != expected
    }
    ort = observed.get("onnxruntime", {})
    if not isinstance(ort, Mapping) or dict(ort) != {
        "distribution": "onnxruntime-gpu",
        "version": "1.19.2",
    }:
        mismatches["onnxruntime"] = (ort, "onnxruntime-gpu==1.19.2")
    if mismatches:
        raise RuntimeError(f"B14 runtime drifted: {mismatches}")
    return observed


def _git_output(root: Path, *arguments: str) -> str:
    return subprocess.run(
        ["git", *arguments],
        cwd=root,
        check=True,
        capture_output=True,
        text=True,
    ).stdout.strip()


def _official_source_contract(root: Path) -> dict[str, object]:
    resolved = root.expanduser().resolve(strict=True)
    adapter = validate_official_source(resolved)
    observed = {
        "head": _git_output(resolved, "rev-parse", "HEAD"),
        "tag": _git_output(resolved, "describe", "--tags", "--exact-match", "HEAD"),
        "tag_commit": _git_output(resolved, "rev-parse", f"{OFFICIAL_TAG}^{{}}"),
        "remote": _git_output(resolved, "remote", "get-url", "origin"),
        "branch": _git_output(resolved, "branch", "--show-current"),
        "status": _git_output(resolved, "status", "--porcelain", "--untracked-files=all"),
        "config_sha256": _sha256(resolved / "configs" / "iFormer_s.yaml"),
        "datasets_sha256": _sha256(resolved / "datasets.py"),
        "license_sha256": _sha256(resolved / "LICENSE"),
    }
    checks = {
        "head_exact": observed["head"] == OFFICIAL_REVISION,
        "tag_exact": observed["tag"] == OFFICIAL_TAG,
        "tag_commit_exact": observed["tag_commit"] == OFFICIAL_REVISION,
        "remote_exact": str(observed["remote"]).rstrip("/")
        == OFFICIAL_REMOTE.rstrip("/"),
        "detached_head": observed["branch"] == "",
        "source_clean": observed["status"] == "",
        "config_exact": observed["config_sha256"] == OFFICIAL_CONFIG_SHA256,
        "datasets_exact": observed["datasets_sha256"] == OFFICIAL_DATASETS_SHA256,
        "license_exact": observed["license_sha256"] == OFFICIAL_LICENSE_SHA256,
        "model_source_exact": adapter["source_sha256"] == OFFICIAL_SOURCE_SHA256,
    }
    if not all(checks.values()):
        raise RuntimeError(f"Official iFormer source contract failed: {checks}")
    return {"root": str(resolved), **observed, "adapter": adapter, "checks": checks}


def _select_release_asset(release: Mapping[str, object]) -> dict[str, object]:
    if release.get("tag_name") != OFFICIAL_TAG:
        raise ValueError("Official release API returned the wrong tag")
    assets = release.get("assets")
    if not isinstance(assets, list):
        raise TypeError("Official release API has no asset list")
    selected = [
        asset
        for asset in assets
        if isinstance(asset, Mapping)
        and int(asset.get("id", -1)) == OFFICIAL_CHECKPOINT_ASSET_ID
        and asset.get("name") == "iFormer_s.pth"
    ]
    if len(selected) != 1:
        raise ValueError("Official iFormer-S release asset is not unique")
    asset = dict(selected[0])
    if (
        int(asset.get("size", -1)) != OFFICIAL_CHECKPOINT_BYTES
        or asset.get("browser_download_url") != OFFICIAL_CHECKPOINT_URL
    ):
        raise ValueError("Official iFormer-S release asset metadata changed")
    return asset


def _url_json(url: str) -> Mapping[str, object]:
    request = urllib.request.Request(url, headers={"User-Agent": "TRKH-B14-audit"})
    with urllib.request.urlopen(request, timeout=60) as response:  # noqa: S310
        payload = json.loads(response.read().decode("utf-8"))
    if not isinstance(payload, Mapping):
        raise TypeError("GitHub release API response is not a mapping")
    return payload


def _stream_url_contract(url: str) -> dict[str, object]:
    request = urllib.request.Request(url, headers={"User-Agent": "TRKH-B14-audit"})
    digest = hashlib.sha256()
    total = 0
    with urllib.request.urlopen(request, timeout=120) as response:  # noqa: S310
        while True:
            chunk = response.read(1024 * 1024)
            if not chunk:
                break
            digest.update(chunk)
            total += len(chunk)
    observed = digest.hexdigest()
    if total != OFFICIAL_CHECKPOINT_BYTES or observed != OFFICIAL_CHECKPOINT_SHA256:
        raise ValueError("Fresh official iFormer-S download does not match the lock")
    return {"url": url, "bytes": total, "sha256": observed, "in_memory": True}


def _release_contract() -> dict[str, object]:
    release = _url_json(RELEASE_API_URL)
    asset = _select_release_asset(release)
    download = _stream_url_contract(OFFICIAL_CHECKPOINT_URL)
    return {
        "api_url": RELEASE_API_URL,
        "release_id": int(release.get("id", -1)),
        "tag": release.get("tag_name"),
        "published_at": release.get("published_at"),
        "asset": {
            "id": int(asset["id"]),
            "name": asset["name"],
            "bytes": int(asset["size"]),
            "url": asset["browser_download_url"],
        },
        "fresh_download": download,
    }


def _run_focused_tests(source_root: Path, checkpoint: Path) -> dict[str, object]:
    root = _repository_root()
    command = [
        sys.executable,
        "-m",
        "pytest",
        "-q",
        "-rs",
        "tests/test_iformer_s_official.py",
        "tests/test_audit_iformer_s_frozen_transfer_b14.py",
    ]
    environment = os.environ.copy()
    environment["TRKH_IFORMER_OFFICIAL_ROOT"] = str(source_root)
    environment["TRKH_IFORMER_CHECKPOINT"] = str(checkpoint)
    result = subprocess.run(
        command,
        cwd=root,
        env=environment,
        capture_output=True,
        text=True,
    )
    combined = (result.stdout + "\n" + result.stderr).strip()
    if result.returncode != 0 or "skipped" in combined.casefold():
        raise RuntimeError(f"B14 focused tests failed or skipped:\n{combined}")
    return {"command": command, "returncode": result.returncode, "output": combined}


def _mobile_preflight(
    *, source_root: Path, checkpoint: Path, dino_weight: Path
) -> dict[str, object]:
    released = load_official_iformer_s_1000(source_root, checkpoint).cpu().eval()
    transfer = build_official_iformer_s_5class(source_root, checkpoint).cpu().eval()
    dino = _build_dino(dino_weight, num_classes=CLASSES).cpu().eval()
    if sum(p.numel() for p in transfer.parameters()) != PARAMETERS_5:
        raise ValueError("iFormer-S 5-class parameter count drifted")
    if sum(p.numel() for p in dino.parameters()) != DINO_5_CLASS_PARAMS:
        raise ValueError("DINO 5-class parameter count drifted")

    torch.manual_seed(SEED + 41)
    released_input = torch.randn(1, 3, 224, 224)
    inputs = {
        "iformer": torch.randn(1, 3, 224, 224),
        "dino": torch.randn(1, 3, DINO_IMAGE_SIZE, DINO_IMAGE_SIZE),
    }
    with torch.inference_mode():
        expected = released.model(released_input)
        reconstructed, descriptor = released.forward_with_descriptor(released_input)
        torch_outputs = {
            "iformer": transfer.model(inputs["iformer"]).numpy(),
            "dino": dino(inputs["dino"]).numpy(),
        }
    reconstruction_exact = torch.equal(expected, reconstructed)
    descriptor_exact = tuple(descriptor.shape) == (1, DESCRIPTOR_DIM)
    if not all(np.isfinite(value).all() for value in torch_outputs.values()):
        raise FloatingPointError("Synthetic B14 logits are non-finite")

    models = {"iformer": transfer.model, "dino": dino}
    with tempfile.TemporaryDirectory(prefix="trkh_b14_onnx_") as temporary:
        temporary_root = Path(temporary)
        paths = {name: temporary_root / f"{name}.onnx" for name in models}
        exports = {
            name: _export_onnx(models[name], inputs[name], paths[name])
            for name in models
        }
        sessions = {name: _ort_session(paths[name]) for name in models}
        arrays = {name: inputs[name].numpy() for name in models}
        parity: dict[str, object] = {}
        for name in models:
            output = sessions[name].run(None, {"images": arrays[name]})[0]
            parity[name] = {
                "max_abs": float(np.max(np.abs(output - torch_outputs[name]))),
                "argmax_equal": bool(
                    np.array_equal(
                        output.argmax(axis=1), torch_outputs[name].argmax(axis=1)
                    )
                ),
            }
        for _ in range(ORT_WARMUPS):
            for name in ("dino", "iformer"):
                sessions[name].run(None, {"images": arrays[name]})
        timings = {name: [] for name in models}
        trial_means = {name: [] for name in models}
        for trial in range(ORT_TRIALS):
            order = ("iformer", "dino") if trial % 2 == 0 else ("dino", "iformer")
            for name in order:
                samples = _timed_ort(sessions[name], arrays[name], ORT_ITERATIONS)
                timings[name].extend(samples)
                trial_means[name].append(float(np.mean(samples)))
        latency = {
            name: {
                "median_ms": float(np.median(samples)),
                "p95_ms": float(np.quantile(samples, 0.95)),
                "trial_mean_ms": trial_means[name],
            }
            for name, samples in timings.items()
        }

    ratios = {
        "parameters": PARAMETERS_5 / DINO_5_CLASS_PARAMS,
        "median": latency["iformer"]["median_ms"] / latency["dino"]["median_ms"],
        "p95": latency["iformer"]["p95_ms"] / latency["dino"]["p95_ms"],
    }
    checks = {
        "iformer_5class_parameters_exact": True,
        "parameter_ratio_lte_0p30": ratios["parameters"] <= 0.30,
        "pooled_descriptor_shape_exact": descriptor_exact,
        "official_logits_reconstructed_exact": reconstruction_exact,
        "onnx_standard_domains_only": all(
            all(domain in {"", "ai.onnx"} for domain in exports[name]["operator_domains"])
            for name in exports
        ),
        "onnx_parity_max_abs_lte_1e_5": all(
            float(parity[name]["max_abs"]) <= 1e-5 for name in parity
        ),
        "onnx_argmax_exact": all(bool(parity[name]["argmax_equal"]) for name in parity),
        "median_latency_ratio_lte_0p25": ratios["median"] <= 0.25,
        "p95_latency_ratio_lte_0p30": ratios["p95"] <= 0.30,
    }
    return {
        "settings": {
            "runtime": "onnxruntime_cpu",
            "threads": ORT_THREADS,
            "batch": 1,
            "warmups": ORT_WARMUPS,
            "trials": ORT_TRIALS,
            "iterations_per_trial": ORT_ITERATIONS,
            "iformer_input": [1, 3, 224, 224],
            "dino_input": [1, 3, DINO_IMAGE_SIZE, DINO_IMAGE_SIZE],
        },
        "parameters": {
            "iformer_5class": PARAMETERS_5,
            "dino_5class": DINO_5_CLASS_PARAMS,
        },
        "exports": exports,
        "parity": parity,
        "latency": latency,
        "ratios": ratios,
        "checks": checks,
        "passed": all(checks.values()),
    }


def _preflight_checks(
    *, project_git: Mapping[str, object], mobile_passed: bool
) -> dict[str, bool]:
    return {
        "canonical_branch_clean": project_git.get("branch") == EXPECTED_BRANCH
        and project_git.get("tracked_worktree_clean") is True,
        "protocol_and_sources_tracked": True,
        "runtime_exact": True,
        "official_source_exact_clean": True,
        "official_release_byte_identity": True,
        "checkpoint_safe_strict_load": True,
        "focused_tests_passed_without_skip": True,
        "mobile_precondition": bool(mobile_passed),
        "train_descriptor_not_read": True,
        "validation_not_constructed": True,
        "test_not_constructed": True,
    }


def build_preflight(args: argparse.Namespace) -> dict[str, object]:
    _assert_locked_locations(args, include_data=False)
    output_dir, _unused_partial = _validate_output_root(
        args.output_dir, _defaults()["data_root"]
    )
    project_git = _git_contract()
    if project_git["branch"] != EXPECTED_BRANCH or not project_git["tracked_worktree_clean"]:
        raise RuntimeError(f"B14 preflight requires clean canonical branch: {project_git}")
    source_hashes = _source_hashes()
    runtime = _runtime_contract()
    official = _official_source_contract(args.official_source_root)
    checkpoint = args.iformer_checkpoint.expanduser().resolve(strict=True)
    checkpoint_contract = load_official_checkpoint(checkpoint)
    checkpoint_contract = {
        key: value for key, value in checkpoint_contract.items() if key != "state"
    }
    release = _release_contract()
    focused_tests = _run_focused_tests(official_root := Path(official["root"]), checkpoint)
    dino_weight = _resolve_dino_weight(args.dino_weight)
    mobile = _mobile_preflight(
        source_root=official_root, checkpoint=checkpoint, dino_weight=dino_weight
    )
    checks = _preflight_checks(project_git=project_git, mobile_passed=bool(mobile["passed"]))
    payload = {
        "schema_version": 1,
        "protocol_id": PROTOCOL_ID,
        "mode": "synthetic_preflight_no_dataset",
        "created_at_unix": time.time(),
        "source_hashes": source_hashes,
        "git": project_git,
        "runtime": runtime,
        "device": _device_contract(args.device),
        "official_source": official,
        "release": release,
        "weights": {
            "iformer": checkpoint_contract,
            "dino": {"path": str(dino_weight), "sha256": DINO_WEIGHT_SHA256},
        },
        "focused_tests": focused_tests,
        "mobile": mobile,
        "permissions": {
            "train_descriptor_read": False,
            "validation_not_constructed": True,
            "test_not_constructed": True,
            "validation_permission": False,
            "test_permission": False,
            "full_train_permission": False,
        },
        "checks": checks,
        "passed": all(checks.values()),
    }
    output_dir.mkdir(parents=True, exist_ok=False)
    _atomic_json(output_dir / "preflight.json", payload)
    return payload


def _int64_sha(array: np.ndarray) -> str:
    values = np.asarray(array, dtype="<i8")
    return hashlib.sha256(values.tobytes()).hexdigest()


def _validate_b13_summary_payload(summary: Mapping[str, object]) -> None:
    git = summary.get("git")
    dataset = summary.get("dataset")
    assignment = summary.get("assignment")
    if not isinstance(git, Mapping) or not isinstance(dataset, Mapping) or not isinstance(
        assignment, Mapping
    ):
        raise ValueError("Locked B13 summary contract is incomplete")
    start = git.get("start")
    end = git.get("end")
    checks = {
        "protocol": summary.get("protocol_id") == B13_PROTOCOL_ID,
        "integrity": summary.get("integrity_complete") is True,
        "start_head": isinstance(start, Mapping) and start.get("head") == B13_HEAD,
        "end_head": isinstance(end, Mapping) and end.get("head") == B13_HEAD,
        "train_used": dataset.get("train_split_used") is True,
        "validation_unused": dataset.get("validation_split_used") is False,
        "test_unused": dataset.get("test_split_used") is False,
        "rows": int(dataset.get("train_rows", -1)) == EXPECTED_TRAIN_ROWS,
        "assignment_csv": assignment.get("csv_sha256") == EXPECTED_ASSIGNMENT_SHA256,
        "fold_vector": assignment.get("assignment_int64_sha256")
        == EXPECTED_FOLD_VECTOR_SHA256,
        "group_vector": assignment.get("group_vector_int64_sha256")
        == EXPECTED_GROUP_VECTOR_SHA256,
    }
    if not all(checks.values()):
        raise ValueError(f"Locked B13 summary contract failed: {checks}")


def _read_allowed_b13_arrays(archive: Any) -> dict[str, np.ndarray]:
    allowed = ("labels", "folds", "groups", "dino_scores")
    return {name: np.asarray(archive[name]).copy() for name in allowed}


def _resolve_train_paths(
    relative_paths: Sequence[str],
    labels: np.ndarray,
    data_root: Path,
    *,
    require_files: bool = True,
) -> list[Path]:
    root = data_root.expanduser().resolve()
    train_root = (root / "train").resolve()
    labels = np.asarray(labels, dtype=np.int64)
    if len(relative_paths) != labels.size:
        raise ValueError("B14 TRAIN paths and labels are not aligned")
    absolute_paths: list[Path] = []
    for index, (raw, label) in enumerate(zip(relative_paths, labels)):
        if not isinstance(raw, str):
            raise TypeError("B14 TRAIN path is not text")
        normalized = raw.replace("\\", "/")
        pure = PurePosixPath(normalized)
        if (
            pure.is_absolute()
            or not pure.parts
            or pure.parts[0] != "train"
            or any(part in {"", ".", ".."} for part in pure.parts)
        ):
            raise ValueError(f"B14 rejected non-TRAIN path at row {index}: {raw}")
        if int(label) not in range(CLASSES) or len(pure.parts) < 3:
            raise ValueError(f"B14 invalid label/path geometry at row {index}")
        if pure.parts[1] != EXPECTED_CLASSES[int(label)]:
            raise ValueError(f"B14 class directory/label mismatch at row {index}")
        absolute = (root / Path(*pure.parts)).resolve()
        try:
            absolute.relative_to(train_root)
        except ValueError as error:
            raise ValueError(f"B14 TRAIN path escaped train root: {raw}") from error
        if require_files and not absolute.is_file():
            raise FileNotFoundError(f"B14 TRAIN image is missing: {absolute}")
        absolute_paths.append(absolute)
    return absolute_paths


def _validate_dino_metrics(labels: np.ndarray, folds: np.ndarray, scores: np.ndarray) -> dict[str, object]:
    observed = classification_summary(labels, scores, folds)
    for key, expected in EXPECTED_DINO_METRICS.items():
        if float(observed[key]) != expected:
            raise ValueError(f"Locked DINO metric drifted: {key}={observed[key]} != {expected}")
    return observed


def _load_locked_comparator(root: Path, data_root: Path) -> dict[str, object]:
    resolved = root.expanduser().resolve(strict=True)
    summary_path = resolved / "summary.json"
    oof_path = resolved / "train_oof_readouts.npz"
    paths_path = resolved / "train_paths.json"
    expected_hashes = {
        summary_path: B13_SUMMARY_SHA256,
        oof_path: B13_OOF_SHA256,
        paths_path: B13_PATHS_SHA256,
    }
    for path, expected in expected_hashes.items():
        if not path.is_file() or _sha256(path) != expected:
            raise ValueError(f"Locked B13 artifact changed: {path}")
    summary = json.loads(summary_path.read_text(encoding="utf-8"))
    if not isinstance(summary, Mapping):
        raise TypeError("Locked B13 summary is not a mapping")
    _validate_b13_summary_payload(summary)
    with np.load(oof_path, allow_pickle=False) as archive:
        arrays = _read_allowed_b13_arrays(archive)
    labels = np.asarray(arrays["labels"], dtype=np.int64)
    folds = np.asarray(arrays["folds"], dtype=np.int64)
    groups = np.asarray(arrays["groups"], dtype=np.int64)
    dino_scores = np.asarray(arrays["dino_scores"], dtype=np.float64)
    if (
        labels.shape != (EXPECTED_TRAIN_ROWS,)
        or folds.shape != labels.shape
        or groups.shape != labels.shape
        or dino_scores.shape != (EXPECTED_TRAIN_ROWS, CLASSES)
        or not np.isfinite(dino_scores).all()
        or tuple(np.bincount(labels, minlength=CLASSES).tolist())
        != EXPECTED_CLASS_COUNTS
        or set(np.unique(folds).tolist()) != set(range(FOLDS))
        or _int64_sha(folds) != EXPECTED_FOLD_VECTOR_SHA256
        or _int64_sha(groups) != EXPECTED_GROUP_VECTOR_SHA256
    ):
        raise ValueError("Locked B13 comparator arrays changed")
    relative_paths = json.loads(paths_path.read_text(encoding="utf-8"))
    if not isinstance(relative_paths, list) or len(relative_paths) != EXPECTED_TRAIN_ROWS:
        raise ValueError("Locked B13 path ledger changed")
    absolute_paths = _resolve_train_paths(relative_paths, labels, data_root)
    dino_metrics = _validate_dino_metrics(labels, folds, dino_scores)
    return {
        "root": str(resolved),
        "summary": summary,
        "labels": labels,
        "folds": folds,
        "groups": groups,
        "dino_scores": dino_scores,
        "relative_paths": relative_paths,
        "absolute_paths": absolute_paths,
        "dino_metrics": dino_metrics,
        "hashes": {
            "summary": B13_SUMMARY_SHA256,
            "oof": B13_OOF_SHA256,
            "paths": B13_PATHS_SHA256,
            "b13_runner": B13_RUNNER_SHA256,
        },
    }


def _iformer_transform() -> Any:
    return v2.Compose(
        (
            v2.Resize(256, interpolation=InterpolationMode.BICUBIC, antialias=True),
            v2.CenterCrop(224),
            v2.ToImage(),
            v2.ToDtype(torch.float32, scale=True),
            v2.Normalize(
                mean=(0.485, 0.456, 0.406), std=(0.229, 0.224, 0.225)
            ),
        )
    )


def _extract_descriptors(
    *,
    model: nn.Module,
    dataset: Dataset[tuple[Tensor, int, int]],
    device: torch.device,
    batch_size: int,
    workers: int,
) -> np.ndarray:
    loader = DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=workers,
        pin_memory=device.type == "cuda",
        drop_last=False,
    )
    output = np.full((len(dataset), DESCRIPTOR_DIM), np.nan, dtype=np.float32)
    expected_index = 0
    next_report = 10
    model.to(device).eval()
    with torch.inference_mode():
        for batch_index, (images, _labels, indices) in enumerate(loader, start=1):
            index_array = indices.numpy().astype(np.int64, copy=False)
            expected = np.arange(expected_index, expected_index + index_array.size)
            if not np.array_equal(index_array, expected):
                raise ValueError("B14 DataLoader order changed")
            images = images.to(device=device, dtype=torch.float32, non_blocking=True)
            features = model(images).float().detach().cpu().numpy()
            if features.shape != (index_array.size, DESCRIPTOR_DIM):
                raise ValueError(f"B14 descriptor shape changed: {features.shape}")
            output[index_array] = features
            expected_index += index_array.size
            percent = int(100 * batch_index / max(len(loader), 1))
            if percent >= next_report:
                print(f"iformer: {percent}% ({expected_index}/{len(dataset)})", flush=True)
                next_report += 10
    model.cpu()
    if expected_index != len(dataset) or not np.isfinite(output).all():
        raise RuntimeError("B14 descriptors are incomplete or non-finite")
    return output


def _gate_for_iformer(
    *,
    dino: Mapping[str, object],
    candidate: Mapping[str, object],
    bootstrap: Mapping[str, object],
    integrity_complete: bool,
    readouts_converged: bool,
) -> dict[str, object]:
    gate = assess_gate(
        dino=dino,
        candidate=candidate,
        bootstrap=bootstrap,
        integrity_complete=integrity_complete,
        readouts_converged=readouts_converged,
    )
    closed = bool(gate.pop("exact_m1_final_feature_route_closed"))
    gate["exact_iformer_s_pooled320_route_closed"] = closed
    return gate


def _output_bytes(root: Path) -> int:
    return sum(path.stat().st_size for path in root.rglob("*") if path.is_file())


def _validate_output_root(output_dir: Path, data_root: Path) -> tuple[Path, Path]:
    output = output_dir.expanduser().resolve()
    runs_root = (_repository_root() / "runs").resolve()
    immutable = data_root.expanduser().resolve()
    try:
        output.relative_to(runs_root)
    except ValueError as error:
        raise ValueError("B14 output must be inside the pretrained runs root") from error
    try:
        output.relative_to(immutable)
    except ValueError:
        pass
    else:
        raise ValueError("B14 output must stay outside immutable class_f")
    partial = output.with_name(output.name + ".partial")
    if output.exists() or partial.exists():
        raise RuntimeError(f"Refuse to overwrite B14 output/partial: {output}")
    return output, partial


def _validate_preflight(
    artifact: Path,
    expected_sha256: str,
    *,
    args: argparse.Namespace,
) -> dict[str, object]:
    path = artifact.expanduser().resolve(strict=True)
    observed_sha = _sha256(path)
    if observed_sha != expected_sha256.casefold():
        raise ValueError("Accepted B14 preflight SHA-256 mismatch")
    payload = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(payload, Mapping) or payload.get("protocol_id") != PROTOCOL_ID:
        raise ValueError("Accepted B14 preflight protocol mismatch")
    current_git = _git_contract()
    current_sources = _source_hashes()
    current_runtime = _runtime_contract()
    current_official = _official_source_contract(args.official_source_root)
    weights = payload.get("weights")
    checks = {
        "passed": payload.get("passed") is True,
        "same_git": payload.get("git") == current_git,
        "same_sources": payload.get("source_hashes") == current_sources,
        "same_runtime": payload.get("runtime") == current_runtime,
        "same_official_source": payload.get("official_source") == current_official,
        "same_device": payload.get("device") == _device_contract(args.device),
        "weights_mapping": isinstance(weights, Mapping),
        "iformer_weight": isinstance(weights, Mapping)
        and isinstance(weights.get("iformer"), Mapping)
        and weights["iformer"].get("sha256") == OFFICIAL_CHECKPOINT_SHA256,
        "dino_weight": isinstance(weights, Mapping)
        and isinstance(weights.get("dino"), Mapping)
        and weights["dino"].get("sha256") == DINO_WEIGHT_SHA256,
    }
    if not all(checks.values()):
        raise RuntimeError(f"Accepted B14 preflight no longer applies: {checks}")
    return {"artifact": str(path), "sha256": observed_sha, "payload": dict(payload)}


def _run_formal(args: argparse.Namespace) -> dict[str, object]:
    started = time.perf_counter()
    _assert_locked_locations(args, include_data=True)
    if args.preflight_artifact is None or not args.preflight_sha256:
        raise ValueError("Formal B14 requires accepted preflight path and SHA-256")
    if args.batch_size != BATCH_SIZE or args.workers != WORKERS:
        raise ValueError("B14 is locked to batch-size=32 and workers=0")
    data_yaml = args.data_yaml.expanduser().resolve(strict=True)
    data_root = args.data_root.expanduser().resolve(strict=True)
    if _sha256(data_yaml) != EXPECTED_DATA_YAML_SHA256:
        raise ValueError("Canonical class_f data YAML bytes changed")
    output_dir, partial_dir = _validate_output_root(args.output_dir, data_root)
    accepted = _validate_preflight(
        args.preflight_artifact, args.preflight_sha256, args=args
    )
    project_git_start = _git_contract()
    source_hashes_start = _source_hashes()
    runtime_start = _runtime_contract()
    official_start = _official_source_contract(args.official_source_root)
    comparator = _load_locked_comparator(args.b13_root, data_root)
    labels = np.asarray(comparator["labels"], dtype=np.int64)
    folds = np.asarray(comparator["folds"], dtype=np.int64)
    groups = np.asarray(comparator["groups"], dtype=np.int64)
    dino_scores = np.asarray(comparator["dino_scores"], dtype=np.float64)
    relative_paths = list(comparator["relative_paths"])
    absolute_paths = list(comparator["absolute_paths"])
    train_content_start = _train_content_contract(relative_paths, absolute_paths)
    if train_content_start["sha256"] != EXPECTED_TRAIN_CONTENT_SHA256:
        raise ValueError("Canonical TRAIN content bytes changed")

    torch.set_num_threads(int(args.torch_threads))
    torch.manual_seed(SEED)
    np.random.seed(SEED)
    torch.use_deterministic_algorithms(True)
    torch.backends.cudnn.benchmark = False
    device = _resolve_device(args.device)
    if device.type == "cuda":
        torch.cuda.reset_peak_memory_stats(device)
    checkpoint = args.iformer_checkpoint.expanduser().resolve(strict=True)
    model = load_official_iformer_s_1000(
        args.official_source_root.expanduser().resolve(strict=True), checkpoint
    )
    dataset = _TrainLedgerDataset(absolute_paths, labels, _iformer_transform())
    partial_dir.mkdir(parents=True, exist_ok=False)
    _atomic_json(
        partial_dir / "run_owner.json",
        {
            "protocol_id": PROTOCOL_ID,
            "intended_output": str(output_dir),
            "git_head": project_git_start["head"],
        },
    )
    descriptors = _extract_descriptors(
        model=model,
        dataset=dataset,
        device=device,
        batch_size=args.batch_size,
        workers=args.workers,
    )
    descriptor_sha = _atomic_npy(
        partial_dir / "train_iformer_s_pooled320_f32.npy", descriptors
    )
    _atomic_json(partial_dir / "train_paths.json", relative_paths)
    paths_sha = _sha256(partial_dir / "train_paths.json")
    candidate_readout = fit_oof_readout(descriptors, labels, folds)
    candidate_scores = np.asarray(candidate_readout["scores"], dtype=np.float64)
    dino_metrics = dict(comparator["dino_metrics"])
    candidate_metrics = classification_summary(labels, candidate_scores, folds)
    _atomic_json(
        partial_dir / "stage_metrics_constructed.json",
        {
            "metrics_constructed": True,
            "retry_allowed": False,
            "reason": "The preregistered gate is final once candidate metrics exist.",
        },
    )
    bootstrap = paired_component_bootstrap(
        labels=labels,
        folds=folds,
        groups=groups,
        dino_scores=dino_scores,
        candidate_scores=candidate_scores,
        replicates=BOOTSTRAP_REPLICATES,
        seed=SEED,
    )
    if bootstrap["draws_int64_sha256"] != EXPECTED_BOOTSTRAP_DRAW_SHA256:
        raise ValueError("B14 bootstrap draw vector changed")
    train_content_end = _train_content_contract(relative_paths, absolute_paths)
    project_git_end = _git_contract()
    source_hashes_end = _source_hashes()
    runtime_end = _runtime_contract()
    official_end = _official_source_contract(args.official_source_root)
    peak_vram = (
        int(torch.cuda.max_memory_allocated(device)) if device.type == "cuda" else 0
    )
    wall_seconds = time.perf_counter() - started
    integrity_complete = bool(
        np.isfinite(descriptors).all()
        and np.isfinite(candidate_scores).all()
        and train_content_end == train_content_start
        and source_hashes_end == source_hashes_start
        and project_git_end == project_git_start
        and project_git_end["tracked_worktree_clean"] is True
        and runtime_end == runtime_start
        and official_end == official_start
    )
    gate = _gate_for_iformer(
        dino=dino_metrics,
        candidate=candidate_metrics,
        bootstrap=bootstrap,
        integrity_complete=integrity_complete,
        readouts_converged=bool(candidate_readout["converged"]),
    )
    oof_sha = _atomic_npz(
        partial_dir / "train_oof_readouts.npz",
        labels=labels,
        folds=folds,
        groups=groups,
        dino_scores=dino_scores,
        iformer_scores=candidate_scores,
        iformer_coefficients=np.asarray(candidate_readout["coefficients"]),
        iformer_intercepts=np.asarray(candidate_readout["intercepts"]),
        iformer_scaler_means=np.asarray(candidate_readout["scaler_means"]),
        iformer_scaler_scales=np.asarray(candidate_readout["scaler_scales"]),
    )
    artifact_bytes_before_summary = _output_bytes(partial_dir)
    budget = {
        "wall_seconds_lte_900": wall_seconds <= WALL_BUDGET_SECONDS,
        "peak_vram_lte_6gib": peak_vram <= VRAM_BUDGET_BYTES,
        "artifact_bytes_lte_100mib": artifact_bytes_before_summary
        <= ARTIFACT_BUDGET_BYTES,
    }
    execution_valid = integrity_complete and all(budget.values())
    gate["scientific_gate_evaluable"] = execution_valid
    gate["fine_tune_protocol_permission"] = bool(
        execution_valid and gate["signal_gate_passed"]
    )
    if not execution_valid:
        gate["exact_iformer_s_pooled320_route_closed"] = False
        gate["execution_status"] = "INFRASTRUCTURE_INVALID"
    else:
        gate["execution_status"] = "SCIENTIFIC_GATE_EVALUATED"
    summary = {
        "schema_version": 1,
        "protocol_id": PROTOCOL_ID,
        "mode": "formal_train_only_frozen_representation_screen",
        "source_hashes": {"start": source_hashes_start, "end": source_hashes_end},
        "git": {"start": project_git_start, "end": project_git_end},
        "runtime": {"start": runtime_start, "end": runtime_end},
        "official_source": {"start": official_start, "end": official_end},
        "accepted_preflight": {
            "artifact": accepted["artifact"],
            "sha256": accepted["sha256"],
        },
        "weights": accepted["payload"]["weights"],
        "dataset": {
            "data_yaml": str(data_yaml),
            "data_yaml_sha256": EXPECTED_DATA_YAML_SHA256,
            "data_root": str(data_root),
            "train_rows": EXPECTED_TRAIN_ROWS,
            "class_names": list(EXPECTED_CLASSES),
            "class_counts": list(EXPECTED_CLASS_COUNTS),
            "train_split_used": True,
            "validation_split_used": False,
            "test_split_used": False,
            "validation_not_constructed": True,
            "test_not_constructed": True,
            "train_content": {"start": train_content_start, "end": train_content_end},
        },
        "comparator": {
            "root": comparator["root"],
            "hashes": comparator["hashes"],
            "allowed_arrays": ["labels", "folds", "groups", "dino_scores"],
            "efficientvim_scores_read": False,
        },
        "preprocessing": {
            "exif_transpose": True,
            "rgb": True,
            "resize_short_side": 256,
            "interpolation": "bicubic",
            "antialias": True,
            "center_crop": 224,
            "mean": [0.485, 0.456, 0.406],
            "std": [0.229, 0.224, 0.225],
            "descriptor": "raw_pre_classifier_bn_adaptive_gap",
            "dimension": DESCRIPTOR_DIM,
        },
        "descriptor_sha256": descriptor_sha,
        "paths_sha256": paths_sha,
        "oof_sha256": oof_sha,
        "readout": {
            "C": 1.0,
            "class_weight": "balanced",
            "solver": "lbfgs",
            "tol": 1e-8,
            "max_iter": 2000,
            "random_state": SEED,
            "fold_records": candidate_readout["fold_records"],
        },
        "metrics": {"dino": dino_metrics, "iformer": candidate_metrics},
        "bootstrap": bootstrap,
        "gate": gate,
        "integrity_complete": integrity_complete,
        "operational_budget": {
            "limits": {
                "wall_seconds": WALL_BUDGET_SECONDS,
                "peak_vram_bytes": VRAM_BUDGET_BYTES,
                "artifact_bytes": ARTIFACT_BUDGET_BYTES,
            },
            "observed": {
                "wall_seconds": wall_seconds,
                "peak_vram_bytes": peak_vram,
                "artifact_bytes_before_summary": artifact_bytes_before_summary,
            },
            "checks": budget,
        },
        "execution_valid": execution_valid,
        "permissions": {
            "fine_tune_protocol_permission": bool(
                execution_valid and gate["signal_gate_passed"]
            ),
            "validation_permission": False,
            "test_permission": False,
            "full_train_permission": False,
            "xai_claim_permission": False,
            "redistribution_permission": False,
        },
    }
    _atomic_json(partial_dir / "summary.json", summary)
    if _output_bytes(partial_dir) > ARTIFACT_BUDGET_BYTES:
        raise RuntimeError("B14 artifacts exceeded the locked budget after summary")
    partial_dir.replace(output_dir)
    return summary


def _quarantine_partial(args: argparse.Namespace, error: BaseException) -> None:
    partial = args.output_dir.expanduser().resolve().with_name(
        args.output_dir.expanduser().resolve().name + ".partial"
    )
    if not partial.is_dir():
        return
    owner_path = partial / "run_owner.json"
    if not owner_path.is_file():
        return
    try:
        owner = json.loads(owner_path.read_text(encoding="utf-8"))
    except (OSError, ValueError, TypeError):
        return
    if not isinstance(owner, Mapping) or owner.get("protocol_id") != PROTOCOL_ID:
        return
    manifest_path = partial / "failure_manifest.json"
    if manifest_path.exists():
        return
    files = []
    for path in sorted(candidate for candidate in partial.rglob("*") if candidate.is_file()):
        files.append(
            {
                "relative_path": path.relative_to(partial).as_posix(),
                "bytes": int(path.stat().st_size),
                "sha256": _sha256(path),
            }
        )
    metrics_constructed = (partial / "stage_metrics_constructed.json").is_file()
    _atomic_json(
        manifest_path,
        {
            "schema_version": 1,
            "protocol_id": PROTOCOL_ID,
            "status": "QUARANTINED_PARTIAL",
            "created_at_unix": time.time(),
            "error_type": type(error).__name__,
            "error": str(error),
            "metrics_constructed": metrics_constructed,
            "retry_allowed": not metrics_constructed,
            "partial_root": str(partial),
            "files": files,
            "deletion_requires_reviewed_cleanup_manifest": True,
        },
    )


def main(argv: Sequence[str] | None = None) -> int:
    args = _parse_args(argv)
    if args.preflight_only:
        payload = build_preflight(args)
    else:
        try:
            payload = _run_formal(args)
        except BaseException as error:
            _quarantine_partial(args, error)
            raise
    print(json.dumps(payload, indent=2), flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
