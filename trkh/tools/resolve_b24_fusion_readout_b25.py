from __future__ import annotations

import argparse
import json
import subprocess
import time
from collections.abc import Sequence
from pathlib import Path

import numpy as np

from trkh.tools import audit_efficientvim_m1_frozen_transfer_b13 as b13
from trkh.tools.audit_dinov3_convnext_b24 import (
    BOOTSTRAP_REPLICATES,
    PROTOCOL_ID as B24_PROTOCOL_ID,
    assess_gate,
)


PROTOCOL_ID = "TRKH_PRETRAINED_CLASSF_B25_B24_FUSION_READOUT_RESOLUTION_20260805"
MAX_ITER = 10_000
SEED = 20260805
EXPECTED = {
    "b13_dino_features": "0429260ab2f619cf8312848e08af1a13f4c81caa95a93daebda6e6e22240dc38",
    "b24_convnext_features": "a0887a008745fabe748936a05c0b61656c570196f8d774e74aadfcfe955ab426",
    "b24_oof": "65631f3f0ed925940bf8b43c699b35c121345bdb33de2c748e1f1644c75110f3",
    "b24_summary": "9de8cd82ccd0a8fb7be7a3e53bc6533c1f0707efe9866bff35db33666d6c5c7e",
}


def _args(argv: Sequence[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "B25 solver-only resolution of the frozen B24 fusion readout; "
            "no image split or model is constructed."
        )
    )
    parser.add_argument("--b13-dir", type=Path, required=True)
    parser.add_argument("--b24-dir", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--preflight-only", action="store_true")
    parser.add_argument("--preflight-artifact", type=Path, default=None)
    parser.add_argument("--preflight-sha256", type=str, default="")
    return parser.parse_args(argv)


def _root() -> Path:
    return Path(__file__).resolve().parents[2]


def _git() -> dict[str, object]:
    value = b13._git_contract()
    if value["branch"] != "research/pretrained-classf-b1" or not value[
        "tracked_worktree_clean"
    ]:
        raise RuntimeError(f"B25 requires the clean canonical branch: {value}")
    return value


def _sources() -> dict[str, str]:
    root = _root()
    paths = {
        "runner": Path(__file__).resolve(),
        "b24": root / "trkh" / "tools" / "audit_dinov3_convnext_b24.py",
        "b13": root
        / "trkh"
        / "tools"
        / "audit_efficientvim_m1_frozen_transfer_b13.py",
        "decision": root / "docs" / "TRKH_PRETRAINED_CLASSF_REFOCUS_20260802.md",
    }
    for name, path in paths.items():
        relative = path.relative_to(root)
        tracked = subprocess.run(
            ["git", "ls-files", "--error-unmatch", str(relative)],
            cwd=root,
            capture_output=True,
            text=True,
        )
        if not path.is_file() or tracked.returncode != 0:
            raise RuntimeError(f"B25 bound source is missing/untracked: {name}")
    return {name: b13._sha256(path) for name, path in paths.items()}


def _artifacts(args: argparse.Namespace) -> dict[str, Path]:
    b13_dir = args.b13_dir.expanduser().resolve()
    b24_dir = args.b24_dir.expanduser().resolve()
    paths = {
        "b13_dino_features": b13_dir / "train_dino_final_f32.npy",
        "b24_convnext_features": b24_dir / "train_convnext_final_f32.npy",
        "b24_oof": b24_dir / "train_oof_readouts.npz",
        "b24_summary": b24_dir / "summary.json",
    }
    for name, path in paths.items():
        if not path.is_file() or b13._sha256(path) != EXPECTED[name]:
            raise ValueError(f"B25 retained artifact changed: {name}")
    return paths


def _load(args: argparse.Namespace) -> dict[str, object]:
    paths = _artifacts(args)
    dino = np.load(paths["b13_dino_features"], allow_pickle=False)
    convnext = np.load(paths["b24_convnext_features"], allow_pickle=False)
    with np.load(paths["b24_oof"], allow_pickle=False) as payload:
        arrays = {
            name: np.asarray(payload[name])
            for name in ("labels", "folds", "groups", "dino_scores", "fusion_scores")
        }
    with paths["b24_summary"].open("r", encoding="utf-8") as handle:
        summary = json.load(handle)
    checks = {
        "b24_protocol": summary.get("protocol_id") == B24_PROTOCOL_ID,
        "train_only": summary.get("dataset", {}).get("train_split_used") is True,
        "validation_closed": summary.get("dataset", {}).get("validation_split_used")
        is False,
        "test_closed": summary.get("dataset", {}).get("test_split_used") is False,
        "b24_integrity": summary.get("integrity_complete") is True,
        "feature_shapes": dino.shape == (8278, 384)
        and convnext.shape == (8278, 768),
        "ledger_shapes": arrays["labels"].shape == (8278,)
        and arrays["folds"].shape == (8278,)
        and arrays["groups"].shape == (8278,),
        "score_shapes": arrays["dino_scores"].shape == (8278, 5)
        and arrays["fusion_scores"].shape == (8278, 5),
        "finite": bool(
            np.isfinite(dino).all()
            and np.isfinite(convnext).all()
            and np.isfinite(arrays["dino_scores"]).all()
        ),
    }
    if not all(checks.values()):
        raise ValueError(f"B25 retained contract failed: {checks}")
    return {
        "paths": paths,
        "dino": np.asarray(dino, dtype=np.float32),
        "convnext": np.asarray(convnext, dtype=np.float32),
        "arrays": arrays,
        "b24_summary": summary,
        "checks": checks,
    }


def _preflight(args: argparse.Namespace) -> dict[str, object]:
    retained = _load(args)
    git = _git()
    checks = {
        **retained["checks"],
        "git_clean": True,
        "sources_bound": True,
        "same_objective_only_max_iter_changes": b13.LOGISTIC_MAX_ITER == 2_000
        and MAX_ITER == 10_000,
        "validation_not_constructed": True,
        "test_not_constructed": True,
    }
    return {
        "schema_version": 1,
        "protocol_id": PROTOCOL_ID,
        "created_at_unix": time.time(),
        "scope": "solver_only_same_objective_train_oof_resolution",
        "source_hashes": _sources(),
        "git": git,
        "artifacts": {
            name: {"path": str(path), "sha256": EXPECTED[name]}
            for name, path in retained["paths"].items()
        },
        "readout": {
            "C": b13.LOGISTIC_C,
            "class_weight": "balanced",
            "solver": "lbfgs",
            "tol": b13.LOGISTIC_TOL,
            "old_max_iter": b13.LOGISTIC_MAX_ITER,
            "new_max_iter": MAX_ITER,
        },
        "gate": "exact_B24_gate_unchanged",
        "checks": checks,
        "passed": all(checks.values()),
    }


def _accepted(args: argparse.Namespace) -> dict[str, object]:
    if args.preflight_artifact is None or not args.preflight_sha256:
        raise ValueError("Formal B25 requires accepted preflight path and SHA-256")
    path = args.preflight_artifact.expanduser().resolve()
    if b13._sha256(path) != args.preflight_sha256.lower():
        raise ValueError("B25 preflight SHA changed")
    with path.open("r", encoding="utf-8") as handle:
        payload = json.load(handle)
    checks = {
        "passed": payload.get("passed") is True,
        "protocol": payload.get("protocol_id") == PROTOCOL_ID,
        "sources": payload.get("source_hashes") == _sources(),
        "git": payload.get("git") == _git(),
    }
    if not all(checks.values()):
        raise ValueError(f"B25 preflight validation failed: {checks}")
    return {"artifact": str(path), "sha256": args.preflight_sha256.lower()}


def _formal(args: argparse.Namespace) -> dict[str, object]:
    output = args.output_dir.expanduser().resolve()
    if output.exists():
        raise RuntimeError(f"refuse to overwrite B25 output: {output}")
    accepted = _accepted(args)
    retained = _load(args)
    start_git, start_sources = _git(), _sources()
    arrays = retained["arrays"]
    features = np.concatenate((retained["dino"], retained["convnext"]), axis=1)
    previous_max_iter = b13.LOGISTIC_MAX_ITER
    b13.LOGISTIC_MAX_ITER = MAX_ITER
    try:
        readout = b13.fit_oof_readout(features, arrays["labels"], arrays["folds"])
    finally:
        b13.LOGISTIC_MAX_ITER = previous_max_iter
    dino = b13.classification_summary(
        arrays["labels"], arrays["dino_scores"], arrays["folds"]
    )
    fusion = b13.classification_summary(
        arrays["labels"], readout["scores"], arrays["folds"]
    )
    bootstrap = b13.paired_component_bootstrap(
        labels=arrays["labels"],
        folds=arrays["folds"],
        groups=arrays["groups"],
        dino_scores=arrays["dino_scores"],
        candidate_scores=readout["scores"],
        replicates=BOOTSTRAP_REPLICATES,
        seed=SEED,
    )
    end_git, end_sources = _git(), _sources()
    integrity = bool(start_git == end_git and start_sources == end_sources)
    gate = assess_gate(
        dino=dino,
        fusion=fusion,
        bootstrap=bootstrap,
        integrity_complete=integrity,
        readouts_converged=bool(readout["converged"]),
    )
    output.mkdir(parents=True, exist_ok=False)
    scores_sha = b13._atomic_npz(
        output / "train_fusion_readout.npz",
        labels=np.asarray(arrays["labels"], dtype=np.int64),
        folds=np.asarray(arrays["folds"], dtype=np.int64),
        groups=np.asarray(arrays["groups"], dtype=np.int64),
        dino_scores=np.asarray(arrays["dino_scores"], dtype=np.float64),
        fusion_scores=np.asarray(readout["scores"], dtype=np.float64),
    )
    summary = {
        "schema_version": 1,
        "protocol_id": PROTOCOL_ID,
        "mode": "formal_solver_only_train_oof_resolution",
        "accepted_preflight": accepted,
        "source_hashes": start_sources,
        "git": {"start": start_git, "end": end_git},
        "dataset": {
            "train_split_used": True,
            "validation_split_used": False,
            "test_split_used": False,
            "rows": 8278,
        },
        "readout": {
            "C": b13.LOGISTIC_C,
            "class_weight": "balanced",
            "solver": "lbfgs",
            "tol": b13.LOGISTIC_TOL,
            "max_iter": MAX_ITER,
            "converged": bool(readout["converged"]),
            "fold_records": readout["fold_records"],
        },
        "metrics": {"dino": dino, "fusion": fusion},
        "bootstrap": bootstrap,
        "gate": gate,
        "scores_sha256": scores_sha,
        "integrity_complete": integrity,
    }
    b13._atomic_json(output / "summary.json", summary)
    return summary


def main(argv: Sequence[str] | None = None) -> int:
    args = _args(argv)
    output = args.output_dir.expanduser().resolve()
    if args.preflight_only:
        if output.exists():
            raise RuntimeError(f"refuse to overwrite B25 preflight: {output}")
        payload = _preflight(args)
        if not payload["passed"]:
            raise RuntimeError(f"B25 preflight failed: {payload['checks']}")
        output.mkdir(parents=True, exist_ok=False)
        b13._atomic_json(output / "preflight.json", payload)
        print(json.dumps({"passed": True, "output": str(output)}, indent=2))
        return 0
    summary = _formal(args)
    print(json.dumps({"gate": summary["gate"], "output": str(output)}, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
