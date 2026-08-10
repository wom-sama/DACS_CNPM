from __future__ import annotations

import json
import os
from pathlib import Path

import numpy as np
import pytest
import torch

from trkh.tools.audit_cap_integral_region_context_a0 import (
    DataAccessLedger,
    LOCK_PATH,
    REPO_ROOT,
    VISUAL_ANCHORS,
    _manifest_payload,
    _maximum_array_error,
    _eval_cap_feature_gradient,
    _project_canonical_map,
    _recursive_numeric_difference,
    _region_attention_map,
    _role_core_gate_checks,
    _verify_manifest,
    build_fold_protocol,
    finalize_visual_review,
    load_lock,
    load_locked_inputs,
    parse_args,
    verify_engine_contract,
    verify_locked_files,
)
from trkh.tools.cap_integral_region_context_a0_engine import (
    CAPIntegralRegionBinaryHead,
)


def test_locked_inputs_fold_protocol_and_ledger_reproduce() -> None:
    lock = load_lock()
    ledger = DataAccessLedger(lock, lock_path=LOCK_PATH)
    with ledger:
        files = verify_locked_files(lock)
        inputs = load_locked_inputs(lock)
        folds = build_fold_protocol(inputs, lock)
    snapshot = ledger.snapshot()
    assert files["passed"]
    assert all(inputs["input_checks"].values())
    assert inputs["input_checks"]["spatial_permutations"]
    assert inputs["input_checks"]["visual_anchors"]
    assert float(inputs["cidt_geometry_probability_max_abs_error"]) == pytest.approx(
        0.0015103518962860107,
        abs=0.0,
    )
    assert int(inputs["cidt_geometry_rival_argmax_difference_count"]) == 1
    assert len(folds) == 5
    assert all(all(record["checks"].values()) for record in folds)
    assert snapshot["passed"]
    assert snapshot["validation_open_count"] == 0
    assert snapshot["test_open_count"] == 0
    assert snapshot["observed_event_count"] > 0

    positions = {
        int(sample_index): position
        for position, sample_index in enumerate(inputs["sample_indices"])
    }
    for offset, anchor in enumerate(VISUAL_ANCHORS):
        position = positions[anchor]
        assert int(inputs["folds"][position]) == offset // 4
        expected = (
            (1, 1),
            (1, 0),
            (0, 1),
            (4 if offset // 4 in {0, 2, 3, 4} else 2, 1),
        )[offset % 4]
        assert (
            int(inputs["targets"][position]),
            int(np.argmax(inputs["keeper_probabilities"][position])),
        ) == expected


def test_data_access_ledger_blocks_forbidden_and_undeclared_paths() -> None:
    lock = load_lock()
    ledger = DataAccessLedger(lock, lock_path=LOCK_PATH)
    with pytest.raises(PermissionError, match="forbidden_split_component"):
        ledger.observe_open(
            (
                r"D:\DataAI\AIEx\newdataset\yolo_f\images\validation\x.jpg",
                "r",
                os.O_RDONLY,
            )
        )
    with pytest.raises(PermissionError, match="undeclared_input"):
        ledger.observe_open(
            (
                r"D:\DataAI\AIEx\TRKH\runs\unknown\payload.npy",
                "r",
                os.O_RDONLY,
            )
        )
    with pytest.raises(PermissionError, match="write_to_data_domain"):
        ledger.observe_open(
            (
                r"D:\DataAI\external_sources\cap_official\LICENSE",
                "w",
                os.O_WRONLY,
            )
        )

    allowed = DataAccessLedger(lock, lock_path=LOCK_PATH)
    visual_paths = [
        rf"D:\DataAI\AIEx\newdataset\yolo_f\images\train\visual_{index}.jpg"
        for index in range(20)
    ]
    allowed.authorize_visual_paths(visual_paths)
    allowed.observe_open((visual_paths[0], "r", os.O_RDONLY))
    assert allowed.events[-1]["decision"] == "allowed"


def test_engine_contract_excludes_script_only_cap_variant() -> None:
    evidence = verify_engine_contract(load_lock())
    assert evidence["passed"]
    assert evidence["checks"]["same_budget_contracts_exact"]
    assert evidence["checks"]["primary_initial_states_exact"]
    assert evidence["checks"]["repeat_initial_state_differs"]
    assert evidence["checks"]["visual_anchors_exact"]
    assert evidence["checks"]["no_sigmoid_module"]
    assert evidence["checks"]["no_squeeze_excitation_module"]
    assert evidence["checks"]["no_spectral_norm_parameter"]


def _role_record(
    *,
    auc: float,
    fn_auc: float,
) -> dict[str, object]:
    return {
        "tp_vs_restricted_fp_auroc": auc,
        "fn_vs_restricted_fp_auroc": fn_auc,
        "tp_retention": 0.98,
        "restricted_fp_rejection": 0.30,
        "corrections": 60,
        "tp_harms": 10,
        "net_corrections": 50,
        "macro_f1_gain": 0.003,
        "class1_precision_gain": 0.04,
        "class1_f1_gain": 0.02,
        "class1_recall_drop": 0.02,
        "maximum_nonfocus_f1_drop": 0.002,
        "fold_aurocs": {str(fold): auc for fold in range(5)},
        "per_fold": [
            {
                "tp_retention": 0.95,
                "restricted_fp_rejection": 0.20,
                "restricted_fp_rejected": 1,
            }
            for _ in range(5)
        ],
    }


def test_locked_candidate_core_gate_is_conjunctive() -> None:
    lock = load_lock()
    analysis = {
        "roles": {
            "keeper_margin_control": _role_record(
                auc=0.75, fn_auc=0.55
            ),
            "global_gap_linear_control": _role_record(
                auc=0.80, fn_auc=0.61
            ),
            "integral_self_only_control": _role_record(
                auc=0.805, fn_auc=0.62
            ),
            "cap_context_candidate": _role_record(
                auc=0.84, fn_auc=0.66
            ),
            "cap_spatial_deranged_control": _role_record(
                auc=0.81, fn_auc=0.62
            ),
            "cap_cross_sample_context_control": _role_record(
                auc=0.815, fn_auc=0.62
            ),
        }
    }
    checks = _role_core_gate_checks(
        "cap_context_candidate",
        analysis=analysis,
        thresholds=lock["performance_gates"],
    )
    assert all(checks.values())
    analysis["roles"]["cap_context_candidate"]["per_fold"][2][
        "restricted_fp_rejection"
    ] = 0.09
    failed = _role_core_gate_checks(
        "cap_context_candidate",
        analysis=analysis,
        thresholds=lock["performance_gates"],
    )
    assert not failed["every_fold_restricted_fp_rejection"]
    assert not all(failed.values())


def test_visual_region_rasterization_and_support_projection_are_normalized() -> None:
    attention = np.eye(27, dtype=np.float64)
    region_map = _region_attention_map(attention)
    assert region_map.shape == (42, 42)
    assert np.isfinite(region_map).all()
    assert float(region_map.sum()) == pytest.approx(1.0, abs=2e-7)
    assert float(region_map.std()) > 0.0

    canonical = np.arange(16 * 16, dtype=np.float64).reshape(16, 16)
    projected = _project_canonical_map(canonical, [3, 2, 13, 14])
    assert projected.shape == (16, 16)
    assert float(projected.sum()) == pytest.approx(1.0, abs=1e-7)
    outside = np.ones((16, 16), dtype=np.bool_)
    outside[2:14, 3:13] = False
    assert not bool(projected[outside].any())


@pytest.mark.skipif(
    not torch.cuda.is_available(),
    reason="cuDNN LSTM eval-backward regression requires CUDA",
)
def test_eval_lstm_feature_attribution_uses_native_rnn_without_mode_drift() -> None:
    device = torch.device("cuda")
    torch.manual_seed(20260724)
    model = CAPIntegralRegionBinaryHead(context_mode="context").to(device)
    model.eval()
    for parameter in model.parameters():
        parameter.requires_grad_(False)
    features = torch.randn(
        1,
        256,
        16,
        16,
        dtype=torch.float32,
        device=device,
    )
    support_boxes = np.asarray([[0, 0, 16, 16]], dtype=np.int64)
    with torch.no_grad():
        reference = model(features, support_boxes)
    leaf = features.detach().clone().requires_grad_(True)
    cudnn_enabled = torch.backends.cudnn.enabled

    try:
        logit, auxiliary, gradient = _eval_cap_feature_gradient(
            model,
            leaf,
            support_boxes,
        )
        assert not model.training
        assert not model.lstm.training
        assert torch.backends.cudnn.enabled == cudnn_enabled
        assert tuple(logit.shape) == (1,)
        assert tuple(gradient.shape) == tuple(leaf.shape)
        assert torch.isfinite(gradient).all()
        assert float(gradient.abs().sum()) > 0.0
        assert "region_attention" in auxiliary
        assert torch.allclose(logit.detach(), reference, atol=1e-5, rtol=0.0)
    finally:
        model.to(torch.device("cpu"))
        del model, features, leaf
        torch.cuda.empty_cache()


def test_replay_comparators_preserve_discrete_types_and_numeric_tolerance() -> None:
    left = {
        "float": np.asarray([1.0, 2.0], dtype=np.float32),
        "bool": np.asarray([True, False], dtype=np.bool_),
    }
    right = {
        "float": np.asarray([1.0, 2.0 + 5e-8], dtype=np.float32),
        "bool": np.asarray([True, False], dtype=np.bool_),
    }
    error, exact = _maximum_array_error(left, right)
    assert error <= 1e-7
    assert exact
    error, exact = _recursive_numeric_difference(
        {"value": True}, {"value": np.bool_(True)}
    )
    assert error == 0.0
    assert exact
    error, exact = _recursive_numeric_difference(
        {"value": True}, {"value": 1}
    )
    assert error == 0.0
    assert not exact
    error, exact = _recursive_numeric_difference(
        {"value": 1}, {"value": 1.0}
    )
    assert error == 0.0
    assert not exact


def test_manifest_and_cli_are_fail_closed(tmp_path: Path) -> None:
    (tmp_path / "a.txt").write_text("a\n", encoding="utf-8")
    (tmp_path / "b.bin").write_bytes(b"\x00\x01")
    manifest = _manifest_payload(
        tmp_path, excluded_names=("manifest.json",)
    )
    (tmp_path / "manifest.json").write_text(
        json.dumps(manifest, indent=2) + "\n",
        encoding="utf-8",
    )
    assert _verify_manifest(tmp_path, "manifest.json")["passed"]
    (tmp_path / "unexpected.txt").write_text("extra\n", encoding="utf-8")
    unexpected = _verify_manifest(tmp_path, "manifest.json")
    assert not unexpected["passed"]
    assert unexpected["unexpected_files"] == ["unexpected.txt"]
    (tmp_path / "unexpected.txt").unlink()
    (tmp_path / "a.txt").write_text("changed\n", encoding="utf-8")
    assert not _verify_manifest(tmp_path, "manifest.json")["passed"]

    traversal = dict(manifest)
    traversal["files"] = [
        {
            "path": "../outside.txt",
            "bytes": 0,
            "sha256": "0" * 64,
        }
    ]
    (tmp_path / "manifest.json").write_text(
        json.dumps(traversal, indent=2) + "\n",
        encoding="utf-8",
    )
    traversal_check = _verify_manifest(tmp_path, "manifest.json")
    assert not traversal_check["passed"]
    assert traversal_check["record_errors"] == ["../outside.txt"]
    traversal["files"] = manifest["files"]
    traversal["files_sha256"] = "0" * 64
    (tmp_path / "manifest.json").write_text(
        json.dumps(traversal, indent=2) + "\n",
        encoding="utf-8",
    )
    digest_check = _verify_manifest(tmp_path, "manifest.json")
    assert not digest_check["passed"]
    assert not digest_check["files_digest_exact"]

    assert parse_args(["--mode", "preflight"]).mode == "preflight"
    with pytest.raises(SystemExit):
        parse_args(["--mode", "all"])


def test_replay_manifest_protects_one_shot_visual_finalization(
    tmp_path: Path,
) -> None:
    (tmp_path / "summary.json").write_text(
        json.dumps(
            {
                "protocol_id": "cap-test",
                "automatic_passed": False,
            }
        )
        + "\n",
        encoding="utf-8",
    )
    (tmp_path / "visual_evidence.json").write_text(
        json.dumps(
            {
                "records": [
                    {"sample_index": int(value)}
                    for value in VISUAL_ANCHORS
                ],
                "visual_arrays_sha256": "visual-sha",
            }
        )
        + "\n",
        encoding="utf-8",
    )
    (tmp_path / "xai_contact_sheet.png").write_bytes(b"sheet")
    formal_manifest = _manifest_payload(
        tmp_path,
        excluded_names=(
            "formal_manifest.json",
            "replay_summary.json",
            "replay_manifest.json",
            "manual_visual_review.json",
            "final_decision.json",
            "artifact_set_manifest.json",
        ),
    )
    (tmp_path / "formal_manifest.json").write_text(
        json.dumps(formal_manifest, indent=2) + "\n",
        encoding="utf-8",
    )
    (tmp_path / "replay_summary.json").write_text(
        json.dumps({"passed": True}) + "\n",
        encoding="utf-8",
    )
    replay_manifest = _manifest_payload(
        tmp_path,
        excluded_names=(
            "replay_manifest.json",
            "manual_visual_review.json",
            "final_decision.json",
            "artifact_set_manifest.json",
        ),
    )
    (tmp_path / "replay_manifest.json").write_text(
        json.dumps(replay_manifest, indent=2) + "\n",
        encoding="utf-8",
    )

    decision = finalize_visual_review(
        output=tmp_path,
        approved=False,
        notes="Reviewed all fixed rows; reject in this fixture.",
    )
    assert decision["decision"] == "reject_exact_cap_a0"
    assert decision["formal_manifest_verified"]
    assert decision["replay_manifest_verified"]
    assert _verify_manifest(
        tmp_path, "artifact_set_manifest.json"
    )["passed"]
    with pytest.raises(FileExistsError):
        finalize_visual_review(
            output=tmp_path,
            approved=True,
            notes="A second finalization must fail closed.",
        )


def test_powershell_launcher_uses_native_exit_codes_and_fresh_process_modes() -> None:
    launcher = (
        REPO_ROOT
        / "scripts"
        / "run_trkh_cap_integral_region_context_a0.ps1"
    ).read_text(encoding="utf-8")
    assert "$LASTEXITCODE" in launcher
    assert "2>&1" not in launcher
    assert '"--mode", $Mode' in launcher
    assert 'Invoke-CapMode -Mode "formal"' in launcher
    assert 'Invoke-CapMode -Mode "replay"' in launcher
    assert "--approve-visual" in launcher
    assert "function Wait-AvailablePhysicalMemory" in launcher
    assert "Waiting for post-test memory recovery" in launcher
    assert "$MinimumGiB = 3.5" in launcher
