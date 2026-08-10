from __future__ import annotations

import copy
import math
from argparse import Namespace
from pathlib import Path

import numpy as np
import pytest
import timm
import torch
from PIL import Image

from trkh.tools import audit_swiftformer_surfacefold_b19_preflight as preflight_b19
from trkh.tools import run_swiftformer_surfacefold_b19_train_oof as b19


def _base_model() -> torch.nn.Module:
    model = timm.create_model(
        b19.STUDENT_TIMM_ID,
        pretrained=False,
        num_classes=1000,
        drop_rate=0.0,
        drop_path_rate=0.0,
    )
    b19.reset_locked_five_class_heads(model)
    return model


@pytest.fixture(autouse=True)
def _reset_metric_globals() -> None:
    b19._METRIC_BACKEND_LOADED = False
    b19._VALID_BARRIER_TOKENS.clear()
    b19._PENDING_BARRIER_SEALS.clear()
    b19._ACTIVE_BARRIER_SEALS.clear()


def _preflight_payload(git: dict[str, object], sources: dict[str, str]) -> dict[str, object]:
    payload = {
        "schema_version": 1,
        "protocol_id": b19.PROTOCOL_ID,
        "protocol_sha256": b19.PROTOCOL_SHA256,
        "mode": "offline_label_free_stock_first_preflight",
        "created_at_unix": 1.0,
        "git": git,
        "source_hashes": sources,
        "runtime": {"locked": True},
        "timm_sources": {"locked": True},
        "device": {"locked": True},
        "weights": {"locked": True},
        "strict_load": {"passed": True},
        "focused_tests": {"passed": True},
        "stock_deployment": {"passed": True},
        "mechanism": {"passed": True},
        "folded_deployment": {"passed": True},
        "cuda": {"passed": True},
        "isolation": {"dataset_attempts": [], "network_attempts": [], "process_attempts": []},
        "offline": {"passed": True},
        "permissions": {
            "dataset_yaml_read": False,
            "dataset_path_enumerated": False,
            "train_image_read": False,
            "real_label_read": False,
            "prior_prediction_read": False,
            "validation_constructed": False,
            "test_constructed": False,
            "network_used": False,
            "formal_train_permission": True,
        },
        "authorization": {
            "scope": "one_train_only_five_fold_causal_screen",
            "deployment_status": "DEVICE_PENDING",
            "validation_permission": False,
            "test_permission": False,
        },
        "checks": {
            name: True
            for name in (
                "clean_committed_canonical_branch",
                "bound_sources_exact",
                "runtime_and_timm_exact",
                "offline_assets_strict",
                "focused_tests_no_skip",
                "stock_deployment_first",
                "mechanism_and_gradient_contract",
                "folded_topology_contract",
                "cuda_batch16_contract",
                "no_dataset_or_network_access",
                "start_end_rehash_exact",
                "train_only_authorization",
            )
        },
        "passed": True,
    }
    payload["end_rehash"] = {
        key: payload[key] for key in ("git", "source_hashes", "runtime", "timm_sources", "weights")
    }
    return payload


def _published_preflight(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    git: dict[str, object],
    sources: dict[str, str],
) -> dict[str, object]:
    monkeypatch.setattr(b19, "_repository_root", lambda: tmp_path)
    directory = tmp_path / "runs" / "preflight_b19_surfacefold_xs_unit"
    directory.mkdir(parents=True)
    payload = _preflight_payload(git, sources)
    artifact = directory / "preflight.json"
    artifact.write_bytes(b19._json_bytes(payload))
    digest = b19.sha256_file(artifact)
    (directory / "preflight.sha256").write_bytes(f"{digest}\n".encode("ascii"))
    return {"artifact": str(artifact.resolve()), "sha256": digest, "payload": payload}


def _barrier_token(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> tuple[b19.MetricBarrierToken, list[b19.FinalStateRecord]]:
    git = {
        "head": "a" * 40,
        "branch": b19.EXPECTED_BRANCH,
        "status": "",
        "clean": True,
        "head_is_commit": True,
    }
    sources = {"runner": "b" * 64}
    accepted = _published_preflight(tmp_path, monkeypatch, git, sources)
    monkeypatch.setattr(b19, "_git_contract", lambda: dict(git))
    monkeypatch.setattr(b19, "_source_hashes", lambda: dict(sources))
    monkeypatch.setattr(b19, "ARM_PARAMETERS", {arm: 1 for arm in b19.ARMS})
    monkeypatch.setattr(b19, "ARM_TRAINABLE_PARAMETERS", {arm: 1 for arm in b19.ARMS})
    output = tmp_path / "formal"
    (output / "states").mkdir(parents=True)
    args = Namespace(
        data=tmp_path / "never_open_data.yaml",
        assignment_csv=tmp_path / "never_open_assignment.csv",
        student_weight=tmp_path / "never_open_student.safetensors",
        dino_weight=tmp_path / "never_open_dino.safetensors",
    )
    claim = b19.claim_train_authorization_once(
        accepted_preflight=accepted,
        git=git,
        sources=sources,
        output=output,
        args=args,
    )
    records: list[b19.FinalStateRecord] = []
    for fold in range(b19.FOLDS):
        fold_dir = output / "folds" / f"fold_{fold}"
        fold_dir.mkdir(parents=True)
        exposure = fold_dir / "exposure.npz"
        losses = fold_dir / "loss_curves.npz"
        np.savez(exposure, x=np.asarray([fold]))
        np.savez(losses, x=np.asarray([fold]))
        for arm in b19.ARMS:
            path = (output / "states" / f"fold_{fold}_{arm}.safetensors").resolve()
            metadata = b19._metadata_for_state(fold, arm)
            digest = b19.atomic_safetensors(path, {"weight": torch.tensor([float(fold)])}, metadata)
            records.append(
                b19.FinalStateRecord(
                    state_id=f"fold_{fold}:{arm}",
                    fold=fold,
                    arm=arm,
                    path=str(path),
                    sha256=digest,
                    bytes=path.stat().st_size,
                    parameter_count=1,
                    trainable_parameter_count=1,
                    mode=b19.ARM_MODES[arm],
                    metadata_sha256=b19._metadata_sha256(metadata),
                    exposure_sha256=b19.sha256_file(exposure),
                    loss_curve_sha256=b19.sha256_file(losses),
                    frozen_projection_sha256="c" * 64,
                    folded_max_abs_error=0.0,
                )
            )
    reloaded: list[str] = []
    cross_mode = []
    token = b19.seal_metric_barrier(
        output,
        records,
        accepted_preflight=accepted,
        authorization_claim=claim,
        args=args,
        git=git,
        sources=sources,
        counters=dict(b19.ZERO_BARRIER_COUNTERS),
        strict_reload=lambda record: reloaded.append(record.state_id),
        cross_mode_rejection=lambda: cross_mode.append(True),
    )
    assert len(reloaded) == 15
    assert cross_mode == [True]
    return token, records


def test_preflight_authorization_is_consumed_exactly_once_before_data(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    git = {"head": "a" * 40}
    sources = {"runner": "b" * 64}
    accepted = _published_preflight(tmp_path, monkeypatch, git, sources)
    args = Namespace(
        data=tmp_path / "must_not_be_opened.yaml",
        assignment_csv=tmp_path / "must_not_be_opened.csv",
        student_weight=tmp_path / "student.safetensors",
        dino_weight=tmp_path / "dino.safetensors",
    )
    output = tmp_path / "runs" / "formal"
    first = b19.claim_train_authorization_once(
        accepted_preflight=accepted,
        git=git,
        sources=sources,
        output=output,
        args=args,
    )
    assert (
        b19.validate_train_authorization_claim(
            first,
            accepted_preflight=accepted,
            git=git,
            sources=sources,
            output=output,
            args=args,
        )
        == first
    )
    with pytest.raises(b19.B19ContractError, match="already consumed"):
        b19.claim_train_authorization_once(
            accepted_preflight=accepted,
            git=git,
            sources=sources,
            output=tmp_path / "runs" / "second",
            args=args,
        )


def test_authorization_and_formal_rng_probe_fail_before_any_data_access(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    output = tmp_path / "formal"
    git = {
        "head": "a" * 40,
        "branch": b19.EXPECTED_BRANCH,
        "status": "",
        "clean": True,
        "head_is_commit": True,
    }
    monkeypatch.setattr(b19, "_git_contract", lambda: git)
    monkeypatch.setattr(b19, "_source_hashes", lambda: {"formal_runner": "b" * 64})
    monkeypatch.setattr(
        b19,
        "validate_accepted_preflight",
        lambda *_args, **_kwargs: {
            "artifact": str(tmp_path / "preflight.json"),
            "sha256": "c" * 64,
            "payload": {},
        },
    )
    monkeypatch.setattr(b19, "_validated_output", lambda _path: output)
    monkeypatch.setattr(
        b19,
        "claim_train_authorization_once",
        lambda **_kwargs: (_ for _ in ()).throw(b19.B19ContractError("already consumed")),
    )
    data_accessed = []
    monkeypatch.setattr(
        b19,
        "_read_locked_data_root",
        lambda _path: data_accessed.append(True),
    )
    args = Namespace(
        confirm_protocol_id=b19.PROTOCOL_ID,
        data=tmp_path / "must_not_open.yaml",
        assignment_csv=tmp_path / "must_not_open.csv",
        student_weight=tmp_path / "student.safetensors",
        dino_weight=tmp_path / "dino.safetensors",
        preflight_artifact=tmp_path / "preflight.json",
        preflight_sha256="c" * 64,
        output_dir=output,
    )
    with pytest.raises(b19.B19ContractError, match="already consumed"):
        b19.run_formal(args)
    assert data_accessed == []
    assert (output / "failure.json").is_file()

    probe_output = tmp_path / "formal_probe_failure"
    monkeypatch.setattr(b19, "_validated_output", lambda _path: probe_output)
    monkeypatch.setattr(
        b19,
        "claim_train_authorization_once",
        lambda **_kwargs: {"path": "unused", "sha256": "d" * 64, "payload": {}},
    )
    monkeypatch.setattr(
        b19,
        "_formal_initialized_cuda_head_reset_probe",
        lambda: (_ for _ in ()).throw(b19.B19ContractError("formal RNG probe failed")),
    )
    args.output_dir = probe_output
    with pytest.raises(b19.B19ContractError, match="formal RNG probe failed"):
        b19.run_formal(args)
    assert data_accessed == []
    assert (probe_output / "failure.json").is_file()


def test_runner_has_no_retired_runtime_or_protocol_binding() -> None:
    source = Path(b19.__file__).read_text(encoding="utf-8")
    assert "swiftformer_surface_b16" not in source
    assert "run_swiftformer_surface_b16_train_oof" not in source
    assert "audit_swiftformer_surface_b16_preflight" not in source
    assert "run_swiftformer_surfacefold_b17_train_oof" not in source
    assert "audit_swiftformer_surfacefold_b17_preflight" not in source
    assert "b17_train_authorization_claims" not in source
    assert "swiftformer_surfacefold_b17" in source
    assert b19.PROTOCOL_ID == "TRKH_PRETRAINED_CLASSF_B19_BITEXACT_LR_SURFACEFOLD_XS_20260805"


def test_preflight_schema_is_exact_and_train_only() -> None:
    git = {"head": "a" * 40}
    sources = {"runner": "b" * 64}
    payload = _preflight_payload(git, sources)
    assert (b19.PROTOCOL_ID, b19.PROTOCOL_SHA256, b19.PREFLIGHT_OUTPUT_PREFIX) == (
        preflight_b19.PROTOCOL_ID,
        preflight_b19.PROTOCOL_SHA256,
        preflight_b19.OUTPUT_PREFIX,
    )
    assert payload["permissions"] == preflight_b19.SUCCESS_PERMISSIONS
    assert payload["authorization"] == preflight_b19.AUTHORIZATION
    assert set(payload["checks"]) == set(preflight_b19.TOP_CHECKS)
    assert payload["isolation"] == {
        "dataset_attempts": [],
        "network_attempts": [],
        "process_attempts": [],
    }
    b19._validate_preflight_payload(payload, git=git, sources=sources)
    for mutation in (
        lambda row: row["permissions"].__setitem__("validation_constructed", True),
        lambda row: row["authorization"].__setitem__("deployment_status", "READY"),
        lambda row: row.__setitem__("unexpected", True),
        lambda row: row["checks"].__setitem__("stock_deployment_first", False),
        lambda row: row["isolation"]["process_attempts"].append("python child.py"),
    ):
        changed = copy.deepcopy(payload)
        mutation(changed)
        with pytest.raises(b19.B19ContractError, match="preflight"):
            b19._validate_preflight_payload(changed, git=git, sources=sources)


def test_assignment_requires_component_disjoint_class_complete_folds() -> None:
    paths, labels, groups, folds = [], [], [], []
    for fold in range(b19.FOLDS):
        for label in range(b19.CLASSES):
            paths.append(f"train/{label}/{fold}.jpg")
            labels.append(label)
            groups.append(fold * 10 + label)
            folds.append(fold)
    contract = b19.validate_assignment_arrays(
        paths,
        np.asarray(labels),
        np.asarray(groups),
        np.asarray(folds),
        enforce_locked=False,
    )
    assert len(contract["fold_rows"]) == 5
    leaked = np.asarray(groups)
    leaked[-1] = leaked[0]
    with pytest.raises(ValueError, match="component-disjoint"):
        b19.validate_assignment_arrays(
            paths,
            np.asarray(labels),
            leaked,
            np.asarray(folds),
            enforce_locked=False,
        )


def test_augmentation_is_private_deterministic_and_shared() -> None:
    image = Image.new("RGB", (257, 193), (30, 90, 180))
    first = torch.Generator().manual_seed(b19.AUGMENTATION_SEED_BASE)
    second = torch.Generator().manual_seed(b19.AUGMENTATION_SEED_BASE)
    crop1 = b19.sample_random_resized_crop(193, 257, first)
    crop2 = b19.sample_random_resized_crop(193, 257, second)
    assert crop1 == crop2
    student, teacher = b19.build_shared_train_views(image, crop1)
    assert student.shape == (3, 224, 224)
    assert teacher.shape == (3, 256, 256)
    assert torch.isfinite(student).all() and torch.isfinite(teacher).all()


def test_head_reset_and_arm_construction_are_rng_safe_and_exact() -> None:
    cuda_evidence = b19._formal_initialized_cuda_head_reset_probe()
    assert cuda_evidence == {
        "cuda_initialized": True,
        "device_count": torch.cuda.device_count(),
        "cpu_rng_equal": True,
        "all_cuda_rng_equal": True,
        "reset": {
            "seed": b19.SEED,
            "order": ["head", "head_dist"],
            "initializer": "timm.trunc_normal_std_0.02_bias_zero",
        },
    }
    torch.manual_seed(77)
    model = timm.create_model(b19.STUDENT_TIMM_ID, pretrained=False, num_classes=1000)
    caller = torch.random.get_rng_state().clone()
    b19.reset_locked_five_class_heads(model)
    assert torch.equal(caller, torch.random.get_rng_state())
    duplicate = copy.deepcopy(model)
    b19.reset_locked_five_class_heads(duplicate)
    assert torch.equal(model.head.weight, duplicate.head.weight)
    arms = b19.build_b19_arm_models(model)
    assert {
        name: sum(parameter.numel() for parameter in arm.parameters()) for name, arm in arms.items()
    } == b19.ARM_PARAMETERS
    control = arms["surfacefold_mean_control"]
    candidate = arms["surfacefold_spatial_candidate"]
    assert torch.equal(control.factor_p, candidate.factor_p)
    assert torch.equal(control.factor_d, candidate.factor_d)
    assert torch.equal(control.effective_delta_weight(), candidate.effective_delta_weight())
    for arm in arms.values():
        projection = arm.backbone.stages[3].downsample.proj
        assert not projection.weight.requires_grad
        assert not projection.bias.requires_grad
    with pytest.raises(RuntimeError, match="immutable B17 mode contract"):
        control.load_state_dict(candidate.state_dict(), strict=True)


def test_optimizer_groups_exclude_frozen_projection_and_promote_factors() -> None:
    candidate = b19.build_b19_arm_models(_base_model())["surfacefold_spatial_candidate"]
    groups = b19.build_discriminative_adamw_groups(candidate)
    names = {group["name"] for group in groups}
    assert names == {
        "backbone_decay",
        "backbone_no_decay",
        "task_decay",
        "task_no_decay",
    }
    assert sum(int(group["parameter_count"]) for group in groups) == sum(
        parameter.numel() for parameter in candidate.parameters() if parameter.requires_grad
    )
    projection_ids = {
        id(candidate.backbone.stages[3].downsample.proj.weight),
        id(candidate.backbone.stages[3].downsample.proj.bias),
    }
    assert not projection_ids & {id(parameter) for group in groups for parameter in group["params"]}
    task_groups = [group for group in groups if group["name"].startswith("task")]
    assert all(group["peak_lr"] == b19.TASK_LR for group in task_groups)


def test_teacher_target_and_all_arm_relation_loss_are_exact() -> None:
    class Teacher(torch.nn.Module):
        def forward_features(self, images: torch.Tensor) -> torch.Tensor:
            values = torch.linspace(
                -1.0,
                1.0,
                images.size(0) * 261 * b19.DINO_FEATURE_DIM,
                dtype=torch.float32,
            )
            return values.reshape(images.size(0), 261, b19.DINO_FEATURE_DIM)

    ledger = b19.TeacherCallLedger()
    target = b19.teacher_relation_target(Teacher(), torch.zeros(2, 3, 256, 256), ledger)
    assert target.shape == (2, 84)
    assert not target.requires_grad
    assert (ledger.calls, ledger.rows) == (1, 2)
    for _arm in b19.ARMS:
        s3 = torch.randn(2, 220, 7, 7, requires_grad=True)
        logits = torch.randn(2, b19.CLASSES, requires_grad=True)
        labels = torch.tensor([0, 1])
        weights = torch.ones(b19.CLASSES)
        total, ce, relation = b19.exact_b19_objective(logits, s3, labels, weights, target)
        assert torch.equal(total, ce + b19.RELATION_WEIGHT * relation)
        total.backward()
        assert torch.isfinite(total)
        assert s3.grad is not None and torch.isfinite(s3.grad).all()
        assert logits.grad is not None and torch.isfinite(logits.grad).all()


def test_formal_update_horizons_are_derived_from_locked_fold_counts() -> None:
    held_counts = (1823, 1694, 1521, 1655, 1585)
    assert sum(held_counts) == b19.EXPECTED_TRAIN_ROWS
    folds = np.concatenate(
        [np.full(count, fold, dtype=np.int64) for fold, count in enumerate(held_counts)]
    )
    assert b19.derive_locked_updates_per_epoch(folds) == (202, 206, 212, 207, 210)

    mutated = folds.copy()
    mutated[:32] = 1
    with pytest.raises(b19.B19ContractError, match="assignment-derived optimizer horizons changed"):
        b19.derive_locked_updates_per_epoch(mutated)
    with pytest.raises(b19.B19ContractError, match="one-dimensional integer vector"):
        b19.derive_locked_updates_per_epoch(folds.reshape(-1, 1))
    invalid = folds.copy()
    invalid[0] = b19.FOLDS
    with pytest.raises(b19.B19ContractError, match="exactly folds 0..4"):
        b19.derive_locked_updates_per_epoch(invalid)


def test_locked_lr_is_bit_exact_at_real_fold_horizons_and_b18_failure_is_reproduced() -> None:
    fold_updates = (202, 206, 212, 207, 210)
    retired_fold0 = b19.BACKBONE_LR * (fold_updates[0] - 1) / (fold_updates[0] - 1)
    assert retired_fold0 != b19.BACKBONE_LR
    assert b19.BACKBONE_LR - retired_fold0 == math.ulp(b19.BACKBONE_LR)
    assert fold_updates == b19.EXPECTED_UPDATES_PER_EPOCH

    def retired_b18_lr(peak: float, update: int, steps_per_epoch: int, total_steps: int) -> float:
        if update < steps_per_epoch:
            return float(peak * update / (steps_per_epoch - 1))
        progress = (update - steps_per_epoch + 1) / (total_steps - steps_per_epoch)
        return float(
            b19.MIN_LR
            + (peak - b19.MIN_LR) * (1.0 + math.cos(math.pi * progress)) / 2.0
        )

    checked_schedules: set[tuple[int, float]] = set()
    endpoint_differences: set[tuple[int, float, int]] = set()

    for steps_per_epoch in fold_updates:
        group_peaks = {
            "backbone_decay": b19.BACKBONE_LR,
            "backbone_no_decay": b19.BACKBONE_LR,
            "task_decay": b19.TASK_LR,
            "task_no_decay": b19.TASK_LR,
        }
        parameters = {
            name: torch.nn.Parameter(torch.ones(())) for name in group_peaks
        }
        optimizer = torch.optim.AdamW(
            [
                {
                    "params": [parameters[name]],
                    "name": name,
                    "peak_lr": peak,
                    "lr": peak,
                }
                for name, peak in group_peaks.items()
            ]
        )
        stepper = b19.LockedOptimizerStepper(optimizer, steps_per_epoch=steps_per_epoch)
        previous_decay = dict(group_peaks)

        for update in range(stepper.total_steps):
            for parameter in parameters.values():
                parameter.grad = torch.ones_like(parameter)
            applied = stepper.step()
            assert set(applied) == set(group_peaks)
            for group in optimizer.param_groups:
                name = str(group["name"])
                assert b19._float64_bits(float(group["lr"])) == b19._float64_bits(applied[name])
            for name, peak in group_peaks.items():
                checked_schedules.add((steps_per_epoch, peak))
                retired = retired_b18_lr(peak, update, steps_per_epoch, stepper.total_steps)
                if update == 0:
                    expected = 0.0
                elif update == steps_per_epoch - 1:
                    expected = peak
                elif update == stepper.total_steps - 1:
                    expected = b19.MIN_LR
                elif update < steps_per_epoch:
                    expected = float(peak * update / (steps_per_epoch - 1))
                else:
                    progress = (update - steps_per_epoch + 1) / (
                        stepper.total_steps - steps_per_epoch
                    )
                    expected = float(
                        b19.MIN_LR
                        + (peak - b19.MIN_LR)
                        * (1.0 + math.cos(math.pi * progress))
                        / 2.0
                    )
                assert b19._float64_bits(applied[name]) == b19._float64_bits(expected)
                if update not in (0, steps_per_epoch - 1, stepper.total_steps - 1):
                    assert b19._float64_bits(applied[name]) == b19._float64_bits(retired)
                elif b19._float64_bits(applied[name]) != b19._float64_bits(retired):
                    endpoint_differences.add((steps_per_epoch, peak, update))
                if update >= steps_per_epoch:
                    assert math.isfinite(applied[name])
                    assert b19.MIN_LR <= applied[name] < peak
                    assert applied[name] <= previous_decay[name]
                    previous_decay[name] = applied[name]

        evidence = stepper.evidence(require_complete=True)
        assert evidence["completed_steps"] == b19.EPOCHS * steps_per_epoch
        assert evidence["total_steps"] == b19.EPOCHS * steps_per_epoch
        assert evidence["steps_per_epoch"] == steps_per_epoch
        for name, peak in group_peaks.items():
            assert b19._float64_bits(evidence["first_applied_lrs"][name]) == b19._float64_bits(0.0)
            assert b19._float64_bits(evidence["warmup_last_applied_lrs"][name]) == b19._float64_bits(peak)
            assert b19._float64_bits(evidence["final_applied_lrs"][name]) == b19._float64_bits(b19.MIN_LR)

    assert checked_schedules == {
        (steps_per_epoch, peak)
        for steps_per_epoch in fold_updates
        for peak in (b19.BACKBONE_LR, b19.TASK_LR)
    }
    assert endpoint_differences == {
        (202, b19.BACKBONE_LR, 201),
        (212, b19.BACKBONE_LR, 211),
        (207, b19.BACKBONE_LR, 206),
    }

    stepper.checkpoints["first"]["backbone_decay"] = -0.0
    assert b19._float64_bits(-0.0) != b19._float64_bits(0.0)
    with pytest.raises(b19.B19ContractError, match="applied LR checkpoints changed"):
        stepper.evidence(require_complete=True)
    stepper.checkpoints["first"]["backbone_decay"] = 0.0
    stepper.checkpoints["final"]["task_decay"] = math.nextafter(b19.MIN_LR, math.inf)
    with pytest.raises(b19.B19ContractError, match="applied LR checkpoints changed"):
        stepper.evidence(require_complete=True)


def test_metric_barrier_binds_all_states_and_rejects_tamper(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    token, records = _barrier_token(tmp_path, monkeypatch)
    b19._assert_metric_token(token)
    state = Path(records[0].path)
    state.write_bytes(state.read_bytes() + b"tamper")
    with pytest.raises(b19.B19ContractError, match="barrier"):
        b19._assert_metric_token(token)


def test_forged_or_replayed_barrier_cannot_mint_metric_capability(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    token, _records = _barrier_token(tmp_path, monkeypatch)
    barrier = Path(token.path)
    assert b19.sha256_file(barrier) == token.sha256
    with pytest.raises(b19.B19ContractError, match="private seal proof"):
        b19.validate_metric_barrier(barrier, token.sha256)


def test_metric_token_requires_its_matching_active_seal_proof(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    token, _records = _barrier_token(tmp_path, monkeypatch)
    key = b19._metric_token_key(token.sha256, token.nonce)
    b19._ACTIVE_BARRIER_SEALS.pop(key)
    with pytest.raises(b19.B19ContractError, match="unchanged validated B19 barrier"):
        b19._assert_metric_token(token)
    assert key not in b19._VALID_BARRIER_TOKENS


@pytest.mark.parametrize("binding", ["git", "source"])
def test_metric_token_rejects_and_revokes_post_seal_binding_drift(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, binding: str
) -> None:
    token, _records = _barrier_token(tmp_path, monkeypatch)
    key = b19._metric_token_key(token.sha256, token.nonce)
    if binding == "git":
        monkeypatch.setattr(
            b19,
            "_git_contract",
            lambda: {
                "head": "d" * 40,
                "branch": b19.EXPECTED_BRANCH,
                "status": "",
                "clean": True,
                "head_is_commit": True,
            },
        )
    else:
        monkeypatch.setattr(b19, "_source_hashes", lambda: {"runner": "e" * 64})
    with pytest.raises(b19.B19ContractError, match="git/source bindings changed"):
        b19._assert_metric_token(token)
    assert key not in b19._VALID_BARRIER_TOKENS
    assert key not in b19._ACTIVE_BARRIER_SEALS


def _summary(
    class1_f1: float,
    macro_f1: float,
    *,
    recall: float = 0.75,
    restricted: int = 20,
    two_to_one: int = 20,
    one_to_two: int = 5,
    pair2: float = 0.80,
    mean_pair: float = 0.80,
    fold_f1: float = 0.70,
) -> dict[str, object]:
    return {
        "class1_f1": class1_f1,
        "macro_f1": macro_f1,
        "class1_recall": recall,
        "restricted_0_2_4_to_1": restricted,
        "transition_2_to_1": two_to_one,
        "transition_1_to_2": one_to_two,
        "pairs": {"2": {"auroc": pair2}},
        "mean_pair_auroc": mean_pair,
        "folds": [{"fold": fold, "class1_f1": fold_f1} for fold in range(b19.FOLDS)],
    }


def _passing_gate_inputs() -> tuple[dict[str, dict[str, object]], dict[str, object]]:
    summaries = {
        "stock_relation": _summary(0.710, 0.820, restricted=18, two_to_one=18),
        "surfacefold_mean_control": _summary(0.710, 0.820, restricted=20, two_to_one=20, one_to_two=6),
        "surfacefold_spatial_candidate": _summary(
            0.730,
            0.821,
            restricted=17,
            two_to_one=17,
            pair2=0.810,
            mean_pair=0.805,
            fold_f1=0.73,
        ),
        "candidate_delta_off": _summary(0.720, 0.820, restricted=18, two_to_one=18),
    }
    intervals = {}
    for comparison in (
        "candidate_vs_control",
        "candidate_vs_stock",
        "candidate_vs_delta_off",
    ):
        intervals[comparison] = {
            metric: {"lower": -0.001 if metric == "macro_f1_delta" else 0.001}
            for metric in (
                "macro_f1_delta",
                "class1_f1_delta",
                "pair2_auroc_delta",
                "mean_pair_auroc_delta",
            )
        }
    return summaries, {"intervals": intervals}


def test_gate_passes_exact_thresholds_and_zero_denominator_fails(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setattr(b19, "_assert_metric_token", lambda _token: None)
    token = b19.MetricBarrierToken("unused", "a" * 64, "nonce")
    summaries, bootstrap = _passing_gate_inputs()
    passed = b19.assess_b19_gate(summaries, bootstrap, token, integrity_complete=True)
    assert passed["passed"] is True
    changed = copy.deepcopy(summaries)
    changed["surfacefold_mean_control"]["restricted_0_2_4_to_1"] = 0
    changed["surfacefold_mean_control"]["transition_2_to_1"] = 0
    failed = b19.assess_b19_gate(changed, bootstrap, token, integrity_complete=True)
    assert failed["passed"] is False
    assert failed["denominator_valid"] == {
        "restricted_reduction_vs_control": False,
        "2_to_1_reduction_vs_control": False,
    }


def test_summary_and_component_bootstrap_after_barrier(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    token, _records = _barrier_token(tmp_path, monkeypatch)
    labels, folds, groups = [], [], []
    for fold in range(b19.FOLDS):
        for group_in_fold in range(2):
            group = fold * 2 + group_in_fold
            for label in range(b19.CLASSES):
                labels.append(label)
                folds.append(fold)
                groups.append(group)
    labels_array = np.asarray(labels, dtype=np.int64)
    logits = np.full((len(labels), b19.CLASSES), -1.0, dtype=np.float64)
    logits[np.arange(len(labels)), labels_array] = 2.0
    scores = {name: logits.copy() for name in b19.OOF_NAMES}
    summary = b19.classification_summary(labels_array, logits, token, np.asarray(folds))
    assert summary["accuracy"] == 1.0
    bootstrap = b19.paired_component_bootstrap(
        labels_array,
        np.asarray(folds),
        np.asarray(groups),
        scores,
        token,
        replicates=3,
        seed=7,
        enforce_locked_draw_hash=False,
    )
    assert bootstrap["replicates"] == 3
    assert all(np.isfinite(values).all() for metrics in bootstrap["values"].values() for values in metrics.values())
