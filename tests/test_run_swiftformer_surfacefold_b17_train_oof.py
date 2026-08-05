from __future__ import annotations

import copy
from argparse import Namespace
from pathlib import Path

import numpy as np
import pytest
import timm
import torch
from PIL import Image

from trkh.tools import run_swiftformer_surfacefold_b17_train_oof as b17


def _base_model() -> torch.nn.Module:
    model = timm.create_model(
        b17.STUDENT_TIMM_ID,
        pretrained=False,
        num_classes=1000,
        drop_rate=0.0,
        drop_path_rate=0.0,
    )
    b17.reset_locked_five_class_heads(model)
    return model


@pytest.fixture(autouse=True)
def _reset_metric_globals() -> None:
    b17._METRIC_BACKEND_LOADED = False
    b17._VALID_BARRIER_TOKENS.clear()
    b17._PENDING_BARRIER_SEALS.clear()
    b17._ACTIVE_BARRIER_SEALS.clear()


def _preflight_payload(git: dict[str, object], sources: dict[str, str]) -> dict[str, object]:
    payload = {
        "schema_version": 1,
        "protocol_id": b17.PROTOCOL_ID,
        "protocol_sha256": b17.PROTOCOL_SHA256,
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
    monkeypatch.setattr(b17, "_repository_root", lambda: tmp_path)
    directory = tmp_path / "runs" / "preflight_b17_surfacefold_xs_unit"
    directory.mkdir(parents=True)
    payload = _preflight_payload(git, sources)
    artifact = directory / "preflight.json"
    artifact.write_bytes(b17._json_bytes(payload))
    digest = b17.sha256_file(artifact)
    (directory / "preflight.sha256").write_bytes(f"{digest}\n".encode("ascii"))
    return {"artifact": str(artifact.resolve()), "sha256": digest, "payload": payload}


def _barrier_token(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> tuple[b17.MetricBarrierToken, list[b17.FinalStateRecord]]:
    git = {
        "head": "a" * 40,
        "branch": b17.EXPECTED_BRANCH,
        "status": "",
        "clean": True,
        "head_is_commit": True,
    }
    sources = {"runner": "b" * 64}
    accepted = _published_preflight(tmp_path, monkeypatch, git, sources)
    monkeypatch.setattr(b17, "_git_contract", lambda: dict(git))
    monkeypatch.setattr(b17, "_source_hashes", lambda: dict(sources))
    monkeypatch.setattr(b17, "ARM_PARAMETERS", {arm: 1 for arm in b17.ARMS})
    monkeypatch.setattr(b17, "ARM_TRAINABLE_PARAMETERS", {arm: 1 for arm in b17.ARMS})
    output = tmp_path / "formal"
    (output / "states").mkdir(parents=True)
    args = Namespace(
        data=tmp_path / "never_open_data.yaml",
        assignment_csv=tmp_path / "never_open_assignment.csv",
        student_weight=tmp_path / "never_open_student.safetensors",
        dino_weight=tmp_path / "never_open_dino.safetensors",
    )
    claim = b17.claim_train_authorization_once(
        accepted_preflight=accepted,
        git=git,
        sources=sources,
        output=output,
        args=args,
    )
    records: list[b17.FinalStateRecord] = []
    for fold in range(b17.FOLDS):
        fold_dir = output / "folds" / f"fold_{fold}"
        fold_dir.mkdir(parents=True)
        exposure = fold_dir / "exposure.npz"
        losses = fold_dir / "loss_curves.npz"
        np.savez(exposure, x=np.asarray([fold]))
        np.savez(losses, x=np.asarray([fold]))
        for arm in b17.ARMS:
            path = (output / "states" / f"fold_{fold}_{arm}.safetensors").resolve()
            metadata = b17._metadata_for_state(fold, arm)
            digest = b17.atomic_safetensors(path, {"weight": torch.tensor([float(fold)])}, metadata)
            records.append(
                b17.FinalStateRecord(
                    state_id=f"fold_{fold}:{arm}",
                    fold=fold,
                    arm=arm,
                    path=str(path),
                    sha256=digest,
                    bytes=path.stat().st_size,
                    parameter_count=1,
                    trainable_parameter_count=1,
                    mode=b17.ARM_MODES[arm],
                    metadata_sha256=b17._metadata_sha256(metadata),
                    exposure_sha256=b17.sha256_file(exposure),
                    loss_curve_sha256=b17.sha256_file(losses),
                    frozen_projection_sha256="c" * 64,
                    folded_max_abs_error=0.0,
                )
            )
    reloaded: list[str] = []
    cross_mode = []
    token = b17.seal_metric_barrier(
        output,
        records,
        accepted_preflight=accepted,
        authorization_claim=claim,
        args=args,
        git=git,
        sources=sources,
        counters=dict(b17.ZERO_BARRIER_COUNTERS),
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
    first = b17.claim_train_authorization_once(
        accepted_preflight=accepted,
        git=git,
        sources=sources,
        output=output,
        args=args,
    )
    assert (
        b17.validate_train_authorization_claim(
            first,
            accepted_preflight=accepted,
            git=git,
            sources=sources,
            output=output,
            args=args,
        )
        == first
    )
    with pytest.raises(b17.B17ContractError, match="already consumed"):
        b17.claim_train_authorization_once(
            accepted_preflight=accepted,
            git=git,
            sources=sources,
            output=tmp_path / "runs" / "second",
            args=args,
        )


def test_reused_authorization_stops_run_before_any_data_access(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    output = tmp_path / "formal"
    git = {
        "head": "a" * 40,
        "branch": b17.EXPECTED_BRANCH,
        "status": "",
        "clean": True,
        "head_is_commit": True,
    }
    monkeypatch.setattr(b17, "_git_contract", lambda: git)
    monkeypatch.setattr(b17, "_source_hashes", lambda: {"formal_runner": "b" * 64})
    monkeypatch.setattr(
        b17,
        "validate_accepted_preflight",
        lambda *_args, **_kwargs: {
            "artifact": str(tmp_path / "preflight.json"),
            "sha256": "c" * 64,
            "payload": {},
        },
    )
    monkeypatch.setattr(b17, "_validated_output", lambda _path: output)
    monkeypatch.setattr(
        b17,
        "claim_train_authorization_once",
        lambda **_kwargs: (_ for _ in ()).throw(b17.B17ContractError("already consumed")),
    )
    data_accessed = []
    monkeypatch.setattr(
        b17,
        "_read_locked_data_root",
        lambda _path: data_accessed.append(True),
    )
    args = Namespace(
        confirm_protocol_id=b17.PROTOCOL_ID,
        data=tmp_path / "must_not_open.yaml",
        assignment_csv=tmp_path / "must_not_open.csv",
        student_weight=tmp_path / "student.safetensors",
        dino_weight=tmp_path / "dino.safetensors",
        preflight_artifact=tmp_path / "preflight.json",
        preflight_sha256="c" * 64,
        output_dir=output,
    )
    with pytest.raises(b17.B17ContractError, match="already consumed"):
        b17.run_formal(args)
    assert data_accessed == []
    assert (output / "failure.json").is_file()


def test_runner_has_no_b16_runtime_import() -> None:
    source = Path(b17.__file__).read_text(encoding="utf-8")
    assert "swiftformer_surface_b16" not in source
    assert "run_swiftformer_surface_b16_train_oof" not in source
    assert "audit_swiftformer_surface_b16_preflight" not in source


def test_preflight_schema_is_exact_and_train_only() -> None:
    git = {"head": "a" * 40}
    sources = {"runner": "b" * 64}
    payload = _preflight_payload(git, sources)
    assert payload["isolation"] == {
        "dataset_attempts": [],
        "network_attempts": [],
        "process_attempts": [],
    }
    b17._validate_preflight_payload(payload, git=git, sources=sources)
    for mutation in (
        lambda row: row["permissions"].__setitem__("validation_constructed", True),
        lambda row: row["authorization"].__setitem__("deployment_status", "READY"),
        lambda row: row.__setitem__("unexpected", True),
        lambda row: row["checks"].__setitem__("stock_deployment_first", False),
        lambda row: row["isolation"]["process_attempts"].append("python child.py"),
    ):
        changed = copy.deepcopy(payload)
        mutation(changed)
        with pytest.raises(b17.B17ContractError, match="preflight"):
            b17._validate_preflight_payload(changed, git=git, sources=sources)


def test_assignment_requires_component_disjoint_class_complete_folds() -> None:
    paths, labels, groups, folds = [], [], [], []
    for fold in range(b17.FOLDS):
        for label in range(b17.CLASSES):
            paths.append(f"train/{label}/{fold}.jpg")
            labels.append(label)
            groups.append(fold * 10 + label)
            folds.append(fold)
    contract = b17.validate_assignment_arrays(
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
        b17.validate_assignment_arrays(
            paths,
            np.asarray(labels),
            leaked,
            np.asarray(folds),
            enforce_locked=False,
        )


def test_augmentation_is_private_deterministic_and_shared() -> None:
    image = Image.new("RGB", (257, 193), (30, 90, 180))
    first = torch.Generator().manual_seed(b17.AUGMENTATION_SEED_BASE)
    second = torch.Generator().manual_seed(b17.AUGMENTATION_SEED_BASE)
    crop1 = b17.sample_random_resized_crop(193, 257, first)
    crop2 = b17.sample_random_resized_crop(193, 257, second)
    assert crop1 == crop2
    student, teacher = b17.build_shared_train_views(image, crop1)
    assert student.shape == (3, 224, 224)
    assert teacher.shape == (3, 256, 256)
    assert torch.isfinite(student).all() and torch.isfinite(teacher).all()


def test_head_reset_and_arm_construction_are_rng_safe_and_exact() -> None:
    torch.manual_seed(77)
    model = timm.create_model(b17.STUDENT_TIMM_ID, pretrained=False, num_classes=1000)
    caller = torch.random.get_rng_state().clone()
    b17.reset_locked_five_class_heads(model)
    assert torch.equal(caller, torch.random.get_rng_state())
    duplicate = copy.deepcopy(model)
    b17.reset_locked_five_class_heads(duplicate)
    assert torch.equal(model.head.weight, duplicate.head.weight)
    arms = b17.build_b17_arm_models(model)
    assert {
        name: sum(parameter.numel() for parameter in arm.parameters()) for name, arm in arms.items()
    } == b17.ARM_PARAMETERS
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
    candidate = b17.build_b17_arm_models(_base_model())["surfacefold_spatial_candidate"]
    groups = b17.build_discriminative_adamw_groups(candidate)
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
    assert all(group["peak_lr"] == b17.TASK_LR for group in task_groups)


def test_teacher_target_and_all_arm_relation_loss_are_exact() -> None:
    class Teacher(torch.nn.Module):
        def forward_features(self, images: torch.Tensor) -> torch.Tensor:
            values = torch.linspace(
                -1.0,
                1.0,
                images.size(0) * 261 * b17.DINO_FEATURE_DIM,
                dtype=torch.float32,
            )
            return values.reshape(images.size(0), 261, b17.DINO_FEATURE_DIM)

    ledger = b17.TeacherCallLedger()
    target = b17.teacher_relation_target(Teacher(), torch.zeros(2, 3, 256, 256), ledger)
    assert target.shape == (2, 84)
    assert not target.requires_grad
    assert (ledger.calls, ledger.rows) == (1, 2)
    for _arm in b17.ARMS:
        s3 = torch.randn(2, 220, 7, 7, requires_grad=True)
        logits = torch.randn(2, b17.CLASSES, requires_grad=True)
        labels = torch.tensor([0, 1])
        weights = torch.ones(b17.CLASSES)
        total, ce, relation = b17.exact_b17_objective(logits, s3, labels, weights, target)
        assert torch.equal(total, ce + b17.RELATION_WEIGHT * relation)
        total.backward()
        assert torch.isfinite(total)
        assert s3.grad is not None and torch.isfinite(s3.grad).all()
        assert logits.grad is not None and torch.isfinite(logits.grad).all()


def test_locked_lr_is_applied_on_optimizer_update() -> None:
    parameter = torch.nn.Parameter(torch.ones(()))
    optimizer = torch.optim.AdamW(
        [
            {
                "params": [parameter],
                "name": "task_decay",
                "peak_lr": b17.TASK_LR,
                "lr": b17.TASK_LR,
            }
        ]
    )
    stepper = b17.LockedOptimizerStepper(optimizer, steps_per_epoch=3)
    observed = []
    for _ in range(stepper.total_steps):
        parameter.grad = torch.ones_like(parameter)
        observed.append(stepper.step()["task_decay"])
    evidence = stepper.evidence(require_complete=True)
    assert observed[0] == 0.0
    assert observed[2] == b17.TASK_LR
    assert observed[-1] == pytest.approx(b17.MIN_LR, abs=1e-15)
    assert evidence["completed_steps"] == 8 * 3


def test_metric_barrier_binds_all_states_and_rejects_tamper(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    token, records = _barrier_token(tmp_path, monkeypatch)
    b17._assert_metric_token(token)
    state = Path(records[0].path)
    state.write_bytes(state.read_bytes() + b"tamper")
    with pytest.raises(b17.B17ContractError, match="barrier"):
        b17._assert_metric_token(token)


def test_forged_or_replayed_barrier_cannot_mint_metric_capability(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    token, _records = _barrier_token(tmp_path, monkeypatch)
    barrier = Path(token.path)
    assert b17.sha256_file(barrier) == token.sha256
    with pytest.raises(b17.B17ContractError, match="private seal proof"):
        b17.validate_metric_barrier(barrier, token.sha256)


def test_metric_token_requires_its_matching_active_seal_proof(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    token, _records = _barrier_token(tmp_path, monkeypatch)
    key = b17._metric_token_key(token.sha256, token.nonce)
    b17._ACTIVE_BARRIER_SEALS.pop(key)
    with pytest.raises(b17.B17ContractError, match="unchanged validated B17 barrier"):
        b17._assert_metric_token(token)
    assert key not in b17._VALID_BARRIER_TOKENS


@pytest.mark.parametrize("binding", ["git", "source"])
def test_metric_token_rejects_and_revokes_post_seal_binding_drift(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, binding: str
) -> None:
    token, _records = _barrier_token(tmp_path, monkeypatch)
    key = b17._metric_token_key(token.sha256, token.nonce)
    if binding == "git":
        monkeypatch.setattr(
            b17,
            "_git_contract",
            lambda: {
                "head": "d" * 40,
                "branch": b17.EXPECTED_BRANCH,
                "status": "",
                "clean": True,
                "head_is_commit": True,
            },
        )
    else:
        monkeypatch.setattr(b17, "_source_hashes", lambda: {"runner": "e" * 64})
    with pytest.raises(b17.B17ContractError, match="git/source bindings changed"):
        b17._assert_metric_token(token)
    assert key not in b17._VALID_BARRIER_TOKENS
    assert key not in b17._ACTIVE_BARRIER_SEALS


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
        "folds": [{"fold": fold, "class1_f1": fold_f1} for fold in range(b17.FOLDS)],
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
    monkeypatch.setattr(b17, "_assert_metric_token", lambda _token: None)
    token = b17.MetricBarrierToken("unused", "a" * 64, "nonce")
    summaries, bootstrap = _passing_gate_inputs()
    passed = b17.assess_b17_gate(summaries, bootstrap, token, integrity_complete=True)
    assert passed["passed"] is True
    changed = copy.deepcopy(summaries)
    changed["surfacefold_mean_control"]["restricted_0_2_4_to_1"] = 0
    changed["surfacefold_mean_control"]["transition_2_to_1"] = 0
    failed = b17.assess_b17_gate(changed, bootstrap, token, integrity_complete=True)
    assert failed["passed"] is False
    assert failed["denominator_valid"] == {
        "restricted_reduction_vs_control": False,
        "2_to_1_reduction_vs_control": False,
    }


def test_summary_and_component_bootstrap_after_barrier(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    token, _records = _barrier_token(tmp_path, monkeypatch)
    labels, folds, groups = [], [], []
    for fold in range(b17.FOLDS):
        for group_in_fold in range(2):
            group = fold * 2 + group_in_fold
            for label in range(b17.CLASSES):
                labels.append(label)
                folds.append(fold)
                groups.append(group)
    labels_array = np.asarray(labels, dtype=np.int64)
    logits = np.full((len(labels), b17.CLASSES), -1.0, dtype=np.float64)
    logits[np.arange(len(labels)), labels_array] = 2.0
    scores = {name: logits.copy() for name in b17.OOF_NAMES}
    summary = b17.classification_summary(labels_array, logits, token, np.asarray(folds))
    assert summary["accuracy"] == 1.0
    bootstrap = b17.paired_component_bootstrap(
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
