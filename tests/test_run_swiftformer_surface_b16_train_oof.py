from __future__ import annotations

import hashlib
import json
from pathlib import Path

import numpy as np
import pytest
import timm
import torch
from PIL import Image

from trkh.models.swiftformer_surface_b16 import SwiftFormerSurfaceB16
from trkh.tools import run_swiftformer_surface_b16_train_oof as b16


@pytest.fixture(autouse=True)
def _isolated_metric_barrier_state(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    monkeypatch.setattr(b16, "_METRIC_BACKEND_LOADED", False)
    canonical_runs = (tmp_path / "accepted_preflight_runs").resolve()
    canonical_runs.mkdir()
    monkeypatch.setattr(
        b16, "_canonical_preflight_runs_root", lambda: canonical_runs
    )
    b16._VALID_BARRIER_TOKENS.clear()
    b16._VERIFIED_BARRIER_STATE_TOKENS.clear()


def _mock_preflight() -> dict[str, object]:
    graph = {
        "qdq": {"format": "QDQ", "quantize_linear": 1, "dequantize_linear": 1},
        "coverage": {"ratio": 0.95},
    }
    return {
        "protocol_id": b16.PROTOCOL_ID,
        "passed": True,
        "permissions": {
            "dataset_yaml_read": False,
            "dataset_path_enumerated": False,
            "train_image_read": False,
            "real_label_read": False,
            "prior_prediction_read": False,
            "validation_constructed": False,
            "test_constructed": False,
            "network_used": False,
            "formal_train_permission": False,
        },
        "deployment": {
            "passed": True,
            "int8": {"stock": graph, "control": graph, "candidate": graph},
            "latency": {"passed": True},
        },
        "cuda_fit": {"passed": True},
    }


def _accepted_preflight(tmp_path: Path) -> dict[str, object]:
    payload = _mock_preflight()
    root = (
        b16._canonical_preflight_runs_root()
        / f"{b16.ACCEPTED_PREFLIGHT_OUTPUT_PREFIX}unit"
    )
    root.mkdir()
    artifact = (root / "preflight.json").resolve()
    digest = b16.atomic_json(artifact, payload)
    (root / "preflight.sha256").write_bytes(f"{digest}\n".encode("ascii"))
    return {"artifact": str(artifact), "sha256": digest, "payload": payload}


def _make_barrier(tmp_path: Path) -> b16.MetricBarrierToken:
    states = tmp_path / "states"
    states.mkdir()
    records = []
    evidence_sha = "a" * 64
    for fold in range(b16.FOLDS):
        for arm in b16.ARMS:
            path = states / f"fold_{fold}_{arm}.safetensors"
            path.write_bytes(f"{fold}:{arm}".encode())
            records.append(
                b16.FinalStateRecord(
                    state_id=f"fold_{fold}:{arm}",
                    fold=fold,
                    arm=arm,
                    path=str(path.resolve()),
                    sha256=b16.sha256_file(path),
                    bytes=path.stat().st_size,
                    parameter_count=b16.ARM_PARAMETERS[arm],
                    mode=b16._state_mode(arm),
                    exposure_sha256=evidence_sha,
                    loss_curve_sha256=evidence_sha,
                )
            )
    return b16.seal_metric_barrier(
        tmp_path,
        records,
        accepted_preflight=_accepted_preflight(tmp_path),
        counters={
            "held_image_open_count": 0,
            "held_forward_count": 0,
            "oof_buffer_count": 0,
            "logit_write_count": 0,
            "metric_call_count": 0,
        },
    )


def test_protocol_constants_are_exact() -> None:
    assert b16.PROTOCOL_SHA256 == "4184395eb2c599d5bbd8b0cfa0a44cd20a73864b2b509f8602b0c73573c99cde"
    assert b16.STUDENT_SHA256 == "c0dfb2e99069ece07b691ff470f5dafed98716d90be738369828693c3bf2071c"
    assert b16.DINO_SHA256 == "2a1ec16ae28ffa07bc0ead0241ee7df9fc26451fe6f9f839b7b3afa0a906b040"
    assert b16.EXPECTED_FOLD_VECTOR_SHA256 == "fca50be670fac7da20dd23cb382f9096c3c071d1a1e0dbaa1fd123e65c8aeb4b"
    assert b16.EXPECTED_GROUP_VECTOR_SHA256 == "1243b61c91497459fa12036d1b9f63770ba4f1f11bb132ca17fb3b6013f1ade2"
    assert b16.EXPECTED_BOOTSTRAP_DRAW_SHA256 == "d8967cbc13b17b78dd225841bd82a1a3074a7f57c3728fc6dd600aba646ec469"
    assert b16._hex40("a" * 40) is True
    assert b16._hex40("a" * 64) is False
    assert b16._hex64("a" * 64) is True
    assert b16._hex64(10**63) is False
    assert b16._git_contract()["head_is_commit"] is True


def test_assignment_hashes_and_component_isolation_are_pure() -> None:
    paths = [f"train/c{i % 5}/row_{i}.jpg" for i in range(10)]
    labels = np.asarray([i % 5 for i in range(10)], dtype=np.int64)
    folds = np.asarray([i // 5 for i in range(10)], dtype=np.int64)
    # Expand to five class-complete folds without reading any CSV/data.
    paths = [f"train/c{label}/f{fold}.jpg" for fold in range(5) for label in range(5)]
    labels = np.asarray([label for _fold in range(5) for label in range(5)], dtype=np.int64)
    folds = np.asarray([fold for fold in range(5) for _label in range(5)], dtype=np.int64)
    groups = np.arange(25, dtype=np.int64)
    result = b16.validate_assignment_arrays(paths, labels, groups, folds, enforce_locked=False)
    assert set(result["hashes"]) == {
        "fold_vector_sha256",
        "path_fold_sha256",
        "group_vector_sha256",
    }
    crossing = groups.copy()
    crossing[5] = crossing[0]
    with pytest.raises(ValueError, match="component-disjoint"):
        b16.validate_assignment_arrays(paths, labels, crossing, folds, enforce_locked=False)


def test_random_crop_and_shared_views_are_deterministic_and_shared() -> None:
    values = np.zeros((40, 50, 3), dtype=np.uint8)
    values[..., 0] = np.arange(50, dtype=np.uint8)[None, :]
    values[..., 1] = np.arange(40, dtype=np.uint8)[:, None]
    image = Image.fromarray(values)
    first = torch.Generator().manual_seed(123)
    second = torch.Generator().manual_seed(123)
    crop_a = b16.sample_random_resized_crop(40, 50, first)
    crop_b = b16.sample_random_resized_crop(40, 50, second)
    assert crop_a == crop_b
    student, teacher = b16.build_shared_train_views(image, crop_a)
    assert student.shape == (3, 224, 224)
    assert teacher.shape == (3, 256, 256)
    assert torch.isfinite(student).all() and torch.isfinite(teacher).all()
    assert 0 <= crop_a.top <= 40 - crop_a.height
    assert 0 <= crop_a.left <= 50 - crop_a.width


class _FakeDino(torch.nn.Module):
    def __init__(self) -> None:
        super().__init__()
        self.calls = 0

    def forward_features(self, images: torch.Tensor) -> torch.Tensor:
        self.calls += 1
        batch = images.size(0)
        values = torch.arange(batch * 261 * 384, dtype=torch.float32, device=images.device)
        return values.reshape(batch, 261, 384).div_(100_000.0)


def test_teacher_is_called_once_and_target_is_shared_by_two_losses() -> None:
    teacher = _FakeDino().eval()
    ledger = b16.TeacherCallLedger()
    images = torch.zeros(2, 3, 256, 256)
    target = b16.teacher_relation_target(teacher, images, ledger)
    first = torch.randn(2, 112, 14, 14, requires_grad=True)
    second = torch.randn(2, 112, 14, 14, requires_grad=True)
    loss_a = b16.relation_loss_from_target(first, target)
    loss_b = b16.relation_loss_from_target(second, target)
    (loss_a + loss_b).backward()
    assert teacher.calls == ledger.calls == 1
    assert ledger.rows == 2
    assert target.shape == (2, 364)
    assert target.requires_grad is False
    assert first.grad is not None and second.grad is not None


def test_stock_is_bare_and_candidate_control_states_match() -> None:
    torch.manual_seed(7)
    base = timm.create_model(
        "swiftformer_xs",
        pretrained=False,
        num_classes=5,
        drop_rate=0.0,
        drop_path_rate=0.0,
    )
    base.distilled_training = False
    arms = b16.build_b16_arm_models(base)
    assert not isinstance(arms["stock"], SwiftFormerSurfaceB16)
    assert isinstance(arms["globalized_control"], SwiftFormerSurfaceB16)
    assert isinstance(arms["swiftsurface_candidate"], SwiftFormerSurfaceB16)
    assert sum(parameter.numel() for parameter in arms["stock"].parameters()) == b16.STOCK_PARAMETERS
    control = arms["globalized_control"].state_dict()
    candidate = arms["swiftsurface_candidate"].state_dict()
    assert control.keys() == candidate.keys()
    assert all(torch.equal(control[name], candidate[name]) for name in control)


def test_branch_template_matches_preflight_default_generator_seed(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    from trkh.tools.audit_swiftformer_surface_b16_preflight import build_b16_arms

    base = timm.create_model("swiftformer_xs", pretrained=False, num_classes=5)

    def forbidden_manual_seed(*_args: object, **_kwargs: object) -> None:
        raise AssertionError("build path must use torch.random.default_generator")

    monkeypatch.setattr(torch, "manual_seed", forbidden_manual_seed)
    formal = b16.build_b16_arm_models(base)
    preflight = build_b16_arms(base, seed=b16.SEED)
    for formal_name, reference in (
        ("stock", preflight.stock),
        ("globalized_control", preflight.control),
        ("swiftsurface_candidate", preflight.candidate),
    ):
        observed = formal[formal_name].state_dict()
        expected = reference.state_dict()
        assert observed.keys() == expected.keys()
        assert all(torch.equal(observed[name], expected[name]) for name in observed)


def test_candidate_residual_telemetry_is_finite_on_synthetic_input() -> None:
    base = timm.create_model("swiftformer_xs", pretrained=False, num_classes=5)
    candidate = b16.build_b16_arm_models(base)["swiftsurface_candidate"].eval()
    with torch.inference_mode():
        logits, ratio = b16._candidate_forward_with_residual_telemetry(
            candidate, torch.randn(2, 3, 64, 64)
        )
    assert logits.shape == (2, 5)
    assert ratio.shape == (2,)
    assert torch.isfinite(ratio).all()
    assert torch.all(ratio >= 0)


def test_all_arm_forwards_preserve_rng_with_stochasticity_disabled() -> None:
    base = timm.create_model(
        "swiftformer_xs",
        pretrained=False,
        num_classes=5,
        drop_rate=0.0,
        drop_path_rate=0.0,
    )
    images = torch.randn(2, 3, 64, 64)
    for model in b16.build_b16_arm_models(base).values():
        model.train()
        before = b16._rng_snapshot(torch.device("cpu"))
        with torch.no_grad():
            model(images)
        after = b16._rng_snapshot(torch.device("cpu"))
        assert b16._rng_snapshot_equal(before, after)


def test_head_reset_preserves_rng_and_uses_locked_initialization() -> None:
    from trkh.tools.audit_swiftformer_surface_b16_preflight import (
        reset_swiftformer_five_class_heads,
    )

    model = timm.create_model("swiftformer_xs", pretrained=False, num_classes=1000)
    reference = timm.create_model("swiftformer_xs", pretrained=False, num_classes=1000)
    reference.load_state_dict(model.state_dict(), strict=True)
    before = torch.random.get_rng_state().clone()
    formal = b16.reset_locked_five_class_heads(model)
    after = torch.random.get_rng_state()
    preflight = reset_swiftformer_five_class_heads(reference, seed=b16.SEED)
    assert torch.equal(before, after)
    assert formal["state_sha256"] == preflight["state_sha256"]
    assert all(
        torch.equal(model.state_dict()[name], reference.state_dict()[name])
        for name in model.state_dict()
    )
    assert model.head.out_features == model.head_dist.out_features == 5
    assert torch.equal(model.head.bias, torch.zeros_like(model.head.bias))
    assert torch.equal(model.head_dist.bias, torch.zeros_like(model.head_dist.bias))
    assert 0.01 < float(model.head.weight.std()) < 0.03


def test_optimizer_groups_cover_exactly_once_and_split_lr_decay() -> None:
    base = timm.create_model("swiftformer_xs", pretrained=False, num_classes=5)
    candidate = b16.build_b16_arm_models(base)["swiftsurface_candidate"]
    groups = b16.build_discriminative_adamw_groups(candidate)
    parameters = [parameter for group in groups for parameter in group["params"]]
    assert len({id(parameter) for parameter in parameters}) == len(parameters)
    assert {id(parameter) for parameter in parameters} == {
        id(parameter) for parameter in candidate.parameters() if parameter.requires_grad
    }
    assert {float(group["peak_lr"]) for group in groups} == {b16.BACKBONE_LR, b16.TASK_LR}
    assert {float(group["weight_decay"]) for group in groups} == {0.0, b16.WEIGHT_DECAY}
    layer_scale_ids = {
        id(parameter)
        for name, parameter in candidate.named_parameters()
        if "layer_scale" in name
    }
    zero_decay_ids = {
        id(parameter)
        for group in groups
        if float(group["weight_decay"]) == 0.0
        for parameter in group["params"]
    }
    assert layer_scale_ids <= zero_decay_ids


def test_locked_optimizer_stepper_applies_and_records_exact_lr_horizon() -> None:
    parameter = torch.nn.Parameter(torch.ones(()))
    applied_by_optimizer: list[float] = []

    class RecordingAdamW(torch.optim.AdamW):
        def step(self, closure=None):  # type: ignore[no-untyped-def]
            applied_by_optimizer.append(float(self.param_groups[0]["lr"]))
            return super().step(closure)

    optimizer = RecordingAdamW(
        [
            {
                "params": [parameter],
                "lr": b16.TASK_LR,
                "peak_lr": b16.TASK_LR,
                "name": "task_decay",
            }
        ]
    )
    stepper = b16.LockedOptimizerStepper(optimizer, steps_per_epoch=3)
    assert optimizer.param_groups[0]["lr"] == 0.0
    for _ in range(stepper.total_steps):
        parameter.grad = torch.ones_like(parameter)
        stepper.step()
    evidence = stepper.evidence(require_complete=True)
    assert applied_by_optimizer[0] == 0.0
    assert applied_by_optimizer[stepper.steps_per_epoch - 1] == pytest.approx(
        b16.TASK_LR
    )
    assert applied_by_optimizer[-1] == pytest.approx(b16.MIN_LR)
    assert evidence["completed_steps"] == evidence["total_steps"] == 24
    assert evidence["first_applied_lrs"] == {"task_decay": 0.0}
    assert evidence["warmup_last_applied_lrs"] == {
        "task_decay": b16.TASK_LR
    }
    assert evidence["final_applied_lrs"] == {"task_decay": b16.MIN_LR}
    with pytest.raises(b16.B16ContractError, match="beyond"):
        stepper.step()


def test_locked_optimizer_stepper_rejects_impossible_one_update_warmup() -> None:
    parameter = torch.nn.Parameter(torch.ones(()))
    optimizer = torch.optim.AdamW(
        [{"params": [parameter], "peak_lr": b16.TASK_LR}]
    )
    with pytest.raises(ValueError, match="at least two"):
        b16.LockedOptimizerStepper(optimizer, steps_per_epoch=1)


def test_gradient_audit_requires_every_trainable_gradient_and_finite_values() -> None:
    model = torch.nn.Sequential(torch.nn.Linear(3, 4), torch.nn.Linear(4, 2))
    model(torch.ones(2, 3)).sum().backward()
    result = b16.audit_and_clip_gradients(model)
    assert result["parameter_tensors"] == 4
    assert result["parameter_elements"] == sum(parameter.numel() for parameter in model.parameters())

    missing = torch.nn.Sequential(torch.nn.Linear(3, 4), torch.nn.Linear(4, 2))
    missing[0](torch.ones(2, 3)).sum().backward()
    with pytest.raises(b16.B16ContractError, match="lack gradients"):
        b16.audit_and_clip_gradients(missing)

    nonfinite = torch.nn.Linear(3, 2)
    nonfinite(torch.ones(1, 3)).sum().backward()
    nonfinite.weight.grad[0, 0] = float("nan")
    with pytest.raises(b16.B16ContractError, match="non-finite"):
        b16.audit_and_clip_gradients(nonfinite)


def test_qdq_preflight_validation_is_fail_closed() -> None:
    payload = _mock_preflight()
    b16.validate_qdq_preflight_payload(payload)
    bad = json.loads(json.dumps(payload))
    bad["deployment"]["int8"]["candidate"]["coverage"]["ratio"] = 0.89
    with pytest.raises(b16.B16ContractError, match="QDQ/coverage"):
        b16.validate_qdq_preflight_payload(bad)
    bool_count = json.loads(json.dumps(payload))
    bool_count["deployment"]["int8"]["candidate"]["qdq"]["quantize_linear"] = True
    with pytest.raises(b16.B16ContractError, match="QDQ/coverage"):
        b16.validate_qdq_preflight_payload(bool_count)
    string_ratio = json.loads(json.dumps(payload))
    string_ratio["deployment"]["int8"]["candidate"]["coverage"]["ratio"] = "0.95"
    with pytest.raises(b16.B16ContractError, match="QDQ/coverage"):
        b16.validate_qdq_preflight_payload(string_ratio)


def test_barrier_rejects_missing_state_and_premetric_counter(tmp_path: Path) -> None:
    states = tmp_path / "states"
    states.mkdir()
    path = states / "one.safetensors"
    path.write_bytes(b"one")
    record = b16.FinalStateRecord(
        state_id="fold_0:stock",
        fold=0,
        arm="stock",
        path=str(path.resolve()),
        sha256=b16.sha256_file(path),
        bytes=3,
        parameter_count=b16.STOCK_PARAMETERS,
        mode="synthetic",
        exposure_sha256="a" * 64,
        loss_curve_sha256="a" * 64,
    )
    with pytest.raises(b16.B16ContractError, match="exactly 15"):
        b16.seal_metric_barrier(
            tmp_path,
            [record],
            accepted_preflight=_accepted_preflight(tmp_path),
            counters={
                "held_image_open_count": 0,
                "held_forward_count": 0,
                "oof_buffer_count": 0,
                "logit_write_count": 0,
                "metric_call_count": 0,
            },
        )


def test_exact_15_state_barrier_precedes_lazy_metric_backend(tmp_path: Path) -> None:
    assert b16.metric_backend_loaded() is False
    token = _make_barrier(tmp_path)
    assert b16.metric_backend_loaded() is False
    labels = np.asarray([0, 1, 2, 3, 4] * 2, dtype=np.int64)
    logits = np.full((labels.size, 5), -2.0, dtype=np.float64)
    logits[np.arange(labels.size), labels] = 2.0
    result = b16.classification_summary(labels, logits, token)
    assert result["accuracy"] == 1.0
    assert b16.metric_backend_loaded() is True


def test_barrier_detects_state_tamper(tmp_path: Path) -> None:
    token = _make_barrier(tmp_path)
    payload = json.loads(Path(token.path).read_text(encoding="utf-8"))
    state = Path(payload["states"][0]["path"])
    state.write_bytes(state.read_bytes() + b"tamper")
    with pytest.raises(b16.B16ContractError, match="changed after validation"):
        b16.classification_summary(
            np.asarray([0, 1, 2, 3, 4]),
            np.eye(5),
            token,
        )


def test_barrier_rejects_type_coercion_even_with_recomputed_hash(tmp_path: Path) -> None:
    token = _make_barrier(tmp_path)
    payload = json.loads(Path(token.path).read_text(encoding="utf-8"))
    payload["counters"]["held_image_open_count"] = False
    coerced_counter = tmp_path / "coerced_counter.json"
    counter_sha = b16.atomic_json(coerced_counter, payload)
    with pytest.raises(b16.B16ContractError, match="counters"):
        b16.validate_metric_barrier(coerced_counter, counter_sha)

    payload = json.loads(Path(token.path).read_text(encoding="utf-8"))
    payload["states"][0]["fold"] = "0"
    coerced_fold = tmp_path / "coerced_fold.json"
    fold_sha = b16.atomic_json(coerced_fold, payload)
    with pytest.raises(b16.B16ContractError, match="field types"):
        b16.validate_metric_barrier(coerced_fold, fold_sha)


def test_barrier_rejects_phase_checks_and_preflight_schema_tampering(
    tmp_path: Path,
) -> None:
    token = _make_barrier(tmp_path)
    original = json.loads(Path(token.path).read_text(encoding="utf-8"))
    mutations = []

    changed_phase = json.loads(json.dumps(original))
    changed_phase["phase"] = "after_metrics"
    mutations.append(("phase.json", changed_phase, "schema/identity"))

    extra_check = json.loads(json.dumps(original))
    extra_check["checks"]["unexpected"] = True
    mutations.append(("extra_check.json", extra_check, "check schema/values"))

    false_check = json.loads(json.dumps(original))
    false_check["checks"]["metrics_absent"] = False
    mutations.append(("false_check.json", false_check, "check schema/values"))

    extra_preflight_key = json.loads(json.dumps(original))
    extra_preflight_key["accepted_preflight"]["payload"] = _mock_preflight()
    mutations.append(
        (
            "extra_preflight.json",
            extra_preflight_key,
            "accepted-preflight schema",
        )
    )

    relative_preflight = json.loads(json.dumps(original))
    relative_preflight["accepted_preflight"]["artifact"] = "preflight.json"
    mutations.append(
        ("relative_preflight.json", relative_preflight, "not absolute")
    )

    wrong_preflight_hash = json.loads(json.dumps(original))
    wrong_preflight_hash["accepted_preflight"]["sha256"] = "0" * 64
    mutations.append(
        ("wrong_preflight_hash.json", wrong_preflight_hash, "SHA-256 mismatch")
    )

    for filename, payload, message in mutations:
        path = tmp_path / filename
        digest = b16.atomic_json(path, payload)
        with pytest.raises(b16.B16ContractError, match=message):
            b16.validate_metric_barrier(path, digest)


def test_barrier_binds_existing_canonical_preflight_bytes(tmp_path: Path) -> None:
    token = _make_barrier(tmp_path)
    payload = json.loads(Path(token.path).read_text(encoding="utf-8"))
    preflight = Path(payload["accepted_preflight"]["artifact"])
    preflight.write_bytes(preflight.read_bytes() + b" ")
    with pytest.raises(b16.B16ContractError, match="preflight SHA-256 mismatch"):
        b16.validate_metric_barrier(Path(token.path), token.sha256)
    with pytest.raises(b16.B16ContractError, match="preflight SHA-256 mismatch"):
        b16.classification_summary(np.arange(5), np.eye(5), token)


def test_barrier_rejects_state_renamed_below_current_states_root(
    tmp_path: Path,
) -> None:
    token = _make_barrier(tmp_path)
    payload = json.loads(Path(token.path).read_text(encoding="utf-8"))
    row = payload["states"][0]
    source = Path(row["path"])
    nested = source.parent / "nested"
    nested.mkdir()
    renamed = nested / source.name
    source.rename(renamed)
    row["path"] = str(renamed.resolve())
    renamed_barrier = tmp_path / "renamed_state_barrier.json"
    digest = b16.atomic_json(renamed_barrier, payload)
    with pytest.raises(b16.B16ContractError, match="filename/current run root"):
        b16.validate_metric_barrier(renamed_barrier, digest)


def _bootstrap_inputs() -> tuple[np.ndarray, np.ndarray, np.ndarray, dict[str, np.ndarray]]:
    labels = np.asarray([label for fold in range(5) for label in range(5) for _ in range(2)], dtype=np.int64)
    folds = np.asarray([fold for fold in range(5) for _label in range(5) for _ in range(2)], dtype=np.int64)
    groups = np.arange(labels.size, dtype=np.int64)
    base = np.full((labels.size, 5), -1.0, dtype=np.float64)
    base[np.arange(labels.size), labels] = 1.0
    candidate = base.copy()
    control = base.copy()
    stock = base.copy()
    off = base.copy()
    return labels, folds, groups, {
        "stock": stock,
        "globalized_control": control,
        "swiftsurface_candidate": candidate,
        "candidate_r0": off,
    }


def test_component_bootstrap_is_deterministic_and_stores_finite_replicates(tmp_path: Path) -> None:
    token = _make_barrier(tmp_path)
    labels, folds, groups, logits = _bootstrap_inputs()
    first = b16.paired_component_bootstrap(
        labels, folds, groups, logits, token, replicates=12, seed=77, enforce_locked_draw_hash=False
    )
    second = b16.paired_component_bootstrap(
        labels, folds, groups, logits, token, replicates=12, seed=77, enforce_locked_draw_hash=False
    )
    assert first["draws_int64_sha256"] == second["draws_int64_sha256"]
    assert first["quantile_method"] == "linear"
    for comparison in first["values"].values():
        for values in comparison.values():
            assert values.shape == (12,)
            assert np.isfinite(values).all()


def test_gate_requires_zero_denominator_to_fail(tmp_path: Path) -> None:
    token = _make_barrier(tmp_path)
    fold_rows = [
        {"fold": fold, "class1_f1": 0.8}
        for fold in range(5)
    ]
    def summary(c1: float, macro: float) -> dict[str, object]:
        return {
            "class1_f1": c1,
            "macro_f1": macro,
            "class1_recall": 0.8,
            "mean_pair_auroc": 0.9,
            "pairs": {"2": {"auroc": 0.9}},
            "restricted_0_2_4_to_1": 0,
            "transition_2_to_1": 0,
            "transition_1_to_2": 1,
            "folds": fold_rows,
        }
    summaries = {
        "stock": summary(0.70, 0.82),
        "globalized_control": summary(0.70, 0.82),
        "swiftsurface_candidate": summary(0.75, 0.83),
        "candidate_r0": summary(0.70, 0.82),
    }
    interval = {"lower": 0.01, "upper": 0.09}
    bootstrap = {
        "intervals": {
            name: {
                "class1_f1_delta": interval,
                "macro_f1_delta": interval,
                "pair2_auroc_delta": interval,
                "mean_pair_auroc_delta": interval,
            }
            for name in ("candidate_vs_control", "candidate_vs_stock", "candidate_vs_ablation")
        }
    }
    gate = b16.assess_b16_gate(summaries, bootstrap, token, integrity_complete=True)
    assert gate["checks"]["recall_and_transition_control"] is False
    assert gate["passed"] is False
    assert gate["validation_protocol_permission"] is False
    assert gate["validation_execution_permission"] is False
    assert gate["test_permission"] is False
    assert gate["denominator_valid"] == {
        "restricted_reduction_vs_control": False,
        "2_to_1_reduction_vs_control": False,
    }
    assert gate["point_deltas"]["restricted_reduction_vs_control"] is None
    assert gate["point_deltas"]["2_to_1_reduction_vs_control"] is None
    # Both the gate and any summary embedding it remain strict-JSON serializable.
    encoded = b16._json_bytes({"gate": gate, "summary": {"gate": gate}})
    assert b"NaN" not in encoded
    assert json.loads(encoded)["gate"]["point_deltas"][
        "restricted_reduction_vs_control"
    ] is None


def test_atomic_json_rejects_nonfinite_and_overwrite(tmp_path: Path) -> None:
    path = tmp_path / "artifact.json"
    digest = b16.atomic_json(path, {"ok": True})
    assert digest == hashlib.sha256(path.read_bytes()).hexdigest()
    with pytest.raises(FileExistsError):
        b16.atomic_json(path, {"ok": True})
    with pytest.raises(ValueError):
        b16.atomic_json(tmp_path / "nan.json", {"value": float("nan")})
