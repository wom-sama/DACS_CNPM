import argparse

from trkh.tools import run_b41_backbone_routed_pair_mean_fold3 as b41


def test_b41_objective_isolates_auxiliary_head() -> None:
    objective = b41._objective_factory((100, 25, 75, 100, 100))
    assert objective.isolate_auxiliary_head is True


def test_b41_contract_locks_backbone_routing_and_scale() -> None:
    contract = b41._objective_contract()
    assert contract["outer_weight"] == 0.20
    assert contract["auxiliary_head_parameter_gradient"] == "exact_zero"
    assert contract["clean_task_head_gradient"] == "unchanged"
    assert contract["gradient_projection_or_surgery"] is False


def test_b41_gate_uses_own_follow_up_language(monkeypatch) -> None:
    monkeypatch.setattr(
        b41.b38,
        "_gate",
        lambda _control, _candidate: {
            "passed": False,
            "next_permission": "stale_value",
        },
    )
    gate = b41._gate({}, {})
    assert gate["next_permission"] == "close_exact_backbone_routed_pair_mean"


def test_b41_preflight_locks_head_isolation_and_gradient_ranges(
    monkeypatch,
    tmp_path,
) -> None:
    observed = {}

    def fake_preflight(*_args, **kwargs):
        observed.update(kwargs)
        return {"passed": True}

    monkeypatch.setattr(b41.b40, "run_pair_mean_preflight", fake_preflight)
    result = b41._run_preflight(
        argparse.Namespace(),
        tmp_path,
        repo=tmp_path,
        git={"clean": True},
    )

    assert result == {"passed": True}
    assert observed["head_isolated_expected"] is True
    assert observed["auxiliary_weight"] == 0.20
    assert (observed["task_ratio_min"], observed["task_ratio_max"]) == (0.04, 0.10)
    assert (observed["focus_ratio_min"], observed["focus_ratio_max"]) == (0.02, 0.10)
    assert observed["pair_cosine_min"] == 0.0
