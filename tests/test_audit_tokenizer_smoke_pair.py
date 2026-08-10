from __future__ import annotations

from trkh.tools.audit_tokenizer_smoke_pair import (
    _validate_provenance,
    parse_args,
)


def _resolved(stem: str) -> dict[str, object]:
    return {
        "model_config": {
            "stem_architecture": stem,
            "token_pruning": True,
        },
        "train_config": {
            "seed": 42,
            "batch_size": 32,
            "grad_accum_steps": 2,
            "epochs": 2,
            "scheduler_total_epochs": 15,
            "warmup_epochs": 1,
            "max_train_batches": 120,
            "max_val_batches": 0,
            "attention_view_loss_weight": 0.0,
            "attention_crop_probability": 0.0,
            "attention_drop_probability": 0.0,
        },
    }


def test_generic_pair_parser_accepts_locked_starnet_profile() -> None:
    args = parse_args(
        [
            "--control-predictions",
            "control.csv",
            "--candidate-predictions",
            "candidate.csv",
            "--control-run-summary",
            "control.json",
            "--candidate-run-summary",
            "candidate.json",
            "--control-resolved-config",
            "control_config.json",
            "--candidate-resolved-config",
            "candidate_config.json",
            "--stage-a-summary",
            "stage_a.json",
            "--locked-protocol",
            "protocol.json",
            "--output-dir",
            "out",
            "--expected-method",
            "starnet_s2_local_multiplicative_tokenizer",
            "--candidate-stem",
            "starnet_s2_tokenizer",
        ]
    )

    assert args.expected_method == "starnet_s2_local_multiplicative_tokenizer"
    assert args.candidate_stem == "starnet_s2_tokenizer"


def test_generic_pair_provenance_accepts_only_matching_profile() -> None:
    method = "starnet_s2_local_multiplicative_tokenizer"
    _validate_provenance(
        control_run={"test_summary": None},
        candidate_run={"test_summary": None},
        control_config=_resolved("conv_pool"),
        candidate_config=_resolved("starnet_s2_tokenizer"),
        stage_a={
            "method": method,
            "gate": {"smoke_permission": True},
            "sources": {"validation_loaded": False, "test_loaded": False},
        },
        protocol={"method": method, "test_allowed": False},
        expected_method=method,
        candidate_stem="starnet_s2_tokenizer",
    )

    try:
        _validate_provenance(
            control_run={"test_summary": None},
            candidate_run={"test_summary": None},
            control_config=_resolved("conv_pool"),
            candidate_config=_resolved("inceptionnext_atto_tokenizer"),
            stage_a={
                "method": method,
                "gate": {"smoke_permission": True},
                "sources": {"validation_loaded": False, "test_loaded": False},
            },
            protocol={"method": method, "test_allowed": False},
            expected_method=method,
            candidate_stem="starnet_s2_tokenizer",
        )
    except ValueError as error:
        assert "Unexpected stem architecture" in str(error)
    else:
        raise AssertionError("Expected mismatched candidate stem rejection")
