from __future__ import annotations

from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]


def _read(relative_path: str) -> str:
    return (ROOT / relative_path).read_text(encoding="utf-8")


def test_full_pipeline_avoids_native_stderr_pipeline_and_orders_final_test() -> None:
    script = _read("scripts/run_trkh_current_best_full_pipeline.ps1")

    assert "2>&1 |" not in script
    assert "ForEach-Object" not in script
    assert "$LASTEXITCODE" in script
    assert "SkipFinalTest = $true" in script
    assert "MinimumRawValMacroF1" in script
    assert "MinimumRawValClass1F1" in script
    assert "MinimumTrainSelectionMetric" in script
    assert "best_selection_metric_value" in script
    assert '[ValidateSet("Scratch", "WarmStart", "StatefulResume")]' in script
    assert "ResumeMode = $TrainingMode" in script
    assert "TrainingMode=Scratch refuses a non-empty -ResumeCheckpoint" in script
    assert '$PSBoundParameters.ContainsKey("ResumeCheckpoint")' in script
    assert "SchedulerTotalEpochs = $SchedulerTotalEpochsEffective" in script
    assert "LearningRate = $LearningRate" in script
    assert "AttentionViewStartEpoch = $AttentionViewStartEpochEffective" in script
    assert 'elseif ($TrainingMode -eq "WarmStart")' in script
    assert "DisableBalancedEpochSampling = [bool]$DisableBalancedEpochSampling" in script
    assert "StatefulResume requires the existing run directory" in script
    assert "Scratch/WarmStart requires a new RunName" in script
    assert "StatefulResume refuses a normally completed run" in script
    assert "validation_promotion_gate.json" in script
    assert "if ($RunFinalTest -and $ValidationGatePassed)" in script
    assert script.index('Invoke-Evaluation -Split "val"') < script.index('if ($RunFinalTest -and $ValidationGatePassed)')
    assert "trkh.evaluation.robustness_eval" in script
    assert "trkh.evaluation.xai_audit" in script
    assert "trkh.tools.audit_prediction_forensics" in script
    assert "trkh.tools.audit_class_confusions" in script
    assert "trkh.tools.build_boundary_review_manifest" in script
    assert "trkh.tools.audit_trkh_artifact_retention" in script


def test_promotion_gate_requires_expected_fair_metric_semantics() -> None:
    script = _read("scripts/run_trkh_current_best_full_pipeline.ps1")

    assert (
        '$ExpectedTrainSelectionMetricName = '
        '"fair_macro_f1_min_class_gap_penalty"'
    ) in script
    assert (
        "[string]$TrainSummary.best_selection_metric_name -ceq "
        "$ExpectedTrainSelectionMetricName"
    ) in script
    assert "$TrainSummary.best_selection_metric_higher_is_better -is [bool]" in script
    assert "$TrainSelectionMetricNameMatches -and" in script
    assert "$TrainSelectionMetricDirectionMatches -and" in script
    assert "train_selection_metric_name = $ExpectedTrainSelectionMetricName" in script
    assert "train_selection_metric_higher_is_better = $true" in script
    assert "name_matches_expected = [bool]$TrainSelectionMetricNameMatches" in script
    assert (
        "direction_matches_expected = [bool]$TrainSelectionMetricDirectionMatches"
        in script
    )


def test_export_and_video_wrappers_are_single_native_command_paths() -> None:
    export_script = _read("scripts/run_trkh_export_engine.ps1")
    video_script = _read("scripts/run_trkh_test_video.ps1")

    for script in (export_script, video_script):
        assert "2>&1 |" not in script
        assert "ForEach-Object" not in script
        assert "$LASTEXITCODE" in script
        assert "Invoke-NativeChecked" in script
        assert "Get-LatestPromotedCheckpoint" in script
        assert "name_matches_expected" in script
        assert "direction_matches_expected" in script
        assert "stale or was promoted by a legacy validation gate" in script

    assert "trkh.inference.deploy" in export_script
    assert "model_fp32_fp16.engine" in export_script
    assert "trkh.inference.stream_infer_trt" in video_script
    assert "trkh.inference.stream_infer" in video_script


def test_tensorrt_stream_supports_classification_only_outputs() -> None:
    trt_stream = _read("trkh/inference/stream_infer_trt.py")

    assert "TensorRT stream moi yeu cau engine detection co output boxes" not in trt_stream
    assert "build_classification_prediction_result" in trt_stream
    assert 'if "boxes" not in outputs:' in trt_stream
    assert 'elif last_prediction_result.get("predictions"):' in trt_stream


def test_command_file_uses_absolute_one_line_wrappers() -> None:
    commands = _read("docs/TRKH_CURRENT_BEST_FULL_TRAIN_COMMANDS_20260706.txt")
    command_lines = [
        line
        for line in commands.splitlines()
        if line.startswith("powershell.exe -NoProfile -ExecutionPolicy Bypass -File")
    ]

    expected_wrapper_counts = {
        "run_trkh_current_best_full_pipeline.ps1": 2,
        "run_trkh_precision_ensemble_pipeline.ps1": 2,
        "run_trkh_export_engine.ps1": 2,
        "run_trkh_test_video.ps1": 3,
    }

    assert len(command_lines) == sum(expected_wrapper_counts.values())
    for wrapper, expected_count in expected_wrapper_counts.items():
        assert sum(wrapper in line for line in command_lines) == expected_count
    assert all("`" not in line for line in command_lines)
    assert all("D:\\DataAI\\AIEx\\TRKH\\scripts\\" in line for line in command_lines)
    assert all("2>&1" not in line for line in command_lines)
