param(
    [string]$Python = "D:\DataAI\.venv\Scripts\python.exe",
    [string]$SourceLauncherArgs = "runs\full_v8_yolof_randominit_30e_20260714_105524\launcher_args.json",
    [string]$ControlRunName = "smoke_octave_conv_control_120b_2e_20260715",
    [string]$CandidateRunName = "smoke_octave_conv_candidate_120b_2e_20260715",
    [string]$PairOutputDir = "runs\audit_octave_conv_matched_smoke_pair_20260715",
    [switch]$PreflightOnly
)

$ErrorActionPreference = "Stop"
$CommonLauncher = Join-Path $PSScriptRoot "run_trkh_tokenizer_matched_smoke.ps1"
& $CommonLauncher `
    -Python $Python `
    -SourceLauncherArgs $SourceLauncherArgs `
    -ControlRunName $ControlRunName `
    -CandidateRunName $CandidateRunName `
    -PairOutputDir $PairOutputDir `
    -StageASummary "runs\audit_octave_conv_stem_stage_a_20260715\summary.json" `
    -LockedStageASha256 "b27335e71b8e228c065aa26cbc0015e89521f2bd03c039f39dc8d35be04e159b" `
    -ExpectedMethod "octave_conv_parameter_matched_frequency_stem" `
    -CandidateStemArchitecture "octave_conv" `
    -EvaluationFamily "TRKH-OctConv-smoke" `
    -CandidatePaperName "TRKH-OctConv-alpha0125-120b-2e" `
    -SmokeGateProfile "octave_conv_precision" `
    -PreflightOnly:$PreflightOnly
