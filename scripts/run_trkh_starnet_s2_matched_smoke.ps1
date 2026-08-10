param(
    [string]$Python = "D:\DataAI\.venv\Scripts\python.exe",
    [string]$SourceLauncherArgs = "runs\full_v8_yolof_randominit_30e_20260714_105524\launcher_args.json",
    [string]$ControlRunName = "smoke_starnet_s2_control_120b_2e_20260715",
    [string]$CandidateRunName = "smoke_starnet_s2_candidate_120b_2e_20260715",
    [string]$PairOutputDir = "runs\audit_starnet_s2_matched_smoke_pair_20260715",
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
    -StageASummary "runs\audit_starnet_s2_stage_a_20260715\summary.json" `
    -LockedStageASha256 "7fdb01920a5d8c03324194efd35d2fd394e516c7e89ca9f4df3233f5624ddcaa" `
    -ExpectedMethod "starnet_s2_local_multiplicative_tokenizer" `
    -CandidateStemArchitecture "starnet_s2_tokenizer" `
    -EvaluationFamily "TRKH-StarNet-S2-smoke" `
    -CandidatePaperName "TRKH-StarNet-S2-120b-2e" `
    -PreflightOnly:$PreflightOnly
