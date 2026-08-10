param(
    [string]$Python = "D:\DataAI\.venv\Scripts\python.exe",
    [string]$SourceLauncherArgs = "runs\full_v8_yolof_randominit_30e_20260714_105524\launcher_args.json",
    [string]$ControlRunName = "smoke_inceptionnext_atto_control_120b_2e_20260715",
    [string]$CandidateRunName = "smoke_inceptionnext_atto_candidate_120b_2e_20260715",
    [string]$PairOutputDir = "runs\audit_inceptionnext_atto_matched_smoke_pair_20260715",
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
    -StageASummary "runs\audit_inceptionnext_atto_stage_a_20260715\summary.json" `
    -LockedStageASha256 "8b1b4e6d1485bfa4e5224c432a769f7ccdb91ef5a15a345a4aa9a9f6b6ff41a5" `
    -ExpectedMethod "inceptionnext_atto_surface_tokenizer" `
    -CandidateStemArchitecture "inceptionnext_atto_tokenizer" `
    -EvaluationFamily "TRKH-InceptionNeXt-Atto-smoke" `
    -CandidatePaperName "TRKH-InceptionNeXt-Atto-120b-2e" `
    -PreflightOnly:$PreflightOnly
