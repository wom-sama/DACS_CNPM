param(
    [string]$Python = "D:\DataAI\.venv\Scripts\python.exe",
    [string]$PairOutputDir = "runs\audit_starnet_s2_matched_smoke_pair_20260715",
    [string]$ControlRunDir = "runs\smoke_starnet_s2_control_120b_2e_20260715",
    [string]$CandidateRunDir = "runs\smoke_starnet_s2_candidate_120b_2e_20260715",
    [switch]$PreflightOnly
)

$ErrorActionPreference = "Stop"
$CommonLauncher = Join-Path $PSScriptRoot "run_trkh_tokenizer_postsmoke_audit.ps1"
& $CommonLauncher `
    -Python $Python `
    -PairOutputDir $PairOutputDir `
    -ControlRunDir $ControlRunDir `
    -CandidateRunDir $CandidateRunDir `
    -StageASummary "runs\audit_starnet_s2_stage_a_20260715\summary.json" `
    -LockedStageASha256 "7fdb01920a5d8c03324194efd35d2fd394e516c7e89ca9f4df3233f5624ddcaa" `
    -LockedPairSummarySha256 "8642e6c1ae69c3d241fef719f67e00a868bb5b6d79e298b4794294b915f49f80" `
    -LockedControlCheckpointSha256 "d0c1522a32c495fc30ed1a06db742a0252c3119d37a9fade76cd3a190f3d6913" `
    -LockedCandidateCheckpointSha256 "008c2f4f5cddb4c85f8fb8b0ddfab9e6614fbfc7528367c45eee5f359d951376" `
    -LockedCandidateTraceSha256 "7519d15bd73614677f5fa172a9777fcb408337b04e443c8f7df9d691bfafb898" `
    -ExpectedMethod "starnet_s2_local_multiplicative_tokenizer" `
    -ExpectedStem "starnet_s2_tokenizer" `
    -ExpectedStemChannels 128 `
    -TraceMode "starnet_s2_architecture_trace_audit" `
    -CohortMode "validation_only_starnet_s2_changed_case_cohort" `
    -PostsmokeMode "starnet_s2_postsmoke_closure" `
    -PreflightOnly:$PreflightOnly
