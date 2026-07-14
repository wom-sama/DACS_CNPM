param(
    [string]$Python = "D:\DataAI\.venv\Scripts\python.exe",
    [string]$PairOutputDir = "runs\audit_inceptionnext_atto_matched_smoke_pair_20260715",
    [string]$ControlRunDir = "runs\smoke_inceptionnext_atto_control_120b_2e_20260715",
    [string]$CandidateRunDir = "runs\smoke_inceptionnext_atto_candidate_120b_2e_20260715",
    [switch]$PreflightOnly
)

$ErrorActionPreference = "Stop"
$CommonLauncher = Join-Path $PSScriptRoot "run_trkh_tokenizer_postsmoke_audit.ps1"
& $CommonLauncher `
    -Python $Python `
    -PairOutputDir $PairOutputDir `
    -ControlRunDir $ControlRunDir `
    -CandidateRunDir $CandidateRunDir `
    -StageASummary "runs\audit_inceptionnext_atto_stage_a_20260715\summary.json" `
    -LockedStageASha256 "8b1b4e6d1485bfa4e5224c432a769f7ccdb91ef5a15a345a4aa9a9f6b6ff41a5" `
    -LockedPairSummarySha256 "163d4d1c2ba522712a1e7f16f9d89d5b9b82114e0760781ccb8f08694ba0d8bb" `
    -LockedControlCheckpointSha256 "9ce2e6fdcf506be6066ec67ece2e37c2f6edb69dd7f403f193fc6e5044791bb1" `
    -LockedCandidateCheckpointSha256 "cfe1050ac98549f5194fbb6cd6a0c2d42f34169d25e6881eabfa6dd20ca7d101" `
    -LockedCandidateTraceSha256 "51387302d477dafb3e89f5d1a69cd9862796254e2457f2288c4d4b5c6b2e99aa" `
    -ExpectedMethod "inceptionnext_atto_surface_tokenizer" `
    -ExpectedStem "inceptionnext_atto_tokenizer" `
    -ExpectedStemChannels 160 `
    -TraceMode "inceptionnext_atto_architecture_trace_audit" `
    -CohortMode "validation_only_inceptionnext_atto_changed_case_cohort" `
    -PostsmokeMode "inceptionnext_atto_postsmoke_closure" `
    -PreflightOnly:$PreflightOnly
