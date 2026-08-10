param(
    [string]$Python = "D:\DataAI\.venv\Scripts\python.exe",
    [string]$PairOutputDir = "runs\audit_octave_conv_matched_smoke_pair_20260715",
    [string]$ControlRunDir = "runs\smoke_octave_conv_control_120b_2e_20260715",
    [string]$CandidateRunDir = "runs\smoke_octave_conv_candidate_120b_2e_20260715",
    [switch]$PreflightOnly
)

$ErrorActionPreference = "Stop"
$CommonLauncher = Join-Path $PSScriptRoot "run_trkh_tokenizer_postsmoke_audit.ps1"
& $CommonLauncher `
    -Python $Python `
    -PairOutputDir $PairOutputDir `
    -ControlRunDir $ControlRunDir `
    -CandidateRunDir $CandidateRunDir `
    -StageASummary "runs\audit_octave_conv_stem_stage_a_20260715\summary.json" `
    -LockedStageASha256 "b27335e71b8e228c065aa26cbc0015e89521f2bd03c039f39dc8d35be04e159b" `
    -LockedPairSummarySha256 "db74f347b2654fedfd6dbb8a16e3b808605cf0252cd751ab9524461d18a6347b" `
    -LockedControlCheckpointSha256 "3f4cad1c3115e585b1e22b93b17df54f05ad92711ba8dbf1f4027efc333d9525" `
    -LockedCandidateCheckpointSha256 "f674dde7bbbb0bcaf85f128f7412da4b4e6baa3bfafdf38e1fb2cf24886fa254" `
    -LockedCandidateTraceSha256 "1e7477e56fdfbb7d6605da772e6858efbd7381287c90b8102b5eceb643081413" `
    -ExpectedMethod "octave_conv_parameter_matched_frequency_stem" `
    -ExpectedStem "octave_conv" `
    -ExpectedStemChannels 256 `
    -ExpectedStemSpatialSize 32 `
    -TraceMode "octave_conv_architecture_trace_audit" `
    -CohortMode "validation_only_octave_conv_changed_case_cohort" `
    -PostsmokeMode "octave_conv_postsmoke_closure" `
    -PreflightOnly:$PreflightOnly
