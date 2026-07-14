param(
    [string]$Python = "D:\DataAI\.venv\Scripts\python.exe",
    [string]$PairOutputDir = "runs\audit_inceptionnext_atto_matched_smoke_pair_20260715",
    [string]$ControlRunDir = "runs\smoke_inceptionnext_atto_control_120b_2e_20260715",
    [string]$CandidateRunDir = "runs\smoke_inceptionnext_atto_candidate_120b_2e_20260715",
    [switch]$PreflightOnly
)

$ErrorActionPreference = "Stop"
$RepoRoot = (Resolve-Path (Join-Path $PSScriptRoot "..")).Path
Set-Location $RepoRoot

$DataYaml = "D:\DataAI\AIEx\newdataset\yolo_f\data.yaml"
$StageASummary = "runs\audit_inceptionnext_atto_stage_a_20260715\summary.json"
$LockedDataSha256 = "716e33df24c63a9e9920f97b685199707fb84ab4c7154544f5dd9a3e00d884ef"
$LockedStageASha256 = "8b1b4e6d1485bfa4e5224c432a769f7ccdb91ef5a15a345a4aa9a9f6b6ff41a5"
$LockedPairSummarySha256 = "163d4d1c2ba522712a1e7f16f9d89d5b9b82114e0760781ccb8f08694ba0d8bb"
$LockedControlCheckpointSha256 = "9ce2e6fdcf506be6066ec67ece2e37c2f6edb69dd7f403f193fc6e5044791bb1"
$LockedCandidateCheckpointSha256 = "cfe1050ac98549f5194fbb6cd6a0c2d42f34169d25e6881eabfa6dd20ca7d101"
$LockedCandidateTraceSha256 = "51387302d477dafb3e89f5d1a69cd9862796254e2457f2288c4d4b5c6b2e99aa"

function Assert-FileSha256 {
    param([string]$Path, [string]$Expected)
    if (-not (Test-Path -LiteralPath $Path -PathType Leaf)) {
        throw "Required file not found: $Path"
    }
    $observed = (Get-FileHash -Algorithm SHA256 -LiteralPath $Path).Hash.ToLowerInvariant()
    if ($observed -ne $Expected.ToLowerInvariant()) {
        throw "SHA-256 mismatch for $Path`: $observed != $Expected"
    }
    return $observed
}

function Assert-NewOutput {
    param([string]$Path)
    if (Test-Path -LiteralPath $Path) {
        throw "Refusing to mix or overwrite post-smoke evidence: $Path"
    }
}

function Invoke-NativePython {
    param([string]$Step, [string[]]$Arguments)
    Write-Host "[$Step] $Python $($Arguments -join ' ')"
    $previousErrorActionPreference = $ErrorActionPreference
    $ErrorActionPreference = "Continue"
    try {
        & $Python @Arguments
        $exitCode = $LASTEXITCODE
    }
    finally {
        $ErrorActionPreference = $previousErrorActionPreference
    }
    if ($exitCode -ne 0) {
        throw "$Step failed with exit code $exitCode."
    }
}

function Save-Status {
    param([hashtable]$Status, [string]$Path)
    $Status | ConvertTo-Json -Depth 4 | Set-Content -LiteralPath $Path -Encoding UTF8
}

if (-not (Test-Path -LiteralPath $Python -PathType Leaf)) {
    throw "Python executable not found: $Python"
}
if (-not (Test-Path -LiteralPath $PairOutputDir -PathType Container)) {
    throw "Matched-smoke pair directory not found: $PairOutputDir"
}

$PairSummary = Join-Path $PairOutputDir "comparison\summary.json"
$ChangedCases = Join-Path $PairOutputDir "comparison\changed_cases.csv"
$ControlPredictions = Join-Path $PairOutputDir "control_val\predictions_detailed.csv"
$CandidatePredictions = Join-Path $PairOutputDir "candidate_val\predictions_detailed.csv"
$ControlCheckpoint = Join-Path $ControlRunDir "checkpoints\best.pt"
$CandidateCheckpoint = Join-Path $CandidateRunDir "checkpoints\best.pt"
$CandidateTrace = Join-Path $CandidateRunDir "architecture_trace\trace_summary.json"

$dataSha256 = Assert-FileSha256 $DataYaml $LockedDataSha256
$stageASha256 = Assert-FileSha256 $StageASummary $LockedStageASha256
$pairSummarySha256 = Assert-FileSha256 $PairSummary $LockedPairSummarySha256
$controlCheckpointSha256 = Assert-FileSha256 $ControlCheckpoint $LockedControlCheckpointSha256
$candidateCheckpointSha256 = Assert-FileSha256 $CandidateCheckpoint $LockedCandidateCheckpointSha256
$candidateTraceSha256 = Assert-FileSha256 $CandidateTrace $LockedCandidateTraceSha256
foreach ($requiredFile in @($ChangedCases, $ControlPredictions, $CandidatePredictions)) {
    if (-not (Test-Path -LiteralPath $requiredFile -PathType Leaf)) {
        throw "Required post-smoke input not found: $requiredFile"
    }
}

$pair = Get-Content -Raw -LiteralPath $PairSummary | ConvertFrom-Json
if ($pair.method -ne "inceptionnext_atto_surface_tokenizer") {
    throw "Matched-smoke method is not InceptionNeXt-Atto."
}
if ([bool]$pair.sources.test_used) {
    throw "Matched-smoke evidence used test data."
}
if ([int]$pair.sources.validation_rows -ne 2606 -or -not [bool]$pair.gate.checks.full_validation_support) {
    throw "Matched-smoke validation support is not the locked 2606 rows."
}

$TraceAuditDir = Join-Path $PairOutputDir "architecture_trace_candidate_audit"
$ForensicsControlDir = Join-Path $PairOutputDir "forensics_control"
$ForensicsCandidateDir = Join-Path $PairOutputDir "forensics_candidate"
$ConfusionsControlDir = Join-Path $PairOutputDir "confusions_control"
$ConfusionsCandidateDir = Join-Path $PairOutputDir "confusions_candidate"
$XaiCohortDir = Join-Path $PairOutputDir "xai_cohort"
$XaiControlDir = Join-Path $PairOutputDir "xai_control"
$XaiCandidateDir = Join-Path $PairOutputDir "xai_candidate"
$PairedXaiDir = Join-Path $PairOutputDir "paired_xai"
$RobustnessControlDir = Join-Path $PairOutputDir "robustness_control"
$RobustnessCandidateDir = Join-Path $PairOutputDir "robustness_candidate"
$PostsmokeAuditDir = Join-Path $PairOutputDir "postsmoke_audit"
$ProtocolPath = Join-Path $PairOutputDir "postsmoke_locked_protocol.json"
$StatusPath = Join-Path $PairOutputDir "postsmoke_status.json"

$newOutputs = @(
    $TraceAuditDir,
    $ForensicsControlDir,
    $ForensicsCandidateDir,
    $ConfusionsControlDir,
    $ConfusionsCandidateDir,
    $XaiCohortDir,
    $XaiControlDir,
    $XaiCandidateDir,
    $PairedXaiDir,
    $RobustnessControlDir,
    $RobustnessCandidateDir,
    $PostsmokeAuditDir,
    $ProtocolPath,
    $StatusPath
)

$protocol = [ordered]@{
    method = "inceptionnext_atto_surface_tokenizer_postsmoke"
    pair_summary = (Resolve-Path $PairSummary).Path
    pair_summary_sha256 = $pairSummarySha256
    stage_a_summary = (Resolve-Path $StageASummary).Path
    stage_a_summary_sha256 = $stageASha256
    data_yaml = (Resolve-Path $DataYaml).Path
    data_yaml_sha256 = $dataSha256
    control_checkpoint = (Resolve-Path $ControlCheckpoint).Path
    control_checkpoint_sha256 = $controlCheckpointSha256
    candidate_checkpoint = (Resolve-Path $CandidateCheckpoint).Path
    candidate_checkpoint_sha256 = $candidateCheckpointSha256
    candidate_trace = (Resolve-Path $CandidateTrace).Path
    candidate_trace_sha256 = $candidateTraceSha256
    split = "val"
    validation_support = 2606
    xai_cases = 16
    xai_precision = "fp32"
    xai_feature_source = "stem_output"
    robustness_conditions = @("clean", "occlusion_center", "lighting_dim", "lighting_bright", "low_contrast")
    raw_dataset_modified = $false
    test_allowed = $false
}

if ($PreflightOnly) {
    [ordered]@{
        status = "ok"
        mode = "preflight_only"
        outputs_available = -not ($newOutputs | Where-Object { Test-Path -LiteralPath $_ })
        protocol = $protocol
    } | ConvertTo-Json -Depth 6
    exit 0
}

foreach ($path in $newOutputs) {
    Assert-NewOutput $path
}
$protocol | ConvertTo-Json -Depth 6 | Set-Content -LiteralPath $ProtocolPath -Encoding UTF8

$status = @{
    status = "running"
    trace = "pending"
    forensics_control = "pending"
    forensics_candidate = "pending"
    confusions_control = "pending"
    confusions_candidate = "pending"
    xai_cohort = "pending"
    xai_control = "pending"
    xai_candidate = "pending"
    paired_xai = "pending"
    robustness_control = "pending"
    robustness_candidate = "pending"
    postsmoke = "pending"
}
Save-Status $status $StatusPath

Invoke-NativePython "audit_candidate_trace" @(
    "-m", "trkh.tools.audit_inceptionnext_atto_trace",
    "--trace-summary", $CandidateTrace,
    "--checkpoint", $CandidateCheckpoint,
    "--stage-a-summary", $StageASummary,
    "--output-dir", $TraceAuditDir
)
$status.trace = "completed"
Save-Status $status $StatusPath

$forensicsBase = @(
    "-m", "trkh.tools.audit_prediction_forensics",
    "--pairs", "0-1,1-2,2-3,4-rest",
    "--focus-class-index", "1",
    "--image-stats-mode", "foreground",
    "--max-image-stats", "0",
    "--top-k-images", "0"
)
Invoke-NativePython "forensics_control" ($forensicsBase + @(
    "--predictions", $ControlPredictions,
    "--output-dir", $ForensicsControlDir
))
$status.forensics_control = "completed"
Save-Status $status $StatusPath

Invoke-NativePython "forensics_candidate" ($forensicsBase + @(
    "--predictions", $CandidatePredictions,
    "--output-dir", $ForensicsCandidateDir
))
$status.forensics_candidate = "completed"
Save-Status $status $StatusPath

$confusionBase = @(
    "-m", "trkh.tools.audit_class_confusions",
    "--data", $DataYaml,
    "--focus-class-index", "1",
    "--top-k-images", "0"
)
Invoke-NativePython "confusions_control" ($confusionBase + @(
    "--predictions", $ControlPredictions,
    "--output-dir", $ConfusionsControlDir
))
$status.confusions_control = "completed"
Save-Status $status $StatusPath

Invoke-NativePython "confusions_candidate" ($confusionBase + @(
    "--predictions", $CandidatePredictions,
    "--output-dir", $ConfusionsCandidateDir
))
$status.confusions_candidate = "completed"
Save-Status $status $StatusPath

Invoke-NativePython "build_xai_cohort" @(
    "-m", "trkh.tools.build_inceptionnext_atto_xai_cohort",
    "--changed-cases", $ChangedCases,
    "--output-dir", $XaiCohortDir,
    "--max-cases", "16"
)
$status.xai_cohort = "completed"
Save-Status $status $StatusPath

$XaiCaseCsv = Join-Path $XaiCohortDir "xai_priority_cases.csv"
$xaiBase = @(
    "-m", "trkh.evaluation.xai_audit",
    "--data", $DataYaml,
    "--split", "val",
    "--class-name-mode", "raw",
    "--expected-num-classes", "5",
    "--max-cases", "16",
    "--case-csv", $XaiCaseCsv,
    "--batch-size", "1",
    "--num-workers", "0",
    "--layer", "-1",
    "--head-reduction", "mean",
    "--query-tokens", "cls_register_mean",
    "--method", "all",
    "--feature-source", "stem_output",
    "--rollout-start-layer", "0",
    "--robustness-probes",
    "--bbox-token-prior-source", "bbox",
    "--disable-amp"
)
Invoke-NativePython "xai_control" ($xaiBase + @(
    "--checkpoint", $ControlCheckpoint,
    "--output-dir", $XaiControlDir
))
$status.xai_control = "completed"
Save-Status $status $StatusPath

Invoke-NativePython "xai_candidate" ($xaiBase + @(
    "--checkpoint", $CandidateCheckpoint,
    "--output-dir", $XaiCandidateDir
))
$status.xai_candidate = "completed"
Save-Status $status $StatusPath

Invoke-NativePython "paired_xai" @(
    "-m", "trkh.tools.audit_paired_xai_cohort",
    "--cohort", $XaiCaseCsv,
    "--left-dir", $XaiControlDir,
    "--right-dir", $XaiCandidateDir,
    "--left-name", "control",
    "--right-name", "candidate",
    "--expected-cases", "16",
    "--output-dir", $PairedXaiDir
)
$status.paired_xai = "completed"
Save-Status $status $StatusPath

$robustnessBase = @(
    "-m", "trkh.evaluation.robustness_eval",
    "--data", $DataYaml,
    "--class-name-mode", "raw",
    "--expected-num-classes", "5",
    "--batch-size", "64",
    "--num-workers", "0",
    "--max-batches", "0",
    "--num-fail-cases", "4",
    "--layer", "-1",
    "--head-reduction", "mean",
    "--seed", "42"
)
Invoke-NativePython "robustness_control" ($robustnessBase + @(
    "--checkpoint", $ControlCheckpoint,
    "--output-dir", $RobustnessControlDir
))
$status.robustness_control = "completed"
Save-Status $status $StatusPath

Invoke-NativePython "robustness_candidate" ($robustnessBase + @(
    "--checkpoint", $CandidateCheckpoint,
    "--output-dir", $RobustnessCandidateDir
))
$status.robustness_candidate = "completed"
Save-Status $status $StatusPath

Invoke-NativePython "postsmoke_closure" @(
    "-m", "trkh.tools.audit_inceptionnext_atto_postsmoke",
    "--pair-summary", $PairSummary,
    "--trace-audit", (Join-Path $TraceAuditDir "summary.json"),
    "--paired-xai", (Join-Path $PairedXaiDir "summary.json"),
    "--robustness-control", $RobustnessControlDir,
    "--robustness-candidate", $RobustnessCandidateDir,
    "--forensics-control", (Join-Path $ForensicsControlDir "summary.json"),
    "--forensics-candidate", (Join-Path $ForensicsCandidateDir "summary.json"),
    "--confusions-control", (Join-Path $ConfusionsControlDir "summary.json"),
    "--confusions-candidate", (Join-Path $ConfusionsCandidateDir "summary.json"),
    "--output-dir", $PostsmokeAuditDir
)
$status.postsmoke = "completed"
$status.status = "completed"
Save-Status $status $StatusPath
Write-Host "InceptionNeXt-Atto post-smoke audit completed: $PostsmokeAuditDir"
