param(
    [string]$Python = "D:\DataAI\.venv\Scripts\python.exe",
    [string]$PairOutputDir = "runs\audit_natural_prior_continuation_matched_smoke_20260715",
    [string]$ControlRunName = "smoke_natural_prior_control_strict_60b_2e_20260715",
    [string]$CandidateRunName = "smoke_natural_prior_candidate_random_60b_2e_20260715",
    [switch]$ResumeCompleted
)

$ErrorActionPreference = "Stop"
$RepoRoot = (Resolve-Path (Join-Path $PSScriptRoot "..")).Path
Set-Location $RepoRoot
$DataYaml = "D:\DataAI\AIEx\newdataset\yolo_f\data.yaml"
$ControlCheckpoint = Join-Path "runs\$ControlRunName" "checkpoints\best.pt"
$CandidateCheckpoint = Join-Path "runs\$CandidateRunName" "checkpoints\best.pt"
$ControlPredictions = Join-Path $PairOutputDir "control_val_fp32\predictions_detailed.csv"
$CandidatePredictions = Join-Path $PairOutputDir "candidate_val_fp32\predictions_detailed.csv"
$Comparison = Join-Path $PairOutputDir "comparison"
$ChangedCases = Join-Path $Comparison "changed_cases.csv"

function Invoke-NativePython {
    param([string]$Step, [string[]]$Arguments)
    Write-Host "[$Step] $Python $($Arguments -join ' ')"
    $PreviousErrorActionPreference = $ErrorActionPreference
    $ErrorActionPreference = "Continue"
    try {
        & $Python @Arguments
        $ExitCode = $LASTEXITCODE
    }
    finally {
        $ErrorActionPreference = $PreviousErrorActionPreference
    }
    if ($ExitCode -ne 0) {
        throw "$Step failed with exit code $ExitCode."
    }
}

function Test-CompletedJson {
    param([string]$Path, [string]$Step)
    if (-not (Test-Path -LiteralPath $Path -PathType Leaf)) {
        return $false
    }
    try {
        $Payload = Get-Content -Raw -LiteralPath $Path | ConvertFrom-Json
    }
    catch {
        throw "$Step completion artifact is not valid JSON: $Path"
    }
    if ($null -eq $Payload) {
        throw "$Step completion artifact is empty: $Path"
    }
    return $true
}

function Assert-XaiSummaryMatchesCohort {
    param(
        [string]$SummaryPath,
        [string]$ExpectedCheckpoint,
        [object[]]$ExpectedRows,
        [int]$ExpectedCount,
        [string]$Step
    )
    $Payload = Get-Content -Raw -LiteralPath $SummaryPath | ConvertFrom-Json
    $ExpectedIndices = @($ExpectedRows | ForEach-Object { [int]$_.sample_index } | Sort-Object)
    $ObservedIndices = @($Payload.cases | ForEach-Object { [int]$_.sample_index } | Sort-Object)
    $ExpectedCheckpointPath = [System.IO.Path]::GetFullPath((Join-Path $RepoRoot $ExpectedCheckpoint))
    $ObservedCheckpointPath = [System.IO.Path]::GetFullPath([string]$Payload.checkpoint)
    if (
        [int]$Payload.selected_cases -ne $ExpectedCount -or
        [string]$Payload.split -ne "val" -or
        [string]$Payload.method -ne "all" -or
        -not [bool]$Payload.robustness_probes -or
        -not $ObservedCheckpointPath.Equals($ExpectedCheckpointPath, [System.StringComparison]::OrdinalIgnoreCase) -or
        (($ExpectedIndices | ConvertTo-Json -Compress) -cne ($ObservedIndices | ConvertTo-Json -Compress))
    ) {
        throw "$Step summary does not match the exact natural-prior XAI cohort."
    }
}

foreach ($Required in @(
    $Python,
    $DataYaml,
    $ControlCheckpoint,
    $CandidateCheckpoint,
    $ControlPredictions,
    $CandidatePredictions,
    (Join-Path $Comparison "summary.json"),
    $ChangedCases
)) {
    if (-not (Test-Path -LiteralPath $Required -PathType Leaf)) {
        throw "Required file not found: $Required"
    }
}

$ComparisonSummary = Get-Content -Raw -LiteralPath (Join-Path $Comparison "summary.json") | ConvertFrom-Json
$ExpectedChanged = [int]$ComparisonSummary.transitions.candidate_vs_control.changed_decisions
$ChangedRows = @(Import-Csv -LiteralPath $ChangedCases)
$UnchangedRows = @($ChangedRows | Where-Object {
    [int]$_.control_prediction -eq [int]$_.candidate_prediction
})
if ($ChangedRows.Count -ne $ExpectedChanged -or $UnchangedRows.Count -ne 0) {
    throw "Changed-case CSV is not exact: rows=$($ChangedRows.Count), expected=$ExpectedChanged, unchanged=$($UnchangedRows.Count)."
}

$AuditRoot = Join-Path $PairOutputDir "post_smoke_audits"
if (Test-Path -LiteralPath $AuditRoot) {
    if (-not $ResumeCompleted) {
        throw "Post-smoke audit directory exists. Pass -ResumeCompleted: $AuditRoot"
    }
}
else {
    New-Item -ItemType Directory -Path $AuditRoot -Force | Out-Null
}
$env:PYTHONPATH = $RepoRoot
$env:TRKH_AMP_DTYPE = "bf16"

foreach ($Role in @("control", "candidate")) {
    $Checkpoint = if ($Role -eq "control") { $ControlCheckpoint } else { $CandidateCheckpoint }
    $Predictions = if ($Role -eq "control") { $ControlPredictions } else { $CandidatePredictions }
    $Confusions = Join-Path $AuditRoot "$($Role)_confusions"
    $Forensics = Join-Path $AuditRoot "$($Role)_forensics"
    $Boundary = Join-Path $AuditRoot "$($Role)_boundary"
    $Robustness = Join-Path $AuditRoot "$($Role)_robustness"
    if (-not (Test-CompletedJson (Join-Path $Confusions "summary.json") "$($Role)_confusions")) {
        Invoke-NativePython "$($Role)_confusions" @(
            "-m", "trkh.tools.audit_class_confusions",
            "--predictions", $Predictions,
            "--data", $DataYaml,
            "--focus-class-index", "1",
            "--top-k-images", "24",
            "--output-dir", $Confusions
        )
    }
    if (-not (Test-CompletedJson (Join-Path $Forensics "summary.json") "$($Role)_forensics")) {
        Invoke-NativePython "$($Role)_forensics" @(
            "-m", "trkh.tools.audit_prediction_forensics",
            "--predictions", $Predictions,
            "--pairs", "0-1,1-2,2-3,4-rest",
            "--focus-class-index", "1",
            "--image-stats-mode", "basic",
            "--max-image-stats", "500",
            "--top-k-images", "24",
            "--output-dir", $Forensics
        )
    }
    if (-not (Test-CompletedJson (Join-Path $Boundary "summary.json") "$($Role)_boundary")) {
        Invoke-NativePython "$($Role)_boundary" @(
            "-m", "trkh.tools.build_boundary_review_manifest",
            "--predictions", $Predictions,
            "--split", "val",
            "--pairs", "0-1,1-2,1-4,2-3",
            "--focus-class-index", "1",
            "--image-stats-mode", "foreground",
            "--max-image-stats", "32",
            "--max-total", "64",
            "--output-dir", $Boundary
        )
    }
    if (-not (Test-CompletedJson (Join-Path $Robustness "robustness_summary.json") "$($Role)_robustness")) {
        Invoke-NativePython "$($Role)_robustness" @(
            "-m", "trkh.evaluation.robustness_eval",
            "--checkpoint", $Checkpoint,
            "--data", $DataYaml,
            "--class-name-mode", "raw",
            "--expected-num-classes", "5",
            "--batch-size", "32",
            "--num-workers", "0",
            "--num-fail-cases", "12",
            "--output-dir", $Robustness
        )
    }
}

$ArchitectureTrace = Join-Path $AuditRoot "candidate_architecture_trace"
if (-not (Test-CompletedJson (Join-Path $ArchitectureTrace "trace_summary.json") "candidate_architecture_trace")) {
    Invoke-NativePython "candidate_architecture_trace" @(
        "-m", "trkh.tools.trace_architecture",
        "--checkpoint", $CandidateCheckpoint,
        "--data", $DataYaml,
        "--class-name-mode", "raw",
        "--expected-num-classes", "5",
        "--device", "cuda",
        "--seed", "42",
        "--output-dir", $ArchitectureTrace
    )
}

$CohortDir = Join-Path $AuditRoot "xai_cohort"
$ControlXai = Join-Path $AuditRoot "xai_control"
$CandidateXai = Join-Path $AuditRoot "xai_candidate"
$Fp32CohortDir = Join-Path $AuditRoot "xai_cohort_fp32"
$PairedXai = Join-Path $AuditRoot "xai_paired"
$PairedXaiStatus = "not_required_no_changed_decisions"
$CaseCount = 0
if ($ExpectedChanged -gt 0) {
    if (-not (Test-CompletedJson (Join-Path $CohortDir "summary.json") "build_xai_cohort")) {
        Invoke-NativePython "build_xai_cohort" @(
            "-m", "trkh.tools.build_tokenizer_xai_cohort",
            "--changed-cases", $ChangedCases,
            "--output-dir", $CohortDir,
            "--max-cases", "16",
            "--mode", "validation_only_natural_prior_changed_case_cohort"
        )
    }
    $CohortSummary = Get-Content -Raw -LiteralPath (Join-Path $CohortDir "summary.json") | ConvertFrom-Json
    $CaseCount = [int]$CohortSummary.selected_case_count
    $CohortCsv = Join-Path $CohortDir "xai_priority_cases.csv"
    $CohortRows = @(Import-Csv -LiteralPath $CohortCsv)
    if (
        [int]$CohortSummary.source_row_count -ne $ExpectedChanged -or
        [int]$CohortSummary.changed_case_count -ne $ExpectedChanged -or
        $CohortRows.Count -ne $CaseCount
    ) {
        throw "Natural-prior XAI cohort does not match changed-case evidence."
    }
    $XaiBase = @(
        "-m", "trkh.evaluation.xai_audit",
        "--data", $DataYaml,
        "--split", "val",
        "--class-name-mode", "raw",
        "--expected-num-classes", "5",
        "--case-csv", $CohortCsv,
        "--max-cases", "$CaseCount",
        "--mistake-cases", "0",
        "--low-confidence-cases", "0",
        "--close-margin-cases", "0",
        "--per-class-cases", "0",
        "--method", "all",
        "--robustness-probes",
        "--bbox-token-prior-source", "bbox",
        "--disable-amp",
        "--batch-size", "32",
        "--num-workers", "0"
    )
    if (Test-CompletedJson (Join-Path $ControlXai "xai_audit_summary.json") "xai_control") {
        Assert-XaiSummaryMatchesCohort (Join-Path $ControlXai "xai_audit_summary.json") $ControlCheckpoint $CohortRows $CaseCount "xai_control"
    }
    else {
        Invoke-NativePython "xai_control" ($XaiBase + @(
            "--checkpoint", $ControlCheckpoint,
            "--output-dir", $ControlXai
        ))
    }
    if (Test-CompletedJson (Join-Path $CandidateXai "xai_audit_summary.json") "xai_candidate") {
        Assert-XaiSummaryMatchesCohort (Join-Path $CandidateXai "xai_audit_summary.json") $CandidateCheckpoint $CohortRows $CaseCount "xai_candidate"
    }
    else {
        Invoke-NativePython "xai_candidate" ($XaiBase + @(
            "--checkpoint", $CandidateCheckpoint,
            "--output-dir", $CandidateXai
        ))
    }
    if (-not (Test-CompletedJson (Join-Path $Fp32CohortDir "summary.json") "reconcile_fp32_xai_cohort")) {
        Invoke-NativePython "reconcile_fp32_xai_cohort" @(
            "-m", "trkh.tools.reconcile_fp32_xai_cohort",
            "--cohort", $CohortCsv,
            "--control-xai-dir", $ControlXai,
            "--candidate-xai-dir", $CandidateXai,
            "--expected-cases", "$CaseCount",
            "--output-dir", $Fp32CohortDir
        )
    }
    $Fp32CohortCsv = Join-Path $Fp32CohortDir "xai_priority_cases_fp32.csv"
    if (-not (Test-CompletedJson (Join-Path $PairedXai "summary.json") "paired_xai_audit")) {
        Invoke-NativePython "paired_xai_audit" @(
            "-m", "trkh.tools.audit_paired_xai_cohort",
            "--cohort", $Fp32CohortCsv,
            "--left-dir", $ControlXai,
            "--right-dir", $CandidateXai,
            "--left-name", "control",
            "--right-name", "candidate",
            "--expected-cases", "$CaseCount",
            "--output-dir", $PairedXai
        )
    }
    $PairedXaiStatus = "completed"
}

$Status = [ordered]@{
    status = "completed"
    test_used = $false
    raw_dataset_modified = $false
    exact_changed_decisions = $ExpectedChanged
    xai_cases = $CaseCount
    paired_xai = $PairedXaiStatus
    comparison_summary_sha256 = (Get-FileHash -Algorithm SHA256 -LiteralPath (Join-Path $Comparison "summary.json")).Hash.ToLowerInvariant()
    candidate_architecture_trace = $ArchitectureTrace
}
$Status | ConvertTo-Json -Depth 4 | Set-Content -LiteralPath (Join-Path $AuditRoot "status.json") -Encoding UTF8
Write-Host "Natural-prior post-smoke audits completed: $AuditRoot"
