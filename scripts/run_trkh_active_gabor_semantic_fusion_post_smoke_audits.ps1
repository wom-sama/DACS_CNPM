param(
    [string]$Python = "D:\DataAI\.venv\Scripts\python.exe",
    [string]$PairOutputDir = "runs\audit_active_gabor_semantic_fusion_matched_smoke_pair_20260715",
    [string]$ControlRunName = "smoke_active_gabor_semantic_control_120b_5e_det_20260715",
    [string]$CandidateRunName = "smoke_active_gabor_semantic_candidate_120b_5e_det_20260715",
    [switch]$ResumeCompleted
)

$ErrorActionPreference = "Stop"
$RepoRoot = (Resolve-Path (Join-Path $PSScriptRoot "..")).Path
Set-Location $RepoRoot
$DataYaml = "D:\DataAI\AIEx\newdataset\yolo_f\data.yaml"
$ControlCheckpoint = Join-Path "runs\$ControlRunName" "checkpoints\best.pt"
$CandidateCheckpoint = Join-Path "runs\$CandidateRunName" "checkpoints\best.pt"
$Comparison = Join-Path $PairOutputDir "comparison"
$ChangedCases = Join-Path $Comparison "changed_cases.csv"
$SourceGroups = Join-Path $Comparison "source_group_forensics.csv"

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
    $ExpectedIndices = @(
        $ExpectedRows | ForEach-Object { [int]$_.sample_index } | Sort-Object
    )
    $ObservedIndices = @(
        $Payload.cases | ForEach-Object { [int]$_.sample_index } | Sort-Object
    )
    $ExpectedCheckpointPath = [System.IO.Path]::GetFullPath(
        (Join-Path $RepoRoot $ExpectedCheckpoint)
    )
    $ObservedCheckpointPath = [System.IO.Path]::GetFullPath(
        [string]$Payload.checkpoint
    )
    if (
        [int]$Payload.selected_cases -ne $ExpectedCount -or
        [string]$Payload.split -ne "val" -or
        [string]$Payload.method -ne "all" -or
        -not [bool]$Payload.robustness_probes -or
        -not $ObservedCheckpointPath.Equals(
            $ExpectedCheckpointPath,
            [System.StringComparison]::OrdinalIgnoreCase
        ) -or
        (($ExpectedIndices | ConvertTo-Json -Compress) -cne
            ($ObservedIndices | ConvertTo-Json -Compress))
    ) {
        throw "$Step summary does not match the exact active XAI cohort/checkpoint."
    }
}

foreach ($Required in @(
    $Python,
    $DataYaml,
    $ControlCheckpoint,
    $CandidateCheckpoint,
    (Join-Path $Comparison "summary.json"),
    $ChangedCases,
    $SourceGroups
)) {
    if (-not (Test-Path -LiteralPath $Required -PathType Leaf)) {
        throw "Required file not found: $Required"
    }
}

$ComparisonSummary = Get-Content -Raw -LiteralPath (Join-Path $Comparison "summary.json") | ConvertFrom-Json
$ExpectedChangedCases = [int]$ComparisonSummary.transitions.candidate_vs_control.changed_decisions
$ComparisonChangedRows = @(Import-Csv -LiteralPath $ChangedCases)
$NonExactComparisonRows = @(
    $ComparisonChangedRows | Where-Object {
        [int]$_.control_prediction -eq [int]$_.candidate_prediction
    }
)
if ($ComparisonChangedRows.Count -ne $ExpectedChangedCases -or $NonExactComparisonRows.Count -ne 0) {
    throw "Comparison changed-case artifact is not exact control-vs-candidate evidence: rows=$($ComparisonChangedRows.Count), expected=$ExpectedChangedCases, unchanged=$($NonExactComparisonRows.Count)."
}

$AuditRoot = Join-Path $PairOutputDir "post_smoke_audits"
if (Test-Path -LiteralPath $AuditRoot) {
    if (-not $ResumeCompleted) {
        throw "Post-smoke audit directory exists. Pass -ResumeCompleted to verify and resume: $AuditRoot"
    }
}
else {
    New-Item -ItemType Directory -Path $AuditRoot -Force | Out-Null
}
$GaborDiagnostics = Join-Path $AuditRoot "active_gabor_diagnostics"
$CohortDir = Join-Path $AuditRoot "xai_cohort"
$ControlXai = Join-Path $AuditRoot "xai_control"
$CandidateXai = Join-Path $AuditRoot "xai_candidate"
$Fp32CohortDir = Join-Path $AuditRoot "xai_cohort_fp32"
$PairedXai = Join-Path $AuditRoot "xai_paired"
$ArchitectureTrace = Join-Path $AuditRoot "candidate_architecture_trace"

if (Test-CompletedJson (Join-Path $GaborDiagnostics "summary.json") "active_gabor_full_val_robustness") {
    Write-Host "[active_gabor_full_val_robustness] verified completed artifact; skipping."
}
else {
    Invoke-NativePython "active_gabor_full_val_robustness" @(
        "-m", "trkh.tools.audit_active_gabor_semantic_fusion_post_smoke",
        "--control-checkpoint", $ControlCheckpoint,
        "--candidate-checkpoint", $CandidateCheckpoint,
        "--data", $DataYaml,
        "--output-dir", $GaborDiagnostics,
        "--batch-size", "64",
        "--num-workers", "0",
        "--seed", "42"
    )
}

foreach ($Role in @("control", "candidate")) {
    $Confusions = Join-Path $AuditRoot "$($Role)_confusions"
    $Predictions = Join-Path $PairOutputDir "$($Role)_val\predictions_detailed.csv"
    if (Test-CompletedJson (Join-Path $Confusions "summary.json") "$($Role)_confusions") {
        Write-Host "[$($Role)_confusions] verified completed artifact; skipping."
    }
    else {
        Invoke-NativePython "$($Role)_confusions" @(
            "-m", "trkh.tools.audit_class_confusions",
            "--predictions", $Predictions,
            "--data", $DataYaml,
            "--focus-class-index", "1",
            "--output-dir", $Confusions
        )
    }
}

$CandidateBoundaryForensics = Join-Path $AuditRoot "candidate_boundary_forensics"
if (Test-CompletedJson (Join-Path $CandidateBoundaryForensics "summary.json") "candidate_boundary_forensics") {
    Write-Host "[candidate_boundary_forensics] verified completed artifact; skipping."
}
else {
    Invoke-NativePython "candidate_boundary_forensics" @(
        "-m", "trkh.tools.build_boundary_review_manifest",
        "--predictions", (Join-Path $PairOutputDir "candidate_val\predictions_detailed.csv"),
        "--split", "val",
        "--pairs", "0-1,1-2,1-4,2-3",
        "--focus-class-index", "1",
        "--image-stats-mode", "foreground",
        "--max-image-stats", "32",
        "--max-total", "64",
        "--output-dir", $CandidateBoundaryForensics
    )
}

if (Test-CompletedJson (Join-Path $CohortDir "summary.json") "build_xai_cohort") {
    Write-Host "[build_xai_cohort] verified completed artifact; skipping."
}
else {
    Invoke-NativePython "build_xai_cohort" @(
        "-m", "trkh.tools.build_tokenizer_xai_cohort",
        "--changed-cases", $ChangedCases,
        "--output-dir", $CohortDir,
        "--max-cases", "16",
        "--mode", "validation_only_active_gabor_semantic_changed_case_cohort"
    )
}
$CohortSummary = Get-Content -Raw -LiteralPath (Join-Path $CohortDir "summary.json") | ConvertFrom-Json
$CaseCount = [int]$CohortSummary.selected_case_count
$CohortCsv = Join-Path $CohortDir "xai_priority_cases.csv"
$CohortRows = @(Import-Csv -LiteralPath $CohortCsv)
$NonExactCohortRows = @(
    $CohortRows | Where-Object {
        [int]$_.control_prediction_index -eq [int]$_.candidate_prediction_index
    }
)
if (
    [int]$CohortSummary.source_row_count -ne $ExpectedChangedCases -or
    [int]$CohortSummary.changed_case_count -ne $ExpectedChangedCases -or
    [int]$CohortSummary.excluded_unchanged_decisions -ne 0 -or
    $CohortRows.Count -ne $CaseCount -or
    $NonExactCohortRows.Count -ne 0
) {
    throw "XAI cohort is not an exact control-vs-candidate decision-change cohort."
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
    "--bbox-token-prior-source", "crop_bbox",
    "--disable-amp",
    "--batch-size", "32",
    "--num-workers", "0"
)
if (Test-CompletedJson (Join-Path $ControlXai "xai_audit_summary.json") "xai_control_fp32") {
    Assert-XaiSummaryMatchesCohort `
        -SummaryPath (Join-Path $ControlXai "xai_audit_summary.json") `
        -ExpectedCheckpoint $ControlCheckpoint `
        -ExpectedRows $CohortRows `
        -ExpectedCount $CaseCount `
        -Step "xai_control_fp32"
    Write-Host "[xai_control_fp32] verified completed artifact; skipping."
}
else {
    Invoke-NativePython "xai_control_fp32" ($XaiBase + @(
        "--checkpoint", $ControlCheckpoint,
        "--output-dir", $ControlXai
    ))
}
if (Test-CompletedJson (Join-Path $CandidateXai "xai_audit_summary.json") "xai_candidate_fp32") {
    Assert-XaiSummaryMatchesCohort `
        -SummaryPath (Join-Path $CandidateXai "xai_audit_summary.json") `
        -ExpectedCheckpoint $CandidateCheckpoint `
        -ExpectedRows $CohortRows `
        -ExpectedCount $CaseCount `
        -Step "xai_candidate_fp32"
    Write-Host "[xai_candidate_fp32] verified completed artifact; skipping."
}
else {
    Invoke-NativePython "xai_candidate_fp32" ($XaiBase + @(
        "--checkpoint", $CandidateCheckpoint,
        "--output-dir", $CandidateXai
    ))
}
if (Test-CompletedJson (Join-Path $Fp32CohortDir "summary.json") "reconcile_fp32_xai_cohort") {
    Write-Host "[reconcile_fp32_xai_cohort] verified completed artifact; skipping."
}
else {
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
$Fp32Summary = Get-Content -Raw -LiteralPath (Join-Path $Fp32CohortDir "summary.json") | ConvertFrom-Json
$ObservedFp32CohortSha256 = (Get-FileHash -Algorithm SHA256 -LiteralPath $Fp32CohortCsv).Hash.ToLowerInvariant()
if (
    [int]$Fp32Summary.cases -ne $CaseCount -or
    [string]$Fp32Summary.source_cohort_sha256 -ne [string]$CohortSummary.cohort_sha256 -or
    [string]$Fp32Summary.reconciled_cohort_sha256 -ne $ObservedFp32CohortSha256 -or
    [string]$Fp32Summary.xai_backend -ne "fp32_disable_amp"
) {
    throw "FP32 reconciliation does not match the exact active XAI cohort."
}
if (Test-CompletedJson (Join-Path $PairedXai "summary.json") "paired_xai_audit") {
    Write-Host "[paired_xai_audit] verified completed artifact; skipping."
}
else {
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
$PairedSummary = Get-Content -Raw -LiteralPath (Join-Path $PairedXai "summary.json") | ConvertFrom-Json
if (
    [int]$PairedSummary.cases -ne $CaseCount -or
    [string]$PairedSummary.cohort_sha256 -ne $ObservedFp32CohortSha256 -or
    [int]$PairedSummary.left.case_json_count -ne $CaseCount -or
    [int]$PairedSummary.right.case_json_count -ne $CaseCount
) {
    throw "Paired XAI summary does not match the reconciled FP32 cohort."
}
if (Test-CompletedJson (Join-Path $ArchitectureTrace "trace_summary.json") "candidate_architecture_trace") {
    Write-Host "[candidate_architecture_trace] verified completed artifact; skipping."
}
else {
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

[ordered]@{
    status = "completed"
    test_used = $false
    raw_dataset_modified = $false
    comparison = (Resolve-Path $Comparison).Path
    comparison_summary_sha256 = (Get-FileHash -Algorithm SHA256 -LiteralPath (Join-Path $Comparison "summary.json")).Hash.ToLowerInvariant()
    exact_changed_decisions = $ExpectedChangedCases
    source_group_forensics = (Resolve-Path $SourceGroups).Path
    active_gabor_diagnostics = (Resolve-Path $GaborDiagnostics).Path
    xai_cohort = (Resolve-Path $CohortDir).Path
    xai_cohort_fp32 = (Resolve-Path $Fp32CohortDir).Path
    xai_control = (Resolve-Path $ControlXai).Path
    xai_candidate = (Resolve-Path $CandidateXai).Path
    paired_xai = (Resolve-Path $PairedXai).Path
    architecture_trace = (Resolve-Path $ArchitectureTrace).Path
    xai_cases = $CaseCount
} | ConvertTo-Json -Depth 4 | Set-Content -LiteralPath (Join-Path $AuditRoot "status.json") -Encoding UTF8
Write-Host "Active-Gabor post-smoke audits completed: $AuditRoot"
