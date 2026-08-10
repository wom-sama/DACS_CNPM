param(
    [string]$Python = "D:\DataAI\.venv\Scripts\python.exe",
    [string]$PairOutputDir = "runs\audit_gabor_lho_fcm_rngneutral_matched_smoke_pair_20260715",
    [string]$ControlRunName = "smoke_gabor_lho_fcm_rngneutral_control_120b_2e_20260715",
    [string]$CandidateRunName = "smoke_gabor_lho_fcm_rngneutral_candidate_120b_2e_20260715",
    [switch]$ResumeCompleted
)

$ErrorActionPreference = "Stop"
$RepoRoot = (Resolve-Path (Join-Path $PSScriptRoot "..")).Path
Set-Location $RepoRoot
$DataYaml = "D:\DataAI\AIEx\newdataset\yolo_f\data.yaml"
$KeeperCheckpoint = "runs\probe_v8_yolof_pairroute_teacherfocusbinary015_boundarydrop_bboxprior_120b_2e_20260701\checkpoints\best.pt"
$ControlCheckpoint = Join-Path "runs\$ControlRunName" "checkpoints\best.pt"
$CandidateCheckpoint = Join-Path "runs\$CandidateRunName" "checkpoints\best.pt"
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

foreach ($Required in @(
    $Python,
    $DataYaml,
    $KeeperCheckpoint,
    $ControlCheckpoint,
    $CandidateCheckpoint,
    (Join-Path $Comparison "summary.json"),
    $ChangedCases
)) {
    if (-not (Test-Path -LiteralPath $Required -PathType Leaf)) {
        throw "Required file not found: $Required"
    }
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
$GaborDiagnostics = Join-Path $AuditRoot "gabor_diagnostics"
$CohortDir = Join-Path $AuditRoot "xai_cohort"
$ControlXai = Join-Path $AuditRoot "xai_control"
$CandidateXai = Join-Path $AuditRoot "xai_candidate"
$Fp32CohortDir = Join-Path $AuditRoot "xai_cohort_fp32"
$PairedXai = Join-Path $AuditRoot "xai_paired"
$ArchitectureTrace = Join-Path $AuditRoot "candidate_architecture_trace"

if (Test-CompletedJson (Join-Path $GaborDiagnostics "summary.json") "gabor_full_val_robustness") {
    Write-Host "[gabor_full_val_robustness] verified completed artifact; skipping."
}
else {
    Invoke-NativePython "gabor_full_val_robustness" @(
        "-m", "trkh.tools.audit_learnable_gabor_texture_post_smoke",
        "--control-checkpoint", $ControlCheckpoint,
        "--initialization-checkpoint", $KeeperCheckpoint,
        "--candidate-checkpoint", $CandidateCheckpoint,
        "--data", $DataYaml,
        "--output-dir", $GaborDiagnostics,
        "--batch-size", "64",
        "--num-workers", "0"
    )
}
$CandidateConfusions = Join-Path $AuditRoot "candidate_confusions"
if (Test-CompletedJson (Join-Path $CandidateConfusions "summary.json") "candidate_confusions") {
    Write-Host "[candidate_confusions] verified completed artifact; skipping."
}
else {
    Invoke-NativePython "candidate_confusions" @(
        "-m", "trkh.tools.audit_class_confusions",
        "--predictions", (Join-Path $PairOutputDir "candidate_val\predictions_detailed.csv"),
        "--data", $DataYaml,
        "--focus-class-index", "1",
        "--output-dir", $CandidateConfusions
    )
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
        "--mode", "validation_only_gabor_matched_changed_case_cohort"
    )
}
$CohortSummary = Get-Content -Raw -LiteralPath (Join-Path $CohortDir "summary.json") | ConvertFrom-Json
$CaseCount = [int]$CohortSummary.selected_case_count
$CohortCsv = Join-Path $CohortDir "xai_priority_cases.csv"
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
    Write-Host "[xai_control_fp32] verified completed artifact; skipping."
}
else {
    Invoke-NativePython "xai_control_fp32" ($XaiBase + @(
        "--checkpoint", $ControlCheckpoint,
        "--output-dir", $ControlXai
    ))
}
if (Test-CompletedJson (Join-Path $CandidateXai "xai_audit_summary.json") "xai_candidate_fp32") {
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
    gabor_diagnostics = (Resolve-Path $GaborDiagnostics).Path
    xai_cohort = (Resolve-Path $CohortDir).Path
    xai_cohort_fp32 = (Resolve-Path $Fp32CohortDir).Path
    xai_control = (Resolve-Path $ControlXai).Path
    xai_candidate = (Resolve-Path $CandidateXai).Path
    paired_xai = (Resolve-Path $PairedXai).Path
    architecture_trace = (Resolve-Path $ArchitectureTrace).Path
    xai_cases = $CaseCount
} | ConvertTo-Json -Depth 4 | Set-Content -LiteralPath (Join-Path $AuditRoot "status.json") -Encoding UTF8
Write-Host "Gabor post-smoke audits completed: $AuditRoot"
