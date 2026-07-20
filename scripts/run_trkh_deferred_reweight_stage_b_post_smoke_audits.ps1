[CmdletBinding()]
param(
    [string]$Python = "D:\DataAI\.venv\Scripts\python.exe",
    [string]$RunName = "smoke_v8_yolof_natural_prior_randominit_5e_20260720",
    [string]$AuditOutputDir = "runs\audit_deferred_reweight_stage_b_natural_smoke_20260720",
    [switch]$ResumeCompleted,
    [switch]$PreflightOnly
)

$ErrorActionPreference = "Stop"
$RepoRoot = (Resolve-Path (Join-Path $PSScriptRoot "..")).Path
Set-Location $RepoRoot
$env:PYTHONPATH = $RepoRoot
$env:TRKH_AMP_DTYPE = "bf16"
$env:OMP_NUM_THREADS = "4"
$env:TRKH_ALLOW_WINDOWS_MULTIPROCESSING = "1"

$DataYaml = "D:\DataAI\AIEx\newdataset\yolo_f\data.yaml"
$RunDir = Join-Path "runs" $RunName
$Checkpoint = Join-Path $RunDir "checkpoints\best.pt"
$History = Join-Path $RunDir "history.csv"
$Trace = Join-Path $RunDir "architecture_trace\trace_summary.json"
$TrainStatus = Join-Path $AuditOutputDir "status.json"
$Metrics = Join-Path $AuditOutputDir "independent_val_fp32\metrics_detailed.json"
$Predictions = Join-Path $AuditOutputDir "independent_val_fp32\predictions_detailed.csv"
$KeeperCheckpoint = "runs\probe_v8_yolof_pairroute_teacherfocusbinary015_boundarydrop_bboxprior_120b_2e_20260701\checkpoints\best.pt"
$KeeperRobustness = "runs\full_v8_yolof_randominit_30e_20260714_105524\final_audit_manual\robustness_keeper_val_raw\robustness_summary.json"
$KeeperXaiDir = "runs\xai_boundary_review_yolof_keeper_val_current_cases16_datasetcrop_20260704"
$KeeperXaiSummary = Join-Path $KeeperXaiDir "xai_audit_summary.json"
$KeeperXaiCases = Join-Path $KeeperXaiDir "xai_cases.csv"
$LockedDataSha256 = "716e33df24c63a9e9920f97b685199707fb84ab4c7154544f5dd9a3e00d884ef"
$LockedKeeperCheckpointSha256 = "1f49d577240c69dc63c30af70db52ec2aa9da65a17aef1c4b1c09ece6c482677"
$LockedKeeperRobustnessSha256 = "23af842206c84e25f22e8530b37fd5f3f148650ea7087e6319492627300f624b"
$LockedKeeperXaiSummarySha256 = "a3a7c22fde05dd404b6a7ad85e5ed3eb71b5b9d005a4cbe33f60c0a50ed3f047"
$LockedKeeperXaiCasesSha256 = "4201a469209edcfb29c88130a99402c3360a3a3f2e1c603eac8ae4882dcfefa9"
$PostRoot = Join-Path $AuditOutputDir "post_smoke_audits"
$Confusions = Join-Path $PostRoot "confusions"
$Forensics = Join-Path $PostRoot "forensics"
$Boundary = Join-Path $PostRoot "boundary"
$Robustness = Join-Path $PostRoot "robustness"
$CohortDir = Join-Path $PostRoot "xai_cohort"
$CohortCsv = Join-Path $CohortDir "fixed_boundary_cases12.csv"
$KeeperSubset = Join-Path $PostRoot "xai_keeper_subset12"
$CandidateXai = Join-Path $PostRoot "xai_candidate"
$PairedXai = Join-Path $PostRoot "xai_paired"
$GateDir = Join-Path $PostRoot "gate"

function Assert-FileSha256 {
    param([string]$Path, [string]$Expected)
    if (-not (Test-Path -LiteralPath $Path -PathType Leaf)) {
        throw "Required file not found: $Path"
    }
    $Observed = (Get-FileHash -Algorithm SHA256 -LiteralPath $Path).Hash.ToLowerInvariant()
    if ($Observed -ne $Expected.ToLowerInvariant()) {
        throw "SHA-256 mismatch for $Path`: $Observed != $Expected"
    }
    return $Observed
}

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
    param([string]$Path)
    if (-not (Test-Path -LiteralPath $Path -PathType Leaf)) {
        return $false
    }
    try {
        $Payload = Get-Content -Raw -LiteralPath $Path | ConvertFrom-Json
    }
    catch {
        throw "Completion artifact is not valid JSON: $Path"
    }
    return $null -ne $Payload
}

foreach ($Required in @(
    $Python,
    $Checkpoint,
    $History,
    $Trace,
    $TrainStatus,
    $Metrics,
    $Predictions,
    $KeeperCheckpoint,
    $KeeperRobustness,
    $KeeperXaiSummary,
    $KeeperXaiCases
)) {
    if (-not (Test-Path -LiteralPath $Required -PathType Leaf)) {
        throw "Required file not found: $Required"
    }
}

$DataSha256 = Assert-FileSha256 $DataYaml $LockedDataSha256
$KeeperCheckpointSha256 = Assert-FileSha256 $KeeperCheckpoint $LockedKeeperCheckpointSha256
$KeeperRobustnessSha256 = Assert-FileSha256 $KeeperRobustness $LockedKeeperRobustnessSha256
$KeeperXaiSummarySha256 = Assert-FileSha256 $KeeperXaiSummary $LockedKeeperXaiSummarySha256
$KeeperXaiCasesSha256 = Assert-FileSha256 $KeeperXaiCases $LockedKeeperXaiCasesSha256
$TrainStatusPayload = Get-Content -Raw -LiteralPath $TrainStatus | ConvertFrom-Json
if (
    [string]$TrainStatusPayload.status -ne "completed_pending_gate_audit" -or
    [string]$TrainStatusPayload.train -ne "completed" -or
    [string]$TrainStatusPayload.independent_validation -ne "completed" -or
    [bool]$TrainStatusPayload.test_used
) {
    throw "Stage-B train status is not a completed no-test smoke."
}

$KeeperXaiPayload = Get-Content -Raw -LiteralPath $KeeperXaiSummary | ConvertFrom-Json
if (
    [string]$KeeperXaiPayload.split -ne "val" -or
    [string]$KeeperXaiPayload.method -ne "all" -or
    -not [bool]$KeeperXaiPayload.robustness_probes -or
    [int]$KeeperXaiPayload.selected_cases -ne 16
) {
    throw "Locked keeper XAI is not the expected corrected validation audit."
}

$TrackedStatus = git status --porcelain --untracked-files=no
if (-not $PreflightOnly -and ($LASTEXITCODE -ne 0 -or $TrackedStatus)) {
    throw "Tracked worktree must be clean before formal post-smoke audits."
}
$RepoCommit = (git rev-parse HEAD).Trim()
if ($LASTEXITCODE -ne 0) {
    throw "Unable to resolve repository HEAD."
}

$LockedInputs = [ordered]@{
    mode = "deferred_reweight_stage_b_post_smoke_audits"
    repository_commit = $RepoCommit
    test_allowed = $false
    raw_dataset_modification_allowed = $false
    full_train_allowed = $false
    deferred_reweight_allowed = $false
    xai_case_count = 12
    xai_case_selection = "first_12_rows_of_locked_corrected_keeper_boundary_audit"
    data_yaml_sha256 = $DataSha256
    keeper_checkpoint_sha256 = $KeeperCheckpointSha256
    keeper_robustness_sha256 = $KeeperRobustnessSha256
    keeper_xai_summary_sha256 = $KeeperXaiSummarySha256
    keeper_xai_cases_sha256 = $KeeperXaiCasesSha256
    candidate_checkpoint_sha256 = (Get-FileHash -Algorithm SHA256 -LiteralPath $Checkpoint).Hash.ToLowerInvariant()
    candidate_history_sha256 = (Get-FileHash -Algorithm SHA256 -LiteralPath $History).Hash.ToLowerInvariant()
    candidate_metrics_sha256 = (Get-FileHash -Algorithm SHA256 -LiteralPath $Metrics).Hash.ToLowerInvariant()
    candidate_predictions_sha256 = (Get-FileHash -Algorithm SHA256 -LiteralPath $Predictions).Hash.ToLowerInvariant()
    candidate_trace_sha256 = (Get-FileHash -Algorithm SHA256 -LiteralPath $Trace).Hash.ToLowerInvariant()
}

if ($PreflightOnly) {
    [ordered]@{status = "ok"; mode = "preflight_only"; locked_inputs = $LockedInputs} |
        ConvertTo-Json -Depth 5
    exit 0
}

if (Test-Path -LiteralPath $PostRoot) {
    if (-not $ResumeCompleted) {
        throw "Post-smoke audit directory exists. Pass -ResumeCompleted: $PostRoot"
    }
}
else {
    New-Item -ItemType Directory -Path $PostRoot -Force | Out-Null
}
$LockedInputs | ConvertTo-Json -Depth 5 | Set-Content -LiteralPath (Join-Path $PostRoot "locked_inputs.json") -Encoding UTF8
$PostStatusPath = Join-Path $PostRoot "status.json"
[ordered]@{status = "running"; test_used = $false} | ConvertTo-Json |
    Set-Content -LiteralPath $PostStatusPath -Encoding UTF8

if (-not (Test-CompletedJson (Join-Path $Confusions "summary.json"))) {
    Invoke-NativePython "confusions" @(
        "-m", "trkh.tools.audit_class_confusions",
        "--predictions", $Predictions,
        "--data", $DataYaml,
        "--focus-class-index", "1",
        "--top-k-images", "24",
        "--output-dir", $Confusions
    )
}
if (-not (Test-CompletedJson (Join-Path $Forensics "summary.json"))) {
    Invoke-NativePython "forensics" @(
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
if (-not (Test-CompletedJson (Join-Path $Boundary "summary.json"))) {
    Invoke-NativePython "boundary" @(
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
if (-not (Test-CompletedJson (Join-Path $Robustness "robustness_summary.json"))) {
    Invoke-NativePython "robustness" @(
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

if (-not (Test-Path -LiteralPath $CohortCsv -PathType Leaf)) {
    New-Item -ItemType Directory -Path $CohortDir -Force | Out-Null
    $Rows = @(Import-Csv -LiteralPath $KeeperXaiCases | Select-Object -First 12)
    if ($Rows.Count -ne 12) {
        throw "Locked keeper XAI case CSV does not contain 12 rows."
    }
    $CohortRows = foreach ($Row in $Rows) {
        $Category = if ([int]$Row.target_index -eq 1 -and [int]$Row.prediction_index -ne 1) {
            "class1_fn"
        }
        elseif ([int]$Row.target_index -ne 1 -and [int]$Row.prediction_index -eq 1) {
            "class1_fp"
        }
        else {
            "nonfocus_boundary"
        }
        [pscustomobject]@{
            sample_index = [int]$Row.sample_index
            target_index = [int]$Row.target_index
            keeper_prediction_index = [int]$Row.prediction_index
            xai_category = $Category
        }
    }
    $CohortRows | Export-Csv -LiteralPath $CohortCsv -NoTypeInformation -Encoding UTF8
}

if (-not (Test-Path -LiteralPath $KeeperSubset)) {
    New-Item -ItemType Directory -Path $KeeperSubset -Force | Out-Null
    $ExpectedIndices = @((Import-Csv -LiteralPath $CohortCsv) | ForEach-Object { [int]$_.sample_index })
    $KeeperCases = @($KeeperXaiPayload.cases | Where-Object { [int]$_.sample_index -in $ExpectedIndices })
    if ($KeeperCases.Count -ne 12) {
        throw "Keeper XAI summary does not cover the exact 12-case cohort."
    }
    $Counter = 0
    foreach ($Case in $KeeperCases) {
        $Counter += 1
        $TargetDir = Join-Path $KeeperSubset ("case_{0:d3}" -f $Counter)
        New-Item -ItemType Directory -Path $TargetDir -Force | Out-Null
        Copy-Item -LiteralPath (Join-Path ([string]$Case.case_dir) "case.json") -Destination (Join-Path $TargetDir "case.json")
    }
}

if (-not (Test-CompletedJson (Join-Path $CandidateXai "xai_audit_summary.json"))) {
    Invoke-NativePython "candidate_xai" @(
        "-m", "trkh.evaluation.xai_audit",
        "--checkpoint", $Checkpoint,
        "--data", $DataYaml,
        "--split", "val",
        "--class-name-mode", "raw",
        "--expected-num-classes", "5",
        "--case-csv", $CohortCsv,
        "--max-cases", "12",
        "--mistake-cases", "0",
        "--low-confidence-cases", "0",
        "--close-margin-cases", "0",
        "--per-class-cases", "0",
        "--method", "all",
        "--robustness-probes",
        "--bbox-token-prior-source", "bbox",
        "--disable-amp",
        "--batch-size", "32",
        "--num-workers", "0",
        "--output-dir", $CandidateXai
    )
}

$CandidateXaiPayload = Get-Content -Raw -LiteralPath (Join-Path $CandidateXai "xai_audit_summary.json") | ConvertFrom-Json
$CandidateByIndex = @{}
foreach ($Case in @($CandidateXaiPayload.cases)) {
    $CandidateByIndex[[string][int]$Case.sample_index] = [int]$Case.prediction_index
}
$UpdatedCohort = foreach ($Row in @(Import-Csv -LiteralPath $CohortCsv)) {
    $Key = [string][int]$Row.sample_index
    if (-not $CandidateByIndex.ContainsKey($Key)) {
        throw "Candidate XAI is missing sample_index=$Key"
    }
    [pscustomobject]@{
        sample_index = [int]$Row.sample_index
        target_index = [int]$Row.target_index
        keeper_prediction_index = [int]$Row.keeper_prediction_index
        candidate_prediction_index = [int]$CandidateByIndex[$Key]
        xai_category = [string]$Row.xai_category
    }
}
$UpdatedCohort | Export-Csv -LiteralPath $CohortCsv -NoTypeInformation -Encoding UTF8

if (-not (Test-CompletedJson (Join-Path $PairedXai "summary.json"))) {
    Invoke-NativePython "paired_xai" @(
        "-m", "trkh.tools.audit_paired_xai_cohort",
        "--cohort", $CohortCsv,
        "--left-dir", $KeeperSubset,
        "--right-dir", $CandidateXai,
        "--left-name", "keeper",
        "--right-name", "candidate",
        "--expected-cases", "12",
        "--output-dir", $PairedXai
    )
}

if (-not (Test-CompletedJson (Join-Path $GateDir "summary.json"))) {
    Invoke-NativePython "stage_b_gate" @(
        "-m", "trkh.tools.audit_deferred_reweight_stage_b",
        "--metrics", $Metrics,
        "--predictions", $Predictions,
        "--trace", $Trace,
        "--robustness", (Join-Path $Robustness "robustness_summary.json"),
        "--keeper-robustness", $KeeperRobustness,
        "--paired-xai", (Join-Path $PairedXai "summary.json"),
        "--history", $History,
        "--checkpoint", $Checkpoint,
        "--cohort", $CohortCsv,
        "--output-dir", $GateDir
    )
}

$Gate = Get-Content -Raw -LiteralPath (Join-Path $GateDir "summary.json") | ConvertFrom-Json
[ordered]@{
    status = "completed"
    decision = [string]$Gate.decision
    all_gates_passed = [bool]$Gate.all_gates_passed
    test_used = $false
    raw_dataset_modified = $false
} | ConvertTo-Json | Set-Content -LiteralPath $PostStatusPath -Encoding UTF8
Write-Host "Deferred-reweight Stage-B post-smoke audits completed: $PostRoot"
