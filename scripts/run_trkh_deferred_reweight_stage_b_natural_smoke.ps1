[CmdletBinding()]
param(
    [string]$Python = "D:\DataAI\.venv\Scripts\python.exe",
    [string]$SourceLauncherArgs = "runs\full_v8_yolof_randominit_30e_20260714_105524\launcher_args.json",
    [string]$A0Summary = "runs\audit_deferred_reweight_prior_a0_20260720\summary.json",
    [string]$WorkerBenchmark = "runs\evidence_spectral_decoupling_rejected_20260720\worker_benchmark\summary.json",
    [string]$RunName = "smoke_v8_yolof_natural_prior_randominit_5e_20260720",
    [string]$AuditOutputDir = "runs\audit_deferred_reweight_stage_b_natural_smoke_20260720",
    [switch]$PreflightOnly
)

$ErrorActionPreference = "Stop"
$RepoRoot = (Resolve-Path (Join-Path $PSScriptRoot "..")).Path
Set-Location $RepoRoot
$env:PYTHONPATH = $RepoRoot
$env:TRKH_AMP_DTYPE = "bf16"
$env:OMP_NUM_THREADS = "4"
$env:TRKH_ALLOW_WINDOWS_MULTIPROCESSING = "1"
Remove-Item Env:TRKH_ALLOW_WINDOWS_PIN_MEMORY -ErrorAction SilentlyContinue
Remove-Item Env:TRKH_ALLOW_WINDOWS_PERSISTENT_WORKERS -ErrorAction SilentlyContinue

$DataYaml = "D:\DataAI\AIEx\newdataset\yolo_f\data.yaml"
$ProtocolDocument = "docs\TRKH_5CLASS_DEFERRED_REWEIGHT_A0_PROTOCOL_20260720.md"
$LockedSourceSha256 = "a493309e7f3f900daf13b5fa6cdd334d36adf4285dd007814c761c350753fad6"
$LockedA0SummarySha256 = "cfa0d366b58e067c168358b4a90dff41611b4151dffbd990e8b0bda1bc9176a5"
$LockedWorkerBenchmarkSha256 = "b55552f02ed51ffab0925f3e618fc9ad142e6dbde1de3917e4e38321402cc90a"
$LockedDataSha256 = "716e33df24c63a9e9920f97b685199707fb84ab4c7154544f5dd9a3e00d884ef"
$LockedProtocolSha256 = "1627a2b541dbbd27394ee2cd95e1673a7b73d978aec605d1e5939e2164c76276"
$NumWorkers = 4
$EvalNumWorkers = 2

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

function Set-ArgumentValue {
    param(
        [System.Collections.Generic.List[string]]$Arguments,
        [string]$Name,
        [string]$Value
    )
    $Index = $Arguments.IndexOf($Name)
    if ($Index -lt 0) {
        $Arguments.Add($Name)
        $Arguments.Add($Value)
        return
    }
    if ($Index + 1 -ge $Arguments.Count -or $Arguments[$Index + 1].StartsWith("--")) {
        throw "Argument $Name does not have a value in the source protocol."
    }
    $Arguments[$Index + 1] = $Value
}

function Remove-Argument {
    param(
        [System.Collections.Generic.List[string]]$Arguments,
        [string]$Name,
        [bool]$HasValue
    )
    while (($Index = $Arguments.IndexOf($Name)) -ge 0) {
        $Arguments.RemoveAt($Index)
        if ($HasValue) {
            if ($Index -ge $Arguments.Count -or $Arguments[$Index].StartsWith("--")) {
                throw "Argument $Name is missing its value."
            }
            $Arguments.RemoveAt($Index)
        }
    }
}

function Add-FlagOnce {
    param(
        [System.Collections.Generic.List[string]]$Arguments,
        [string]$Name
    )
    if (-not $Arguments.Contains($Name)) {
        $Arguments.Add($Name)
    }
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

$SourceSha256 = Assert-FileSha256 $SourceLauncherArgs $LockedSourceSha256
$A0SummarySha256 = Assert-FileSha256 $A0Summary $LockedA0SummarySha256
$WorkerBenchmarkSha256 = Assert-FileSha256 $WorkerBenchmark $LockedWorkerBenchmarkSha256
$DataSha256 = Assert-FileSha256 $DataYaml $LockedDataSha256
$ProtocolSha256 = Assert-FileSha256 $ProtocolDocument $LockedProtocolSha256
if (-not (Test-Path -LiteralPath $Python -PathType Leaf)) {
    throw "Python executable not found: $Python"
}

$A0 = Get-Content -Raw -LiteralPath $A0Summary | ConvertFrom-Json
if (
    -not [bool]$A0.all_gates_passed -or
    [string]$A0.authorization -ne "one_natural_only_stage_b_smoke" -or
    [string]$A0.mode -ne "deferred_reweight_prior_a0_no_training_authorization"
) {
    throw "A0 does not authorize the natural-only Stage-B smoke."
}
if (
    [bool]$A0.scope.validation_predictions_used -or
    [bool]$A0.scope.test_data_used -or
    [bool]$A0.scope.dataset_pixels_loaded -or
    [bool]$A0.scope.checkpoint_loaded -or
    [bool]$A0.scope.training_used
) {
    throw "A0 scope is not a no-training train-only authorization."
}

$Benchmark = Get-Content -Raw -LiteralPath $WorkerBenchmark | ConvertFrom-Json
$BestWorkers = $Benchmark.best_by_samples_per_second
if (
    [string]$Benchmark.device -ne "cuda" -or
    [int]$Benchmark.dataset_samples -ne 9215 -or
    [int]$Benchmark.batch_size -ne 32 -or
    [int]$Benchmark.image_size -ne 256 -or
    [int]$BestWorkers.requested_workers -ne $NumWorkers -or
    [int]$BestWorkers.effective_workers -ne $NumWorkers
) {
    throw "Worker benchmark does not authorize train num_workers=4."
}

$Source = Get-Content -Raw -LiteralPath $SourceLauncherArgs | ConvertFrom-Json
$Arguments = [System.Collections.Generic.List[string]]::new()
foreach ($Item in @($Source.train_args)) {
    $Arguments.Add([string]$Item)
}
Set-ArgumentValue $Arguments "--run-name" $RunName
Set-ArgumentValue $Arguments "--output-dir" "runs"
Set-ArgumentValue $Arguments "--epochs" "5"
Set-ArgumentValue $Arguments "--scheduler-total-epochs" "30"
Set-ArgumentValue $Arguments "--patience" "3"
Set-ArgumentValue $Arguments "--max-train-batches" "0"
Set-ArgumentValue $Arguments "--max-val-batches" "0"
Set-ArgumentValue $Arguments "--num-workers" "$NumWorkers"
Set-ArgumentValue $Arguments "--eval-num-workers" "$EvalNumWorkers"
Remove-Argument $Arguments "--resume" $true
foreach ($ResumeFlag in @(
    "--resume-use-cli-config",
    "--resume-reset-epoch",
    "--resume-reset-optimizer",
    "--resume-reset-scheduler",
    "--resume-reset-scaler"
)) {
    Remove-Argument $Arguments $ResumeFlag $false
}
Add-FlagOnce $Arguments "--disable-balanced-epoch-sampling"

foreach ($RequiredFlag in @(
    "--no-pretrained",
    "--no-pretrained-distillation",
    "--disable-resume",
    "--disable-class-weights",
    "--disable-balanced-epoch-sampling",
    "--skip-final-test",
    "--trace-architecture"
)) {
    if (-not $Arguments.Contains($RequiredFlag)) {
        throw "Locked Stage-B arguments are missing $RequiredFlag."
    }
}
foreach ($ForbiddenFlag in @("--pretrained", "--resume", "--deterministic")) {
    if ($Arguments.Contains($ForbiddenFlag)) {
        throw "Locked Stage-B arguments unexpectedly contain $ForbiddenFlag."
    }
}

$TrackedStatus = git status --porcelain --untracked-files=no
if ($LASTEXITCODE -ne 0 -or $TrackedStatus) {
    throw "Tracked worktree must be clean before Stage-B training."
}
$RepoCommit = (git rev-parse HEAD).Trim()
if ($LASTEXITCODE -ne 0) {
    throw "Unable to resolve repository HEAD."
}

$Protocol = [ordered]@{
    method = "natural_first_deferred_reweight_stage_b"
    protocol_stage = "B_natural_only_randominit_5e_full_validation"
    protocol_document = (Resolve-Path $ProtocolDocument).Path
    protocol_document_sha256 = $ProtocolSha256
    a0_summary = (Resolve-Path $A0Summary).Path
    a0_summary_sha256 = $A0SummarySha256
    source_launcher_args = (Resolve-Path $SourceLauncherArgs).Path
    source_launcher_args_sha256 = $SourceSha256
    worker_benchmark = (Resolve-Path $WorkerBenchmark).Path
    worker_benchmark_sha256 = $WorkerBenchmarkSha256
    data_yaml = $DataYaml
    data_yaml_sha256 = $DataSha256
    repository_commit = $RepoCommit
    test_allowed = $false
    deferred_reweight_allowed = $false
    seed = 42
    epochs = 5
    scheduler_total_epochs = 30
    max_train_batches = 0
    max_val_batches = 0
    batch_size = 32
    grad_accum_steps = 2
    num_workers = $NumWorkers
    eval_num_workers = $EvalNumWorkers
    natural_sampling = $true
    class_weights = $false
    trace_architecture = $true
    run_name = $RunName
    train_arguments = @($Arguments)
}

if ($PreflightOnly) {
    [ordered]@{status = "ok"; mode = "preflight_only"; protocol = $Protocol} |
        ConvertTo-Json -Depth 8
    exit 0
}

$RunDir = Join-Path "runs" $RunName
foreach ($Path in @($RunDir, $AuditOutputDir)) {
    if (Test-Path -LiteralPath $Path) {
        throw "Refusing to overwrite existing Stage-B artifact: $Path"
    }
}
New-Item -ItemType Directory -Path $AuditOutputDir | Out-Null
$ProtocolPath = Join-Path $AuditOutputDir "locked_protocol.json"
$Protocol | ConvertTo-Json -Depth 8 | Set-Content -LiteralPath $ProtocolPath -Encoding UTF8
$StatusPath = Join-Path $AuditOutputDir "status.json"
$Status = [ordered]@{
    status = "running"
    train = "pending"
    independent_validation = "pending"
    test_used = $false
}
$Status | ConvertTo-Json | Set-Content -LiteralPath $StatusPath -Encoding UTF8

Invoke-NativePython "train_natural_only" (@("-m", "trkh.training.train") + @($Arguments))
$Status.train = "completed"
$Status | ConvertTo-Json | Set-Content -LiteralPath $StatusPath -Encoding UTF8

$ValidationDir = Join-Path $AuditOutputDir "independent_val_fp32"
$ValidationArguments = @(
    "-m", "trkh.evaluation.evaluate",
    "--checkpoint", (Join-Path $RunDir "checkpoints\best.pt"),
    "--data", $DataYaml,
    "--class-name-mode", "raw",
    "--expected-num-classes", "5",
    "--split", "val",
    "--batch-size", "64",
    "--num-workers", "$EvalNumWorkers",
    "--seed", "42",
    "--output-dir", $ValidationDir,
    "--paper-name", "TRKH-natural-prior-randominit-5e",
    "--family", "TRKH-deferred-reweight-stage-b",
    "--disable-calibration"
)
Invoke-NativePython "independent_validation_fp32" $ValidationArguments
$Status.independent_validation = "completed"
$Status.status = "completed_pending_gate_audit"
$Status | ConvertTo-Json | Set-Content -LiteralPath $StatusPath -Encoding UTF8

[ordered]@{
    status = $Status.status
    run_dir = (Resolve-Path $RunDir).Path
    audit_dir = (Resolve-Path $AuditOutputDir).Path
    test_used = $false
    deferred_reweight_used = $false
} | ConvertTo-Json
