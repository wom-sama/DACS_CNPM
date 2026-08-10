param(
    [string]$Python = "D:\DataAI\.venv\Scripts\python.exe",
    [string]$SourceLauncherArgs = "runs\full_v8_yolof_randominit_30e_20260714_105524\launcher_args.json",
    [string]$KeeperPredictions = "runs\eval_yolof_teacherfocusbinary015_best_val_20260702\predictions_detailed.csv",
    [string]$StageASummary = "runs\audit_spectral_decoupling_stage_a_v2_20260720\summary.json",
    [string]$WorkerBenchmark = "runs\benchmark_spectral_decoupling_workers_20260720\summary.json",
    [string]$ControlRunName = "smoke_spectral_decoupling_control_ce_120b_5e_det_20260720",
    [string]$CandidateRunName = "smoke_spectral_decoupling_candidate_l001_120b_5e_det_20260720",
    [string]$PairOutputDir = "runs\audit_spectral_decoupling_matched_smoke_pair_20260720",
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
$ProtocolDocument = "docs\TRKH_5CLASS_SPECTRAL_DECOUPLING_A0_PROTOCOL_20260720.md"
$LockedSourceSha256 = "a493309e7f3f900daf13b5fa6cdd334d36adf4285dd007814c761c350753fad6"
$LockedKeeperPredictionsSha256 = "9a3ee6b7ee1ab53ad8c140a864789d0dc29029ac17c78b55bca594c6af66a088"
$LockedStageASha256 = "c2daaa7d4133b64fd7c6bbf3ba6083805feca5679de35ab94c7c56238e91d348"
$LockedWorkerBenchmarkSha256 = "b55552f02ed51ffab0925f3e618fc9ad142e6dbde1de3917e4e38321402cc90a"
$LockedDataSha256 = "716e33df24c63a9e9920f97b685199707fb84ab4c7154544f5dd9a3e00d884ef"
$LockedProtocolDocumentSha256 = "fe7c14ee923f2b60ac5744831a4524ab721c61ba5f19f7745c5daa13830ac576"
$ExpectedMethod = "spectral_decoupling_multiclass_mean_lambda_half"
$ProtocolStage = "B_deterministic_scratch_120b_5e_full_validation"
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

function New-LockedTrainArguments {
    param(
        [object[]]$SourceArguments,
        [string]$RunName,
        [string]$ClassificationLoss
    )
    $Arguments = [System.Collections.Generic.List[string]]::new()
    foreach ($Item in $SourceArguments) {
        $Arguments.Add([string]$Item)
    }
    Set-ArgumentValue $Arguments "--run-name" $RunName
    Set-ArgumentValue $Arguments "--output-dir" "runs"
    Set-ArgumentValue $Arguments "--seed" "42"
    Set-ArgumentValue $Arguments "--batch-size" "32"
    Set-ArgumentValue $Arguments "--grad-accum-steps" "2"
    Set-ArgumentValue $Arguments "--epochs" "5"
    Set-ArgumentValue $Arguments "--scheduler-total-epochs" "30"
    Set-ArgumentValue $Arguments "--patience" "3"
    Set-ArgumentValue $Arguments "--max-train-batches" "120"
    Set-ArgumentValue $Arguments "--max-val-batches" "0"
    Set-ArgumentValue $Arguments "--learning-rate" "0.00025"
    Set-ArgumentValue $Arguments "--min-learning-rate" "0.000001"
    Set-ArgumentValue $Arguments "--warmup-epochs" "1"
    Set-ArgumentValue $Arguments "--weight-decay" "0"
    Set-ArgumentValue $Arguments "--label-smoothing" "0"
    Set-ArgumentValue $Arguments "--classification-loss" $ClassificationLoss
    Set-ArgumentValue $Arguments "--spectral-decoupling-lambda" "0.01"
    Set-ArgumentValue $Arguments "--bbox-token-prior-source" "crop_bbox"
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
    Remove-Argument $Arguments "--deterministic" $false
    $Arguments.Add("--deterministic")
    foreach ($RequiredFlag in @(
        "--no-pretrained",
        "--no-pretrained-distillation",
        "--disable-resume",
        "--skip-final-test",
        "--token-pruning",
        "--trace-architecture",
        "--deterministic"
    )) {
        if (-not $Arguments.Contains($RequiredFlag)) {
            throw "Locked source arguments are missing $RequiredFlag."
        }
    }
    foreach ($ForbiddenFlag in @("--pretrained", "--resume")) {
        if ($Arguments.Contains($ForbiddenFlag)) {
            throw "Locked arguments unexpectedly contain $ForbiddenFlag."
        }
    }
    return $Arguments
}

function Normalize-MatchedArguments {
    param([object[]]$Arguments)
    $Comparable = @($Arguments)
    foreach ($Name in @("--run-name", "--classification-loss")) {
        $Index = [Array]::IndexOf($Comparable, $Name)
        if ($Index -lt 0 -or $Index + 1 -ge $Comparable.Count) {
            throw "Normalized arguments do not contain a value for $Name."
        }
        $Comparable[$Index + 1] = "<matched-role>"
    }
    return $Comparable
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
$KeeperPredictionsSha256 = Assert-FileSha256 $KeeperPredictions $LockedKeeperPredictionsSha256
$StageASha256 = Assert-FileSha256 $StageASummary $LockedStageASha256
$WorkerBenchmarkSha256 = Assert-FileSha256 $WorkerBenchmark $LockedWorkerBenchmarkSha256
$DataSha256 = Assert-FileSha256 $DataYaml $LockedDataSha256
$ProtocolDocumentSha256 = Assert-FileSha256 $ProtocolDocument $LockedProtocolDocumentSha256
if (-not (Test-Path -LiteralPath $Python -PathType Leaf)) {
    throw "Python executable not found: $Python"
}

$StageA = Get-Content -Raw -LiteralPath $StageASummary | ConvertFrom-Json
if ($StageA.method -ne $ExpectedMethod -or -not [bool]$StageA.gate.smoke_permission) {
    throw "Stage-A evidence does not authorize the Spectral Decoupling pair."
}
if ([bool]$StageA.sources.validation_loaded -or [bool]$StageA.sources.test_loaded) {
    throw "Stage-A evidence is not train-only."
}
if ([string]$StageA.sources.protocol_sha256 -ne $LockedProtocolDocumentSha256) {
    throw "Stage-A protocol hash does not match the locked document."
}

$Benchmark = Get-Content -Raw -LiteralPath $WorkerBenchmark | ConvertFrom-Json
$BestWorkers = $Benchmark.best_by_samples_per_second
if (
    [string]$Benchmark.device -ne "cuda" -or
    [int]$Benchmark.dataset_samples -ne 9215 -or
    [int]$Benchmark.batch_size -ne 32 -or
    [int]$Benchmark.image_size -ne 256 -or
    -not [bool]$Benchmark.disable_pin_memory -or
    -not [bool]$Benchmark.disable_persistent_workers -or
    [int]$BestWorkers.requested_workers -ne $NumWorkers -or
    [int]$BestWorkers.effective_workers -ne $NumWorkers -or
    [bool]$BestWorkers.pin_memory -or
    [bool]$BestWorkers.persistent_workers
) {
    throw "Worker benchmark does not authorize the locked train loader."
}

$Source = Get-Content -Raw -LiteralPath $SourceLauncherArgs | ConvertFrom-Json
$ControlArguments = New-LockedTrainArguments @($Source.train_args) $ControlRunName "cross_entropy"
$CandidateArguments = New-LockedTrainArguments @($Source.train_args) $CandidateRunName "spectral_decoupling"
$ComparableControl = Normalize-MatchedArguments @($ControlArguments)
$ComparableCandidate = Normalize-MatchedArguments @($CandidateArguments)
if (($ComparableControl | ConvertTo-Json -Compress) -cne ($ComparableCandidate | ConvertTo-Json -Compress)) {
    throw "Control and candidate arguments differ beyond the locked loss role."
}

$Protocol = [ordered]@{
    method = $ExpectedMethod
    protocol_stage = $ProtocolStage
    protocol_document = (Resolve-Path $ProtocolDocument).Path
    protocol_document_sha256 = $ProtocolDocumentSha256
    source_launcher_args = (Resolve-Path $SourceLauncherArgs).Path
    source_launcher_args_sha256 = $SourceSha256
    keeper_predictions = (Resolve-Path $KeeperPredictions).Path
    keeper_predictions_sha256 = $KeeperPredictionsSha256
    stage_a_summary = (Resolve-Path $StageASummary).Path
    stage_a_summary_sha256 = $StageASha256
    worker_benchmark = (Resolve-Path $WorkerBenchmark).Path
    worker_benchmark_sha256 = $WorkerBenchmarkSha256
    data_yaml = $DataYaml
    data_yaml_sha256 = $DataSha256
    test_allowed = $false
    seed = 42
    deterministic = $true
    epochs = 5
    max_train_batches = 120
    max_val_batches = 0
    scheduler_total_epochs = 30
    batch_size = 32
    grad_accum_steps = 2
    learning_rate = 0.00025
    min_learning_rate = 0.000001
    warmup_epochs = 1
    patience = 3
    weight_decay = 0.0
    label_smoothing = 0.0
    spectral_decoupling_lambda = 0.01
    bbox_token_prior_source = "crop_bbox"
    num_workers = $NumWorkers
    eval_num_workers = $EvalNumWorkers
    control_run_name = $ControlRunName
    candidate_run_name = $CandidateRunName
    control_arguments = @($ControlArguments)
    candidate_arguments = @($CandidateArguments)
}

if ($PreflightOnly) {
    [ordered]@{status = "ok"; mode = "preflight_only"; protocol = $Protocol} |
        ConvertTo-Json -Depth 8
    exit 0
}

foreach ($Path in @(
    (Join-Path "runs" $ControlRunName),
    (Join-Path "runs" $CandidateRunName),
    $PairOutputDir
)) {
    if (Test-Path -LiteralPath $Path) {
        throw "Refusing to overwrite existing smoke artifact: $Path"
    }
}
New-Item -ItemType Directory -Path $PairOutputDir -Force | Out-Null
$ProtocolPath = Join-Path $PairOutputDir "locked_protocol.json"
$Protocol | ConvertTo-Json -Depth 8 | Set-Content -LiteralPath $ProtocolPath -Encoding UTF8
$StatusPath = Join-Path $PairOutputDir "status.json"
$Status = [ordered]@{
    status = "running"
    control = "pending"
    candidate = "pending"
    control_validation = "pending"
    candidate_validation = "pending"
    comparison = "pending"
}
$Status | ConvertTo-Json | Set-Content -LiteralPath $StatusPath -Encoding UTF8

Invoke-NativePython "train_control" (@("-m", "trkh.training.train") + @($ControlArguments))
$Status.control = "completed"
$Status | ConvertTo-Json | Set-Content -LiteralPath $StatusPath -Encoding UTF8
Invoke-NativePython "train_candidate" (@("-m", "trkh.training.train") + @($CandidateArguments))
$Status.candidate = "completed"
$Status | ConvertTo-Json | Set-Content -LiteralPath $StatusPath -Encoding UTF8

$ControlRunDir = Join-Path "runs" $ControlRunName
$CandidateRunDir = Join-Path "runs" $CandidateRunName
$ControlVal = Join-Path $PairOutputDir "control_val"
$CandidateVal = Join-Path $PairOutputDir "candidate_val"
$EvaluationBase = @(
    "-m", "trkh.evaluation.evaluate",
    "--data", $DataYaml,
    "--class-name-mode", "raw",
    "--expected-num-classes", "5",
    "--split", "val",
    "--batch-size", "64",
    "--num-workers", "$EvalNumWorkers",
    "--amp",
    "--bbox-token-prior-source", "crop_bbox",
    "--family", "TRKH-spectral-decoupling-matched-smoke"
)
Invoke-NativePython "evaluate_control_val" ($EvaluationBase + @(
    "--checkpoint", (Join-Path $ControlRunDir "checkpoints\best.pt"),
    "--paper-name", "TRKH-spectral-decoupling-control-CE-120b-5e",
    "--output-dir", $ControlVal
))
$Status.control_validation = "completed"
$Status | ConvertTo-Json | Set-Content -LiteralPath $StatusPath -Encoding UTF8
Invoke-NativePython "evaluate_candidate_val" ($EvaluationBase + @(
    "--checkpoint", (Join-Path $CandidateRunDir "checkpoints\best.pt"),
    "--paper-name", "TRKH-spectral-decoupling-lambda001-120b-5e",
    "--output-dir", $CandidateVal
))
$Status.candidate_validation = "completed"
$Status | ConvertTo-Json | Set-Content -LiteralPath $StatusPath -Encoding UTF8

Invoke-NativePython "compare_smoke_pair" @(
    "-m", "trkh.tools.audit_spectral_decoupling_smoke_pair",
    "--keeper-predictions", $KeeperPredictions,
    "--control-predictions", (Join-Path $ControlVal "predictions_detailed.csv"),
    "--candidate-predictions", (Join-Path $CandidateVal "predictions_detailed.csv"),
    "--control-run-summary", (Join-Path $ControlRunDir "summary.json"),
    "--candidate-run-summary", (Join-Path $CandidateRunDir "summary.json"),
    "--control-resolved-config", (Join-Path $ControlRunDir "resolved_config.json"),
    "--candidate-resolved-config", (Join-Path $CandidateRunDir "resolved_config.json"),
    "--stage-a-summary", $StageASummary,
    "--locked-protocol", $ProtocolPath,
    "--output-dir", (Join-Path $PairOutputDir "comparison")
)
$Status.comparison = "completed"
$Status.status = "completed"
$Status | ConvertTo-Json | Set-Content -LiteralPath $StatusPath -Encoding UTF8
Write-Host "Matched Spectral Decoupling smoke pair completed: $PairOutputDir"
