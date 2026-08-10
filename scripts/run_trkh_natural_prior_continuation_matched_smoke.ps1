param(
    [string]$Python = "D:\DataAI\.venv\Scripts\python.exe",
    [string]$SourceLauncherArgs = "runs\full_v8_yolof_randominit_30e_20260714_105524\launcher_args.json",
    [string]$ResumeCheckpoint = "runs\full_v8_yolof_randominit_30e_20260714_105524\checkpoints\best.pt",
    [string]$ScratchPredictions = "runs\full_v8_yolof_randominit_30e_20260714_105524\final_audit_manual\val_raw_fp32\predictions_detailed.csv",
    [string]$KeeperPredictions = "runs\eval_yolof_teacherfocusbinary015_best_val_20260702\predictions_detailed.csv",
    [string]$ControlRunName = "smoke_natural_prior_control_strict_60b_2e_20260715",
    [string]$CandidateRunName = "smoke_natural_prior_candidate_random_60b_2e_20260715",
    [string]$PairOutputDir = "runs\audit_natural_prior_continuation_matched_smoke_20260715",
    [switch]$PreflightOnly
)

$ErrorActionPreference = "Stop"
$RepoRoot = (Resolve-Path (Join-Path $PSScriptRoot "..")).Path
Set-Location $RepoRoot

$DataYaml = "D:\DataAI\AIEx\newdataset\yolo_f\data.yaml"
$ProtocolDocument = "docs\TRKH_5CLASS_NATURAL_PRIOR_CONTINUATION_PROTOCOL_20260715.md"
$LockedDataSha256 = "716e33df24c63a9e9920f97b685199707fb84ab4c7154544f5dd9a3e00d884ef"
$LockedSourceSha256 = "a493309e7f3f900daf13b5fa6cdd334d36adf4285dd007814c761c350753fad6"
$LockedCheckpointSha256 = "f8bd6309b9820d2b2d6db28815a9a01e11ebbaf0bd5eb6c5ce097b6aa081a549"
$LockedScratchPredictionsSha256 = "079ff545df0d0272fd2640453715621843da2b16653293d52b4fc1c2688ff0d8"
$LockedKeeperPredictionsSha256 = "9a3ee6b7ee1ab53ad8c140a864789d0dc29029ac17c78b55bca594c6af66a088"
$LockedProtocolDocumentSha256 = "5922d0c5258de029acc413538c0b61785957f077a116243480c855c35376ec20"
$Method = "natural_prior_full_model_continuation"
$ProtocolStage = "matched_60b_2e_full_validation"
$CandidateFlag = "--disable-balanced-epoch-sampling"

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
        [bool]$UseNaturalPrior
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
    Set-ArgumentValue $Arguments "--epochs" "2"
    Set-ArgumentValue $Arguments "--scheduler-total-epochs" "10"
    Set-ArgumentValue $Arguments "--patience" "2"
    Set-ArgumentValue $Arguments "--max-train-batches" "60"
    Set-ArgumentValue $Arguments "--max-val-batches" "0"
    Set-ArgumentValue $Arguments "--learning-rate" "0.00008"
    Set-ArgumentValue $Arguments "--min-learning-rate" "0.000001"
    Set-ArgumentValue $Arguments "--warmup-epochs" "1"
    Set-ArgumentValue $Arguments "--weight-decay" "0.05"

    Remove-Argument $Arguments "--disable-resume" $false
    Remove-Argument $Arguments "--resume" $true
    foreach ($Flag in @(
        "--resume-use-cli-config",
        "--resume-reset-epoch",
        "--resume-reset-optimizer",
        "--resume-reset-scheduler",
        "--resume-reset-scaler",
        "--deterministic",
        $CandidateFlag
    )) {
        Remove-Argument $Arguments $Flag $false
    }
    $Arguments.Add("--resume")
    $Arguments.Add([System.IO.Path]::GetFullPath($ResumeCheckpoint))
    foreach ($Flag in @(
        "--resume-use-cli-config",
        "--resume-reset-epoch",
        "--resume-reset-optimizer",
        "--resume-reset-scheduler",
        "--resume-reset-scaler",
        "--deterministic"
    )) {
        $Arguments.Add($Flag)
    }
    if ($UseNaturalPrior) {
        $Arguments.Add($CandidateFlag)
    }
    foreach ($RequiredFlag in @(
        "--no-pretrained",
        "--no-pretrained-distillation",
        "--skip-final-test",
        "--trace-architecture",
        "--resume",
        "--resume-use-cli-config",
        "--resume-reset-epoch",
        "--resume-reset-optimizer",
        "--resume-reset-scheduler",
        "--resume-reset-scaler",
        "--deterministic"
    )) {
        if (-not $Arguments.Contains($RequiredFlag)) {
            throw "Locked arguments are missing $RequiredFlag."
        }
    }
    if ($Arguments.Contains("--disable-resume")) {
        throw "Locked continuation still contains --disable-resume."
    }
    return $Arguments
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

$DataSha256 = Assert-FileSha256 $DataYaml $LockedDataSha256
$SourceSha256 = Assert-FileSha256 $SourceLauncherArgs $LockedSourceSha256
$CheckpointSha256 = Assert-FileSha256 $ResumeCheckpoint $LockedCheckpointSha256
$ScratchPredictionsSha256 = Assert-FileSha256 $ScratchPredictions $LockedScratchPredictionsSha256
$KeeperPredictionsSha256 = Assert-FileSha256 $KeeperPredictions $LockedKeeperPredictionsSha256
$ProtocolDocumentSha256 = Assert-FileSha256 $ProtocolDocument $LockedProtocolDocumentSha256
if (-not (Test-Path -LiteralPath $Python -PathType Leaf)) {
    throw "Python executable not found: $Python"
}

$Source = Get-Content -Raw -LiteralPath $SourceLauncherArgs | ConvertFrom-Json
$ControlArguments = New-LockedTrainArguments @($Source.train_args) $ControlRunName $false
$CandidateArguments = New-LockedTrainArguments @($Source.train_args) $CandidateRunName $true
$ComparableControl = @($ControlArguments)
$ComparableCandidate = @($CandidateArguments | Where-Object { $_ -ne $CandidateFlag })
$ControlRunIndex = [Array]::IndexOf($ComparableControl, "--run-name")
$CandidateRunIndex = [Array]::IndexOf($ComparableCandidate, "--run-name")
if ($ControlRunIndex -lt 0 -or $CandidateRunIndex -lt 0) {
    throw "Normalized arguments do not contain --run-name."
}
$ComparableControl[$ControlRunIndex + 1] = "<matched-run>"
$ComparableCandidate[$CandidateRunIndex + 1] = "<matched-run>"
if (($ComparableControl | ConvertTo-Json -Compress) -cne ($ComparableCandidate | ConvertTo-Json -Compress)) {
    throw "Control and candidate differ beyond the natural-prior flag."
}

$Protocol = [ordered]@{
    method = $Method
    protocol_stage = $ProtocolStage
    protocol_document = (Resolve-Path $ProtocolDocument).Path
    protocol_document_sha256 = $ProtocolDocumentSha256
    source_launcher_args = (Resolve-Path $SourceLauncherArgs).Path
    source_launcher_args_sha256 = $SourceSha256
    resume_checkpoint = (Resolve-Path $ResumeCheckpoint).Path
    resume_checkpoint_sha256 = $CheckpointSha256
    scratch_predictions = (Resolve-Path $ScratchPredictions).Path
    scratch_predictions_sha256 = $ScratchPredictionsSha256
    keeper_predictions = (Resolve-Path $KeeperPredictions).Path
    keeper_predictions_sha256 = $KeeperPredictionsSha256
    data_yaml = $DataYaml
    data_yaml_sha256 = $DataSha256
    test_allowed = $false
    candidate_only_argument = $CandidateFlag
    seed = 42
    deterministic = $true
    epochs = 2
    source_checkpoint_epoch = 20
    maximum_effective_epoch = 22
    max_train_batches = 60
    max_val_batches = 0
    scheduler_total_epochs = 10
    batch_size = 32
    grad_accum_steps = 2
    learning_rate = 0.00008
    min_learning_rate = 0.000001
    warmup_epochs = 1
    patience = 2
    weight_decay = 0.05
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
    test_used = $false
    control = "pending"
    candidate = "pending"
    control_validation = "pending"
    candidate_validation = "pending"
    comparison = "pending"
}
$Status | ConvertTo-Json | Set-Content -LiteralPath $StatusPath -Encoding UTF8

$env:PYTHONPATH = $RepoRoot
$env:TRKH_AMP_DTYPE = "bf16"
$env:OMP_NUM_THREADS = "4"
$env:TRKH_ALLOW_WINDOWS_MULTIPROCESSING = "1"
Invoke-NativePython "train_control" (@("-m", "trkh.training.train") + @($ControlArguments))
$Status.control = "completed"
$Status | ConvertTo-Json | Set-Content -LiteralPath $StatusPath -Encoding UTF8
Invoke-NativePython "train_candidate" (@("-m", "trkh.training.train") + @($CandidateArguments))
$Status.candidate = "completed"
$Status | ConvertTo-Json | Set-Content -LiteralPath $StatusPath -Encoding UTF8

$ControlRunDir = Join-Path "runs" $ControlRunName
$CandidateRunDir = Join-Path "runs" $CandidateRunName
$ControlVal = Join-Path $PairOutputDir "control_val_fp32"
$CandidateVal = Join-Path $PairOutputDir "candidate_val_fp32"
$EvaluationBase = @(
    "-m", "trkh.evaluation.evaluate",
    "--data", $DataYaml,
    "--class-name-mode", "raw",
    "--expected-num-classes", "5",
    "--split", "val",
    "--batch-size", "64",
    "--num-workers", "2",
    "--bbox-token-prior-source", "bbox",
    "--family", "TRKH-natural-prior-continuation-smoke"
)
Invoke-NativePython "evaluate_control_val_fp32" ($EvaluationBase + @(
    "--checkpoint", (Join-Path $ControlRunDir "checkpoints\best.pt"),
    "--paper-name", "TRKH-natural-prior-control-strict-60b-2e",
    "--output-dir", $ControlVal
))
$Status.control_validation = "completed"
$Status | ConvertTo-Json | Set-Content -LiteralPath $StatusPath -Encoding UTF8
Invoke-NativePython "evaluate_candidate_val_fp32" ($EvaluationBase + @(
    "--checkpoint", (Join-Path $CandidateRunDir "checkpoints\best.pt"),
    "--paper-name", "TRKH-natural-prior-candidate-random-60b-2e",
    "--output-dir", $CandidateVal
))
$Status.candidate_validation = "completed"
$Status | ConvertTo-Json | Set-Content -LiteralPath $StatusPath -Encoding UTF8

Invoke-NativePython "compare_natural_prior_pair" @(
    "-m", "trkh.tools.audit_natural_prior_continuation_smoke_pair",
    "--keeper-predictions", $KeeperPredictions,
    "--scratch-predictions", $ScratchPredictions,
    "--control-predictions", (Join-Path $ControlVal "predictions_detailed.csv"),
    "--candidate-predictions", (Join-Path $CandidateVal "predictions_detailed.csv"),
    "--control-run-summary", (Join-Path $ControlRunDir "summary.json"),
    "--candidate-run-summary", (Join-Path $CandidateRunDir "summary.json"),
    "--control-resolved-config", (Join-Path $ControlRunDir "resolved_config.json"),
    "--candidate-resolved-config", (Join-Path $CandidateRunDir "resolved_config.json"),
    "--locked-protocol", $ProtocolPath,
    "--output-dir", (Join-Path $PairOutputDir "comparison")
)
$Status.comparison = "completed"
$Status.status = "completed"
$Status | ConvertTo-Json | Set-Content -LiteralPath $StatusPath -Encoding UTF8
Write-Host "Natural-prior matched smoke completed: $PairOutputDir"
