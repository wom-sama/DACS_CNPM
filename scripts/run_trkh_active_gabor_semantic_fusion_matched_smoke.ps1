param(
    [string]$Python = "D:\DataAI\.venv\Scripts\python.exe",
    [string]$SourceLauncherArgs = "runs\full_v8_yolof_randominit_30e_20260714_105524\launcher_args.json",
    [string]$KeeperPredictions = "runs\eval_yolof_teacherfocusbinary015_best_val_20260702\predictions_detailed.csv",
    [string]$StageASummary = "runs\audit_active_gabor_semantic_fusion_stage_a_20260715\summary.json",
    [string]$ControlRunName = "smoke_active_gabor_semantic_control_120b_5e_det_20260715",
    [string]$CandidateRunName = "smoke_active_gabor_semantic_candidate_120b_5e_det_20260715",
    [string]$PairOutputDir = "runs\audit_active_gabor_semantic_fusion_matched_smoke_pair_20260715",
    [switch]$PreflightOnly
)

$ErrorActionPreference = "Stop"
$RepoRoot = (Resolve-Path (Join-Path $PSScriptRoot "..")).Path
Set-Location $RepoRoot

$DataYaml = "D:\DataAI\AIEx\newdataset\yolo_f\data.yaml"
$ProtocolDocument = "docs\TRKH_5CLASS_ACTIVE_GABOR_SEMANTIC_FUSION_PROTOCOL_20260715.md"
$LockedSourceSha256 = "a493309e7f3f900daf13b5fa6cdd334d36adf4285dd007814c761c350753fad6"
$LockedKeeperPredictionsSha256 = "9a3ee6b7ee1ab53ad8c140a864789d0dc29029ac17c78b55bca594c6af66a088"
$LockedStageASha256 = "0917424934b9299c6efb5b04b592376be0f1e196fd7fd62beb9bdfe4e1fefea3"
$LockedDataSha256 = "716e33df24c63a9e9920f97b685199707fb84ab4c7154544f5dd9a3e00d884ef"
$LockedProtocolDocumentSha256 = "6a1c810f47baea5728a076da98b1410bb5ea4ddd0089ffd4e05150bda6e47afe"
$ExpectedMethod = "active_gabor_lho_fcm_semantic_sum"
$ProtocolStage = "B_deterministic_scratch_120b_5e_full_validation"
$CandidateFlag = "--learnable-gabor-texture-semantic-fusion"

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
        [bool]$EnableActiveGabor
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
    Set-ArgumentValue $Arguments "--weight-decay" "0.05"
    Set-ArgumentValue $Arguments "--bbox-token-prior-source" "crop_bbox"

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
    Remove-Argument $Arguments "--learnable-gabor-texture-residual" $false
    Remove-Argument $Arguments $CandidateFlag $false
    Remove-Argument $Arguments "--deterministic" $false
    $Arguments.Add("--deterministic")
    if ($EnableActiveGabor) {
        $Arguments.Add($CandidateFlag)
    }
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

$SourceSha256 = Assert-FileSha256 $SourceLauncherArgs $LockedSourceSha256
$KeeperPredictionsSha256 = Assert-FileSha256 $KeeperPredictions $LockedKeeperPredictionsSha256
$StageASha256 = Assert-FileSha256 $StageASummary $LockedStageASha256
$DataSha256 = Assert-FileSha256 $DataYaml $LockedDataSha256
$ProtocolDocumentSha256 = Assert-FileSha256 $ProtocolDocument $LockedProtocolDocumentSha256
if (-not (Test-Path -LiteralPath $Python -PathType Leaf)) {
    throw "Python executable not found: $Python"
}

$StageA = Get-Content -Raw -LiteralPath $StageASummary | ConvertFrom-Json
if ($StageA.method -ne $ExpectedMethod -or -not [bool]$StageA.gate.smoke_permission) {
    throw "Stage-A evidence does not authorize the active Gabor smoke."
}
if ([bool]$StageA.sources.validation_loaded -or [bool]$StageA.sources.test_loaded) {
    throw "Stage-A evidence is not train-only."
}
foreach ($RequiredCheck in @(
    "candidate_rng_neutral",
    "active_direct_sum_exact",
    "micro_representation_diversifies"
)) {
    if (-not [bool]$StageA.gate.checks.$RequiredCheck) {
        throw "Stage-A prerequisite did not pass: $RequiredCheck"
    }
}

$Source = Get-Content -Raw -LiteralPath $SourceLauncherArgs | ConvertFrom-Json
$ControlArguments = New-LockedTrainArguments @($Source.train_args) $ControlRunName $false
$CandidateArguments = New-LockedTrainArguments @($Source.train_args) $CandidateRunName $true
$ComparableCandidate = @($CandidateArguments | Where-Object { $_ -ne $CandidateFlag })
$ComparableControl = @($ControlArguments)
$ControlRunIndex = [Array]::IndexOf($ComparableControl, "--run-name")
$CandidateRunIndex = [Array]::IndexOf($ComparableCandidate, "--run-name")
if ($ControlRunIndex -lt 0 -or $CandidateRunIndex -lt 0) {
    throw "Normalized arguments do not contain --run-name."
}
$ComparableControl[$ControlRunIndex + 1] = "<matched-run>"
$ComparableCandidate[$CandidateRunIndex + 1] = "<matched-run>"
if (($ComparableControl | ConvertTo-Json -Compress) -ne ($ComparableCandidate | ConvertTo-Json -Compress)) {
    throw "Control and candidate arguments differ beyond the active Gabor flag."
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
    data_yaml = $DataYaml
    data_yaml_sha256 = $DataSha256
    test_allowed = $false
    candidate_only_argument = $CandidateFlag
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
    weight_decay = 0.05
    bbox_token_prior_source = "crop_bbox"
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
    "--num-workers", "2",
    "--amp",
    "--bbox-token-prior-source", "crop_bbox",
    "--family", "TRKH-active-Gabor-semantic-sum-smoke"
)
Invoke-NativePython "evaluate_control_val" ($EvaluationBase + @(
    "--checkpoint", (Join-Path $ControlRunDir "checkpoints\best.pt"),
    "--paper-name", "TRKH-active-Gabor-control-120b-5e",
    "--output-dir", $ControlVal
))
$Status.control_validation = "completed"
$Status | ConvertTo-Json | Set-Content -LiteralPath $StatusPath -Encoding UTF8
Invoke-NativePython "evaluate_candidate_val" ($EvaluationBase + @(
    "--checkpoint", (Join-Path $CandidateRunDir "checkpoints\best.pt"),
    "--paper-name", "TRKH-active-Gabor-semantic-sum-120b-5e",
    "--output-dir", $CandidateVal
))
$Status.candidate_validation = "completed"
$Status | ConvertTo-Json | Set-Content -LiteralPath $StatusPath -Encoding UTF8

Invoke-NativePython "compare_smoke_pair" @(
    "-m", "trkh.tools.audit_active_gabor_semantic_fusion_smoke_pair",
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
Write-Host "Matched active-Gabor smoke pair completed: $PairOutputDir"
