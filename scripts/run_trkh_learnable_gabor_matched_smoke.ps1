param(
    [string]$Python = "D:\DataAI\.venv\Scripts\python.exe",
    [string]$SourceLauncherArgs = "runs\probe_v8_yolof_pairroute_teacherfocusbinary015_boundarydrop_bboxprior_120b_2e_20260701\launcher_args.json",
    [string]$KeeperCheckpoint = "runs\probe_v8_yolof_pairroute_teacherfocusbinary015_boundarydrop_bboxprior_120b_2e_20260701\checkpoints\best.pt",
    [string]$KeeperPredictions = "runs\eval_yolof_teacherfocusbinary015_best_val_20260702\predictions_detailed.csv",
    [string]$ControlRunName = "smoke_gabor_lho_fcm_rngneutral_control_120b_2e_20260715",
    [string]$CandidateRunName = "smoke_gabor_lho_fcm_rngneutral_candidate_120b_2e_20260715",
    [string]$PairOutputDir = "runs\audit_gabor_lho_fcm_rngneutral_matched_smoke_pair_20260715",
    [switch]$PreflightOnly
)

$ErrorActionPreference = "Stop"
$RepoRoot = (Resolve-Path (Join-Path $PSScriptRoot "..")).Path
Set-Location $RepoRoot

$DataYaml = "D:\DataAI\AIEx\newdataset\yolo_f\data.yaml"
$StageASummary = "runs\audit_learnable_gabor_texture_stage_a_rngneutral_20260715\summary.json"
$LockedSourceSha256 = "908a05cf66b2a01162cae62e4ff2251eaae1297d31e70510144e4954159b7eff"
$LockedKeeperSha256 = "1f49d577240c69dc63c30af70db52ec2aa9da65a17aef1c4b1c09ece6c482677"
$LockedKeeperPredictionsSha256 = "9a3ee6b7ee1ab53ad8c140a864789d0dc29029ac17c78b55bca594c6af66a088"
$LockedStageASha256 = "e53d8c1ae79bad853b735bd61b94bb7bc0a5c5865964b1ec1f5baf6c71b81e8f"
$LockedDataSha256 = "716e33df24c63a9e9920f97b685199707fb84ab4c7154544f5dd9a3e00d884ef"
$ExpectedMethod = "learnable_gabor_lho_fcm_edge_token_residual"

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
        [bool]$EnableGabor
    )
    $Arguments = [System.Collections.Generic.List[string]]::new()
    foreach ($Item in $SourceArguments) {
        $Arguments.Add([string]$Item)
    }
    Set-ArgumentValue $Arguments "--run-name" $RunName
    Set-ArgumentValue $Arguments "--output-dir" "runs"
    Set-ArgumentValue $Arguments "--resume" $KeeperCheckpoint
    Set-ArgumentValue $Arguments "--seed" "42"
    Set-ArgumentValue $Arguments "--batch-size" "32"
    Set-ArgumentValue $Arguments "--grad-accum-steps" "2"
    Set-ArgumentValue $Arguments "--epochs" "2"
    Set-ArgumentValue $Arguments "--scheduler-total-epochs" "15"
    Set-ArgumentValue $Arguments "--patience" "3"
    Set-ArgumentValue $Arguments "--max-train-batches" "120"
    Set-ArgumentValue $Arguments "--max-val-batches" "0"
    Set-ArgumentValue $Arguments "--learning-rate" "0.00008"
    Set-ArgumentValue $Arguments "--min-learning-rate" "0.000001"
    Set-ArgumentValue $Arguments "--warmup-epochs" "1"
    Set-ArgumentValue $Arguments "--weight-decay" "0.05"
    Set-ArgumentValue $Arguments "--bbox-token-prior-source" "crop_bbox"
    Remove-Argument $Arguments "--disable-resume" $false
    Remove-Argument $Arguments "--learnable-gabor-texture-residual" $false
    if ($EnableGabor) {
        $Arguments.Add("--learnable-gabor-texture-residual")
    }
    foreach ($RequiredFlag in @(
        "--no-pretrained",
        "--resume-use-cli-config",
        "--resume-reset-epoch",
        "--resume-reset-optimizer",
        "--resume-reset-scheduler",
        "--resume-reset-scaler",
        "--skip-final-test",
        "--token-pruning"
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
$KeeperSha256 = Assert-FileSha256 $KeeperCheckpoint $LockedKeeperSha256
$KeeperPredictionsSha256 = Assert-FileSha256 $KeeperPredictions $LockedKeeperPredictionsSha256
$StageASha256 = Assert-FileSha256 $StageASummary $LockedStageASha256
$DataSha256 = Assert-FileSha256 $DataYaml $LockedDataSha256
if (-not (Test-Path -LiteralPath $Python -PathType Leaf)) {
    throw "Python executable not found: $Python"
}
$StageA = Get-Content -Raw -LiteralPath $StageASummary | ConvertFrom-Json
if ($StageA.method -ne $ExpectedMethod -or -not [bool]$StageA.gate.smoke_permission) {
    throw "Stage-A evidence does not authorize the locked Gabor smoke."
}
if ([bool]$StageA.sources.validation_loaded -or [bool]$StageA.sources.test_loaded) {
    throw "Stage-A evidence is not train-only."
}
if (-not [bool]$StageA.gate.checks.candidate_post_constructor_rng_neutral) {
    throw "Stage-A evidence does not prove control/candidate RNG neutrality."
}
$Source = Get-Content -Raw -LiteralPath $SourceLauncherArgs | ConvertFrom-Json
$ControlArguments = New-LockedTrainArguments @($Source.train_args) $ControlRunName $false
$CandidateArguments = New-LockedTrainArguments @($Source.train_args) $CandidateRunName $true
$ComparableCandidate = @($CandidateArguments | Where-Object { $_ -ne "--learnable-gabor-texture-residual" })
$ComparableControl = @($ControlArguments)
$ControlRunIndex = [Array]::IndexOf($ComparableControl, "--run-name")
$CandidateRunIndex = [Array]::IndexOf($ComparableCandidate, "--run-name")
$ComparableControl[$ControlRunIndex + 1] = "<matched-run>"
$ComparableCandidate[$CandidateRunIndex + 1] = "<matched-run>"
if (($ComparableControl | ConvertTo-Json -Compress) -ne ($ComparableCandidate | ConvertTo-Json -Compress)) {
    throw "Control and candidate arguments differ beyond the locked Gabor flag."
}

$Protocol = [ordered]@{
    method = $ExpectedMethod
    source_launcher_args = (Resolve-Path $SourceLauncherArgs).Path
    source_launcher_args_sha256 = $SourceSha256
    keeper_checkpoint = (Resolve-Path $KeeperCheckpoint).Path
    keeper_checkpoint_sha256 = $KeeperSha256
    keeper_predictions = (Resolve-Path $KeeperPredictions).Path
    keeper_predictions_sha256 = $KeeperPredictionsSha256
    stage_a_summary = (Resolve-Path $StageASummary).Path
    stage_a_summary_sha256 = $StageASha256
    data_yaml = $DataYaml
    data_yaml_sha256 = $DataSha256
    test_allowed = $false
    candidate_only_argument = "--learnable-gabor-texture-residual"
    seed = 42
    epochs = 2
    max_train_batches = 120
    max_val_batches = 0
    scheduler_total_epochs = 15
    batch_size = 32
    grad_accum_steps = 2
    learning_rate = 0.00008
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
    "--family", "TRKH-Gabor-LHO-FCM-smoke"
)
Invoke-NativePython "evaluate_control_val" ($EvaluationBase + @(
    "--checkpoint", (Join-Path $ControlRunDir "checkpoints\best.pt"),
    "--paper-name", "TRKH-Gabor-control-120b-2e",
    "--output-dir", $ControlVal
))
$Status.control_validation = "completed"
$Status | ConvertTo-Json | Set-Content -LiteralPath $StatusPath -Encoding UTF8
Invoke-NativePython "evaluate_candidate_val" ($EvaluationBase + @(
    "--checkpoint", (Join-Path $CandidateRunDir "checkpoints\best.pt"),
    "--paper-name", "TRKH-Gabor-LHO-FCM-120b-2e",
    "--output-dir", $CandidateVal
))
$Status.candidate_validation = "completed"
$Status | ConvertTo-Json | Set-Content -LiteralPath $StatusPath -Encoding UTF8

Invoke-NativePython "compare_smoke_pair" @(
    "-m", "trkh.tools.audit_learnable_gabor_texture_smoke_pair",
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
Write-Host "Matched Gabor smoke pair completed: $PairOutputDir"
