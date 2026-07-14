param(
    [string]$Python = "D:\DataAI\.venv\Scripts\python.exe",
    [string]$SourceLauncherArgs = "runs\full_v8_yolof_randominit_30e_20260714_105524\launcher_args.json",
    [string]$ControlRunName = "smoke_inceptionnext_atto_control_120b_2e_20260715",
    [string]$CandidateRunName = "smoke_inceptionnext_atto_candidate_120b_2e_20260715",
    [string]$PairOutputDir = "runs\audit_inceptionnext_atto_matched_smoke_pair_20260715",
    [string]$StageASummary = "runs\audit_inceptionnext_atto_stage_a_20260715\summary.json",
    [string]$LockedStageASha256 = "8b1b4e6d1485bfa4e5224c432a769f7ccdb91ef5a15a345a4aa9a9f6b6ff41a5",
    [string]$ExpectedMethod = "inceptionnext_atto_surface_tokenizer",
    [string]$CandidateStemArchitecture = "inceptionnext_atto_tokenizer",
    [string]$EvaluationFamily = "TRKH-InceptionNeXt-Atto-smoke",
    [string]$CandidatePaperName = "TRKH-InceptionNeXt-Atto-120b-2e",
    [switch]$PreflightOnly
)

$ErrorActionPreference = "Stop"
$RepoRoot = (Resolve-Path (Join-Path $PSScriptRoot "..")).Path
Set-Location $RepoRoot

$LockedSourceSha256 = "a493309e7f3f900daf13b5fa6cdd334d36adf4285dd007814c761c350753fad6"
$LockedDataSha256 = "716e33df24c63a9e9920f97b685199707fb84ab4c7154544f5dd9a3e00d884ef"
$DataYaml = "D:\DataAI\AIEx\newdataset\yolo_f\data.yaml"

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

function Set-ArgumentValue {
    param(
        [System.Collections.Generic.List[string]]$Arguments,
        [string]$Name,
        [string]$Value
    )
    $index = $Arguments.IndexOf($Name)
    if ($index -lt 0) {
        $Arguments.Add($Name)
        $Arguments.Add($Value)
        return
    }
    if ($index + 1 -ge $Arguments.Count -or $Arguments[$index + 1].StartsWith("--")) {
        throw "Argument $Name does not have a value in the source protocol."
    }
    $Arguments[$index + 1] = $Value
}

function Remove-Argument {
    param(
        [System.Collections.Generic.List[string]]$Arguments,
        [string]$Name,
        [bool]$HasValue
    )
    while (($index = $Arguments.IndexOf($Name)) -ge 0) {
        $Arguments.RemoveAt($index)
        if ($HasValue) {
            if ($index -ge $Arguments.Count -or $Arguments[$index].StartsWith("--")) {
                throw "Argument $Name is missing its value."
            }
            $Arguments.RemoveAt($index)
        }
    }
}

function New-LockedTrainArguments {
    param(
        [object[]]$SourceArguments,
        [string]$RunName,
        [string]$StemArchitecture
    )
    $arguments = [System.Collections.Generic.List[string]]::new()
    foreach ($item in $SourceArguments) {
        $arguments.Add([string]$item)
    }
    Set-ArgumentValue $arguments "--run-name" $RunName
    Set-ArgumentValue $arguments "--output-dir" "runs"
    Set-ArgumentValue $arguments "--stem-architecture" $StemArchitecture
    Set-ArgumentValue $arguments "--seed" "42"
    Set-ArgumentValue $arguments "--batch-size" "32"
    Set-ArgumentValue $arguments "--grad-accum-steps" "2"
    Set-ArgumentValue $arguments "--epochs" "2"
    Set-ArgumentValue $arguments "--scheduler-total-epochs" "15"
    Set-ArgumentValue $arguments "--patience" "3"
    Set-ArgumentValue $arguments "--max-train-batches" "120"
    Set-ArgumentValue $arguments "--max-val-batches" "0"
    Set-ArgumentValue $arguments "--learning-rate" "0.0005"
    Set-ArgumentValue $arguments "--min-learning-rate" "0.000001"
    Set-ArgumentValue $arguments "--warmup-epochs" "1"
    Set-ArgumentValue $arguments "--weight-decay" "0.05"
    Set-ArgumentValue $arguments "--attention-view-loss-weight" "0"
    Set-ArgumentValue $arguments "--attention-crop-probability" "0"
    Set-ArgumentValue $arguments "--attention-drop-probability" "0"
    Remove-Argument $arguments "--visual-contrast-attention" $false
    Remove-Argument $arguments "--visual-contrast-attention-layers" $true
    Remove-Argument $arguments "--visual-contrast-tokens" $true
    foreach ($requiredFlag in @("--disable-resume", "--skip-final-test", "--no-pretrained", "--token-pruning")) {
        if (-not $arguments.Contains($requiredFlag)) {
            throw "Locked source arguments are missing $requiredFlag."
        }
    }
    return $arguments
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

$sourceSha256 = Assert-FileSha256 $SourceLauncherArgs $LockedSourceSha256
$stageASha256 = Assert-FileSha256 $StageASummary $LockedStageASha256
$dataSha256 = Assert-FileSha256 $DataYaml $LockedDataSha256
if (-not (Test-Path -LiteralPath $Python -PathType Leaf)) {
    throw "Python executable not found: $Python"
}
$stageA = Get-Content -Raw -LiteralPath $StageASummary | ConvertFrom-Json
if ($stageA.method -ne $ExpectedMethod) {
    throw "Stage-A method does not match the locked tokenizer method."
}
if (-not [bool]$stageA.gate.smoke_permission) {
    throw "Stage-A tokenizer gate did not grant smoke permission."
}
if ([bool]$stageA.sources.validation_loaded -or [bool]$stageA.sources.test_loaded) {
    throw "Stage-A evidence is not train-only."
}
$source = Get-Content -Raw -LiteralPath $SourceLauncherArgs | ConvertFrom-Json
$controlArguments = New-LockedTrainArguments @($source.train_args) $ControlRunName "conv_pool"
$candidateArguments = New-LockedTrainArguments @($source.train_args) $CandidateRunName $CandidateStemArchitecture

$protocol = [ordered]@{
    method = $ExpectedMethod
    source_launcher_args = (Resolve-Path $SourceLauncherArgs).Path
    source_launcher_args_sha256 = $sourceSha256
    stage_a_summary = (Resolve-Path $StageASummary).Path
    stage_a_summary_sha256 = $stageASha256
    stage_a_smoke_permission = [bool]$stageA.gate.smoke_permission
    data_yaml = $DataYaml
    data_yaml_sha256 = $dataSha256
    test_allowed = $false
    seed = 42
    epochs = 2
    max_train_batches = 120
    max_val_batches = 0
    scheduler_total_epochs = 15
    batch_size = 32
    grad_accum_steps = 2
    learning_rate = 0.0005
    warmup_epochs = 1
    token_pruning = $true
    attention_view_supervision = $false
    control_stem_architecture = "conv_pool"
    candidate_stem_architecture = $CandidateStemArchitecture
    control_run_name = $ControlRunName
    candidate_run_name = $CandidateRunName
    control_arguments = @($controlArguments)
    candidate_arguments = @($candidateArguments)
}

if ($PreflightOnly) {
    [ordered]@{
        status = "ok"
        mode = "preflight_only"
        protocol = $protocol
    } | ConvertTo-Json -Depth 8
    exit 0
}

foreach ($path in @(
    (Join-Path "runs" $ControlRunName),
    (Join-Path "runs" $CandidateRunName),
    $PairOutputDir
)) {
    if (Test-Path -LiteralPath $path) {
        throw "Refusing to overwrite existing smoke artifact: $path"
    }
}
New-Item -ItemType Directory -Path $PairOutputDir -Force | Out-Null
$protocolPath = Join-Path $PairOutputDir "locked_protocol.json"
$protocol | ConvertTo-Json -Depth 8 | Set-Content -LiteralPath $protocolPath -Encoding UTF8

$status = [ordered]@{
    status = "running"
    control = "pending"
    candidate = "pending"
    control_validation = "pending"
    candidate_validation = "pending"
    comparison = "pending"
}
$statusPath = Join-Path $PairOutputDir "status.json"
$status | ConvertTo-Json | Set-Content -LiteralPath $statusPath -Encoding UTF8

Invoke-NativePython "train_control" (@("-m", "trkh.training.train") + @($controlArguments))
$status.control = "completed"
$status | ConvertTo-Json | Set-Content -LiteralPath $statusPath -Encoding UTF8

Invoke-NativePython "train_candidate" (@("-m", "trkh.training.train") + @($candidateArguments))
$status.candidate = "completed"
$status | ConvertTo-Json | Set-Content -LiteralPath $statusPath -Encoding UTF8

$controlRunDir = Join-Path "runs" $ControlRunName
$candidateRunDir = Join-Path "runs" $CandidateRunName
$controlCheckpoint = Join-Path $controlRunDir "checkpoints\best.pt"
$candidateCheckpoint = Join-Path $candidateRunDir "checkpoints\best.pt"
$controlVal = Join-Path $PairOutputDir "control_val"
$candidateVal = Join-Path $PairOutputDir "candidate_val"
$evaluationBase = @(
    "-m", "trkh.evaluation.evaluate",
    "--data", $DataYaml,
    "--class-name-mode", "raw",
    "--expected-num-classes", "5",
    "--split", "val",
    "--batch-size", "64",
    "--num-workers", "2",
    "--amp",
    "--bbox-token-prior-source", "bbox",
    "--family", $EvaluationFamily
)
Invoke-NativePython "evaluate_control_val" ($evaluationBase + @(
    "--checkpoint", $controlCheckpoint,
    "--paper-name", "TRKH-control-120b-2e",
    "--output-dir", $controlVal
))
$status.control_validation = "completed"
$status | ConvertTo-Json | Set-Content -LiteralPath $statusPath -Encoding UTF8

Invoke-NativePython "evaluate_candidate_val" ($evaluationBase + @(
    "--checkpoint", $candidateCheckpoint,
    "--paper-name", $CandidatePaperName,
    "--output-dir", $candidateVal
))
$status.candidate_validation = "completed"
$status | ConvertTo-Json | Set-Content -LiteralPath $statusPath -Encoding UTF8

Invoke-NativePython "compare_smoke_pair" @(
    "-m", "trkh.tools.audit_tokenizer_smoke_pair",
    "--control-predictions", (Join-Path $controlVal "predictions_detailed.csv"),
    "--candidate-predictions", (Join-Path $candidateVal "predictions_detailed.csv"),
    "--control-run-summary", (Join-Path $controlRunDir "summary.json"),
    "--candidate-run-summary", (Join-Path $candidateRunDir "summary.json"),
    "--control-resolved-config", (Join-Path $controlRunDir "resolved_config.json"),
    "--candidate-resolved-config", (Join-Path $candidateRunDir "resolved_config.json"),
    "--stage-a-summary", $StageASummary,
    "--locked-protocol", $protocolPath,
    "--expected-method", $ExpectedMethod,
    "--candidate-stem", $CandidateStemArchitecture,
    "--output-dir", (Join-Path $PairOutputDir "comparison")
)
$status.comparison = "completed"
$status.status = "completed"
$status | ConvertTo-Json | Set-Content -LiteralPath $statusPath -Encoding UTF8
Write-Host "Matched tokenizer smoke pair completed: $PairOutputDir"
