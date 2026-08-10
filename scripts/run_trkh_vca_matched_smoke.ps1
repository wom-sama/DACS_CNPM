param(
    [string]$Python = "D:\DataAI\.venv\Scripts\python.exe",
    [string]$SourceLauncherArgs = "runs\full_v8_yolof_randominit_30e_20260714_105524\launcher_args.json",
    [string]$ControlRunName = "smoke_vca_dense_mhsa_control_120b_2e_20260714",
    [string]$CandidateRunName = "smoke_vca_dense_candidate_120b_2e_20260714",
    [string]$PairOutputDir = "runs\audit_vca_matched_smoke_pair_20260714",
    [switch]$PreflightOnly
)

$ErrorActionPreference = "Stop"
$RepoRoot = (Resolve-Path (Join-Path $PSScriptRoot "..")).Path
Set-Location $RepoRoot

$LockedSourceSha256 = "a493309e7f3f900daf13b5fa6cdd334d36adf4285dd007814c761c350753fad6"
$DataYaml = "D:\DataAI\AIEx\newdataset\yolo_f\data.yaml"
$StageASummary = "runs\audit_vca_stage_a_train_only_20260714\summary.json"

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
        [bool]$EnableVca
    )
    $arguments = [System.Collections.Generic.List[string]]::new()
    foreach ($item in $SourceArguments) {
        $arguments.Add([string]$item)
    }
    Set-ArgumentValue $arguments "--run-name" $RunName
    Set-ArgumentValue $arguments "--output-dir" "runs"
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
    Set-ArgumentValue $arguments "--attention-dropout" "0"
    Set-ArgumentValue $arguments "--attention-view-loss-weight" "0"
    Set-ArgumentValue $arguments "--attention-crop-probability" "0"
    Set-ArgumentValue $arguments "--attention-drop-probability" "0"
    Set-ArgumentValue $arguments "--early-token-mask-keep-rate" "1"
    Remove-Argument $arguments "--token-pruning" $false
    Remove-Argument $arguments "--visual-contrast-attention" $false
    Remove-Argument $arguments "--visual-contrast-attention-layers" $true
    Remove-Argument $arguments "--visual-contrast-tokens" $true
    if ($EnableVca) {
        $arguments.Add("--visual-contrast-attention")
        $arguments.Add("--visual-contrast-attention-layers")
        $arguments.Add("1,2,3,4,5,6,7,8")
        $arguments.Add("--visual-contrast-tokens")
        $arguments.Add("64")
    }
    foreach ($requiredFlag in @("--disable-resume", "--skip-final-test", "--no-pretrained")) {
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
if (-not (Test-Path -LiteralPath $Python -PathType Leaf)) {
    throw "Python executable not found: $Python"
}
if (-not (Test-Path -LiteralPath $DataYaml -PathType Leaf)) {
    throw "Data YAML not found: $DataYaml"
}
if (-not (Test-Path -LiteralPath $StageASummary -PathType Leaf)) {
    throw "Stage-A VCA evidence not found: $StageASummary"
}
$stageA = Get-Content -Raw -LiteralPath $StageASummary | ConvertFrom-Json
if (-not [bool]$stageA.gate.smoke_permission) {
    throw "Stage-A VCA gate did not grant smoke permission."
}
$source = Get-Content -Raw -LiteralPath $SourceLauncherArgs | ConvertFrom-Json
$controlArguments = New-LockedTrainArguments @($source.train_args) $ControlRunName $false
$candidateArguments = New-LockedTrainArguments @($source.train_args) $CandidateRunName $true

$protocol = [ordered]@{
    method = "visual_contrast_attention"
    source_launcher_args = (Resolve-Path $SourceLauncherArgs).Path
    source_launcher_args_sha256 = $sourceSha256
    stage_a_summary = (Resolve-Path $StageASummary).Path
    stage_a_smoke_permission = [bool]$stageA.gate.smoke_permission
    data_yaml = $DataYaml
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
    token_pruning = $false
    attention_dropout = 0.0
    attention_view_supervision = $false
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
$protocol | ConvertTo-Json -Depth 8 | Set-Content -LiteralPath (Join-Path $PairOutputDir "locked_protocol.json") -Encoding UTF8

$status = [ordered]@{
    status = "running"
    control = "pending"
    candidate = "pending"
    control_validation = "pending"
    candidate_validation = "pending"
    comparison = "pending"
}
$status | ConvertTo-Json | Set-Content -LiteralPath (Join-Path $PairOutputDir "status.json") -Encoding UTF8

Invoke-NativePython "train_control" (@("-m", "trkh.training.train") + @($controlArguments))
$status.control = "completed"
$status | ConvertTo-Json | Set-Content -LiteralPath (Join-Path $PairOutputDir "status.json") -Encoding UTF8

Invoke-NativePython "train_candidate" (@("-m", "trkh.training.train") + @($candidateArguments))
$status.candidate = "completed"
$status | ConvertTo-Json | Set-Content -LiteralPath (Join-Path $PairOutputDir "status.json") -Encoding UTF8

$controlCheckpoint = Join-Path (Join-Path "runs" $ControlRunName) "checkpoints\best.pt"
$candidateCheckpoint = Join-Path (Join-Path "runs" $CandidateRunName) "checkpoints\best.pt"
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
    "--family", "TRKH-VCA-smoke"
)
Invoke-NativePython "evaluate_control_val" ($evaluationBase + @(
    "--checkpoint", $controlCheckpoint,
    "--paper-name", "TRKH-dense-MHSA-control-120b-2e",
    "--output-dir", $controlVal
))
$status.control_validation = "completed"
$status | ConvertTo-Json | Set-Content -LiteralPath (Join-Path $PairOutputDir "status.json") -Encoding UTF8

Invoke-NativePython "evaluate_candidate_val" ($evaluationBase + @(
    "--checkpoint", $candidateCheckpoint,
    "--paper-name", "TRKH-dense-VCA-candidate-120b-2e",
    "--output-dir", $candidateVal
))
$status.candidate_validation = "completed"
$status | ConvertTo-Json | Set-Content -LiteralPath (Join-Path $PairOutputDir "status.json") -Encoding UTF8

Invoke-NativePython "compare_smoke_pair" @(
    "-m", "trkh.tools.audit_visual_contrast_smoke_pair",
    "--control-predictions", (Join-Path $controlVal "predictions_detailed.csv"),
    "--candidate-predictions", (Join-Path $candidateVal "predictions_detailed.csv"),
    "--control-run-summary", (Join-Path (Join-Path "runs" $ControlRunName) "summary.json"),
    "--candidate-run-summary", (Join-Path (Join-Path "runs" $CandidateRunName) "summary.json"),
    "--stage-a-summary", $StageASummary,
    "--output-dir", (Join-Path $PairOutputDir "comparison")
)
$status.comparison = "completed"
$status.status = "completed"
$status | ConvertTo-Json | Set-Content -LiteralPath (Join-Path $PairOutputDir "status.json") -Encoding UTF8
Write-Host "Matched VCA smoke pair completed: $PairOutputDir"
