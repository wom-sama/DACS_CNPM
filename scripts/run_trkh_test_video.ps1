param(
    [string]$Python = "D:\DataAI\.venv\Scripts\python.exe",
    [string]$Checkpoint = "",
    [string]$RunName = "",
    [string]$Engine = "",
    [string]$Source = "",
    [int]$TopK = 5,
    [int]$ClassifyEvery = 1,
    [ValidateSet("none", "ema", "vote")]
    [string]$TemporalSmoothing = "ema",
    [int]$SmoothingWindow = 7,
    [double]$SmoothingAlpha = 0.65,
    [switch]$UsePyTorch,
    [switch]$SaveOutput,
    [string]$OutputPath = "",
    [switch]$PreflightOnly
)

$ErrorActionPreference = "Stop"
$ProjectRoot = Split-Path -Parent $PSScriptRoot
Set-Location $ProjectRoot

function Resolve-RequiredPath {
    param([string]$Value, [string]$Label)
    $candidate = if ([System.IO.Path]::IsPathRooted($Value)) { $Value } else { Join-Path $ProjectRoot $Value }
    $fullPath = [System.IO.Path]::GetFullPath($candidate)
    if (-not (Test-Path -LiteralPath $fullPath)) {
        throw "$Label not found: $fullPath"
    }
    return $fullPath
}

function Get-LatestPromotedCheckpoint {
    param([string]$LatestPath)

    if (-not (Test-Path -LiteralPath $LatestPath)) {
        throw "Checkpoint/RunName not provided and latest_full_pipeline.json is missing. Run the full pipeline first."
    }
    $latest = Get-Content -LiteralPath $LatestPath -Raw | ConvertFrom-Json
    $gate = $latest.validation_gate
    $selection = $gate.train_selection
    $usesCurrentPromotionContract = (
        $null -ne $gate -and
        ($gate.passed -is [bool]) -and
        [bool]$gate.passed -and
        $null -ne $selection -and
        ($selection.name_matches_expected -is [bool]) -and
        [bool]$selection.name_matches_expected -and
        ($selection.direction_matches_expected -is [bool]) -and
        [bool]$selection.direction_matches_expected
    )
    if (-not $usesCurrentPromotionContract) {
        throw (
            "latest_full_pipeline.json is stale or was promoted by a legacy validation gate. " +
            "Pass -Checkpoint/-RunName explicitly or complete a run under the current fair-selection gate."
        )
    }
    if ([string]::IsNullOrWhiteSpace([string]$latest.checkpoint)) {
        throw "latest_full_pipeline.json passed the gate contract but has no checkpoint path."
    }
    return [string]$latest.checkpoint
}

function Invoke-NativeChecked {
    param([string]$FilePath, [string[]]$Arguments, [string]$Name)
    $previousErrorActionPreference = $ErrorActionPreference
    $exitCode = 1
    try {
        $ErrorActionPreference = "Continue"
        & $FilePath @Arguments
        $exitCode = $LASTEXITCODE
    }
    finally {
        $ErrorActionPreference = $previousErrorActionPreference
    }
    if ($exitCode -ne 0) {
        throw "$Name failed with exit code $exitCode."
    }
}

$PythonPath = Resolve-RequiredPath -Value $Python -Label "Python"
if ([string]::IsNullOrWhiteSpace($Checkpoint)) {
    if (-not [string]::IsNullOrWhiteSpace($RunName)) {
        $Checkpoint = "runs\$RunName\checkpoints\best.pt"
    }
    else {
        $latestPath = Join-Path $ProjectRoot "runs\latest_full_pipeline.json"
        $Checkpoint = Get-LatestPromotedCheckpoint -LatestPath $latestPath
    }
}
$CheckpointPath = Resolve-RequiredPath -Value $Checkpoint -Label "Checkpoint"

if (-not $UsePyTorch) {
    if ([string]::IsNullOrWhiteSpace($Engine)) {
        $runDir = Split-Path -Parent (Split-Path -Parent $CheckpointPath)
        $Engine = Join-Path $runDir "deploy\model_fp32_fp16.engine"
    }
    $EnginePath = Resolve-RequiredPath -Value $Engine -Label "TensorRT engine"
}
else {
    $EnginePath = $null
}

if (-not $PreflightOnly -and [string]::IsNullOrWhiteSpace($Source)) {
    throw "Source is required. Pass a video path/URL or use the module directly with --camera-index."
}

$module = if ($UsePyTorch) { "trkh.inference.stream_infer" } else { "trkh.inference.stream_infer_trt" }
Invoke-NativeChecked -FilePath $PythonPath -Name "video_module_preflight" -Arguments @(
    "-c",
    ("import {0}; print('video module import ok')" -f $module)
)

if ($PreflightOnly) {
    [ordered]@{
        status = "ok"
        mode = "preflight_only"
        backend = if ($UsePyTorch) { "pytorch" } else { "tensorrt" }
        checkpoint = $CheckpointPath
        engine = $EnginePath
        source = $Source
        classification_only_supported = $true
        native_stderr_policy = "direct invocation; no 2>&1 pipeline; LASTEXITCODE checked"
    } | ConvertTo-Json -Depth 5
    exit 0
}

$arguments = @(
    "-m", $module,
    "--checkpoint", $CheckpointPath,
    "--source", $Source,
    "--top-k", [string]$TopK,
    "--classify-every", [string]$ClassifyEvery,
    "--temporal-smoothing", $TemporalSmoothing,
    "--smoothing-window", [string]$SmoothingWindow,
    "--smoothing-alpha", [string]$SmoothingAlpha
)
if (-not $UsePyTorch) {
    $arguments += @("--engine", $EnginePath)
}
if ($SaveOutput) {
    $arguments += "--save-output"
}
if (-not [string]::IsNullOrWhiteSpace($OutputPath)) {
    $arguments += @("--output-path", $OutputPath)
}

Invoke-NativeChecked -FilePath $PythonPath -Name "test_video" -Arguments $arguments
