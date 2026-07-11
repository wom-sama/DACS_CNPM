param(
    [string]$Python = "D:\DataAI\.venv\Scripts\python.exe",
    [string]$DataYaml = "D:\DataAI\AIEx\newdataset\yolo_f\data.yaml",
    [string]$Checkpoint = "",
    [string]$RunName = "",
    [string]$OutputDir = "",
    [string]$TrtExecPath = "",
    [int]$BenchmarkWarmup = 20,
    [int]$BenchmarkRuns = 100,
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

function Invoke-NativeChecked {
    param([string]$FilePath, [string[]]$Arguments, [string]$Name)
    Write-Host "`n=== $Name ===" -ForegroundColor Cyan
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
$DataYamlPath = Resolve-RequiredPath -Value $DataYaml -Label "Data YAML"

if ([string]::IsNullOrWhiteSpace($Checkpoint)) {
    if (-not [string]::IsNullOrWhiteSpace($RunName)) {
        $Checkpoint = "runs\$RunName\checkpoints\best.pt"
    }
    else {
        $latestPath = Join-Path $ProjectRoot "runs\latest_full_pipeline.json"
        if (-not (Test-Path -LiteralPath $latestPath)) {
            throw "Checkpoint/RunName not provided and latest_full_pipeline.json is missing. Run the full pipeline first."
        }
        $latest = Get-Content -LiteralPath $latestPath -Raw | ConvertFrom-Json
        $Checkpoint = [string]$latest.checkpoint
    }
}
$CheckpointPath = Resolve-RequiredPath -Value $Checkpoint -Label "Checkpoint"

if ([string]::IsNullOrWhiteSpace($OutputDir)) {
    $runDir = Split-Path -Parent (Split-Path -Parent $CheckpointPath)
    $OutputDir = Join-Path $runDir "deploy"
}
elseif (-not [System.IO.Path]::IsPathRooted($OutputDir)) {
    $OutputDir = Join-Path $ProjectRoot $OutputDir
}
$OutputDir = [System.IO.Path]::GetFullPath($OutputDir)

$resolvedTrtExec = ""
if (-not [string]::IsNullOrWhiteSpace($TrtExecPath)) {
    $resolvedTrtExec = Resolve-RequiredPath -Value $TrtExecPath -Label "trtexec"
}
else {
    $fallback = "C:\TensorRT\TensorRT-10.7.0.23\bin\trtexec.exe"
    if (Test-Path -LiteralPath $fallback) {
        $resolvedTrtExec = $fallback
    }
}
if ([string]::IsNullOrWhiteSpace($resolvedTrtExec)) {
    throw "trtexec.exe was not found. Pass -TrtExecPath explicitly."
}

Invoke-NativeChecked -FilePath $PythonPath -Name "engine_module_preflight" -Arguments @(
    "-c",
    "import tensorrt, torch, onnxruntime; print({'tensorrt': tensorrt.__version__, 'cuda': torch.cuda.is_available()})"
)

if ($PreflightOnly) {
    [ordered]@{
        status = "ok"
        mode = "preflight_only"
        checkpoint = $CheckpointPath
        data_yaml = $DataYamlPath
        output_dir = $OutputDir
        trtexec = $resolvedTrtExec
        expected_engine = Join-Path $OutputDir "model_fp32_fp16.engine"
        native_stderr_policy = "direct invocation; no 2>&1 pipeline; LASTEXITCODE checked"
    } | ConvertTo-Json -Depth 5
    exit 0
}

New-Item -ItemType Directory -Path $OutputDir -Force | Out-Null
$arguments = @(
    "-m", "trkh.inference.deploy",
    "--checkpoint", $CheckpointPath,
    "--data", $DataYamlPath,
    "--class-name-mode", "raw",
    "--expected-num-classes", "5",
    "--output-dir", $OutputDir,
    "--batch-size", "64",
    "--num-workers", "0",
    "--benchmark-batch-size", "1",
    "--benchmark-warmup", [string]$BenchmarkWarmup,
    "--benchmark-runs", [string]$BenchmarkRuns,
    "--split", "val",
    "--skip-accuracy",
    "--trtexec-path", $resolvedTrtExec
)
Invoke-NativeChecked -FilePath $PythonPath -Name "export_onnx_and_tensorrt_fp16" -Arguments $arguments

$EnginePath = Join-Path $OutputDir "model_fp32_fp16.engine"
if (-not (Test-Path -LiteralPath $EnginePath)) {
    throw "Deploy command completed without TensorRT engine: $EnginePath"
}

[ordered]@{
    status = "completed"
    checkpoint = $CheckpointPath
    output_dir = $OutputDir
    engine = $EnginePath
    deploy_summary = Join-Path $OutputDir "deploy_summary.json"
    completed_at = (Get-Date).ToString("o")
} | ConvertTo-Json -Depth 5
