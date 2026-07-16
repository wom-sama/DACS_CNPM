param(
    [string]$Python = "D:\DataAI\.venv\Scripts\python.exe",
    [string]$Data = "D:\DataAI\AIEx\newdataset\yolo_f\data.yaml",
    [string]$RunName = "audit_supervised_minority_condition_view_readiness_20260716",
    [switch]$PreflightOnly
)

$ErrorActionPreference = "Stop"
$env:CUBLAS_WORKSPACE_CONFIG = ":4096:8"
$RepoRoot = Split-Path -Parent $PSScriptRoot
Set-Location -LiteralPath $RepoRoot

if (-not (Test-Path -LiteralPath $Python -PathType Leaf)) {
    throw "Python executable not found: $Python"
}
if (-not (Test-Path -LiteralPath $Data -PathType Leaf)) {
    throw "Data YAML not found: $Data"
}

$Arguments = @(
    "-m",
    "trkh.tools.audit_supervised_minority_condition_view_readiness",
    "--data",
    $Data
)
if ($PreflightOnly) {
    $Arguments += "--preflight-only"
}
else {
    $Arguments += @(
        "--output-dir",
        (Join-Path $RepoRoot "runs\$RunName")
    )
}

& $Python @Arguments
if ($LASTEXITCODE -ne 0) {
    throw "Supervised-Minority condition-view readiness audit failed with exit code $LASTEXITCODE."
}
