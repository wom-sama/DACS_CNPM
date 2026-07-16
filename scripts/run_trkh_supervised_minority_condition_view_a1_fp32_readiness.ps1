param(
    [string]$Python = "D:\DataAI\.venv\Scripts\python.exe",
    [string]$Data = "D:\DataAI\AIEx\newdataset\yolo_f\data.yaml",
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
    "--contract",
    "a1_fp32_cidt",
    "--protocol",
    (Join-Path $RepoRoot "docs\TRKH_5CLASS_SUPERVISED_MINORITY_CONDITION_VIEW_A1_NUMERIC_PROTOCOL_20260716.md"),
    "--data",
    $Data,
    "--batch-size",
    "64",
    "--num-workers",
    "4",
    "--xai-batch-size",
    "4",
    "--seed",
    "42",
    "--fold",
    "0",
    "--output-dir",
    (Join-Path $RepoRoot "runs\audit_supervised_minority_condition_view_a1_fp32_20260716")
)
if ($PreflightOnly) {
    $Arguments += "--preflight-only"
}

& $Python @Arguments
if ($LASTEXITCODE -ne 0) {
    throw "Supervised-Minority condition-view A1 FP32 readiness audit failed with exit code $LASTEXITCODE."
}
