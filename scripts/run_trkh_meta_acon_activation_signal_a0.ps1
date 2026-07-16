param(
    [string]$Python = "D:\DataAI\.venv\Scripts\python.exe",
    [string]$RunName = "audit_meta_acon_activation_signal_a0_20260717",
    [switch]$PreflightOnly
)

$ErrorActionPreference = "Stop"
$RepoRoot = Split-Path -Parent $PSScriptRoot
Set-Location -LiteralPath $RepoRoot

if (-not (Test-Path -LiteralPath $Python -PathType Leaf)) {
    throw "Python executable not found: $Python"
}

$BaseArguments = @(
    "-m",
    "trkh.tools.audit_meta_acon_activation_signal"
)

& $Python @BaseArguments "--preflight-only"
if ($LASTEXITCODE -ne 0) {
    throw "Meta-ACON Activation Signal A0 preflight failed with exit code $LASTEXITCODE."
}

if ($PreflightOnly) {
    return
}

$OutputDir = Join-Path $RepoRoot "runs\$RunName"
& $Python @BaseArguments "--output-dir" $OutputDir
if ($LASTEXITCODE -ne 0) {
    throw "Meta-ACON Activation Signal A0 formal audit failed with exit code $LASTEXITCODE."
}
