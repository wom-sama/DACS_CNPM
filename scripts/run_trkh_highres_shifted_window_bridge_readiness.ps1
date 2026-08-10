param(
    [string]$Python = "D:\DataAI\.venv\Scripts\python.exe",
    [string]$RunName = "audit_highres_shifted_window_bridge_a0_$(Get-Date -Format yyyyMMdd_HHmmss)",
    [switch]$PreflightOnly
)

$ErrorActionPreference = "Stop"
$RepoRoot = Split-Path -Parent $PSScriptRoot
Set-Location -LiteralPath $RepoRoot

if (-not (Test-Path -LiteralPath $Python -PathType Leaf)) {
    throw "Python executable not found: $Python"
}

$env:CUBLAS_WORKSPACE_CONFIG = ":4096:8"

$Arguments = @(
    "-m",
    "trkh.tools.audit_highres_shifted_window_bridge_readiness",
    "--output-dir",
    (Join-Path $RepoRoot "runs\$RunName")
)
if ($PreflightOnly) {
    $Arguments += "--preflight-only"
}

& $Python @Arguments
if ($LASTEXITCODE -ne 0) {
    throw "High-resolution shifted-window bridge A0 failed with exit code $LASTEXITCODE."
}
