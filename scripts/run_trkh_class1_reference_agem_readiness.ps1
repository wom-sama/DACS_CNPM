param(
    [string]$Python = "D:\DataAI\.venv\Scripts\python.exe",
    [string]$RunName = "audit_class1_reference_agem_readiness_20260715",
    [switch]$PreflightOnly
)

$ErrorActionPreference = "Stop"
$RepoRoot = Split-Path -Parent $PSScriptRoot
Set-Location -LiteralPath $RepoRoot

if (-not (Test-Path -LiteralPath $Python -PathType Leaf)) {
    throw "Python executable not found: $Python"
}

$Arguments = @(
    "-m",
    "trkh.tools.audit_class1_reference_agem_readiness"
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
    throw "Class-1-reference A-GEM readiness audit failed with exit code $LASTEXITCODE."
}
