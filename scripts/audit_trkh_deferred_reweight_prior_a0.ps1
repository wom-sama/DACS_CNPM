[CmdletBinding()]
param(
    [string]$Python = "D:\DataAI\.venv\Scripts\python.exe",
    [string]$OutputDir = "runs\audit_deferred_reweight_prior_a0_20260720",
    [switch]$PreflightOnly
)

$ErrorActionPreference = "Stop"
$RepoRoot = Split-Path -Parent $PSScriptRoot
$ResolvedOutput = [System.IO.Path]::GetFullPath((Join-Path $RepoRoot $OutputDir))

Push-Location $RepoRoot
try {
    & $Python -m trkh.tools.audit_deferred_reweight_prior_a0 --preflight-only
    if ($LASTEXITCODE -ne 0) {
        throw "Deferred-reweight A0 preflight failed with exit code $LASTEXITCODE"
    }
    if ($PreflightOnly) {
        return
    }

    & $Python -m trkh.tools.audit_deferred_reweight_prior_a0 --output-dir $ResolvedOutput
    if ($LASTEXITCODE -ne 0) {
        throw "Deferred-reweight A0 formal run failed with exit code $LASTEXITCODE"
    }

    $SummaryPath = Join-Path $ResolvedOutput "summary.json"
    $ReplayPath = Join-Path $ResolvedOutput "independent_replay.json"
    $SummaryHash = (Get-FileHash -Algorithm SHA256 -LiteralPath $SummaryPath).Hash.ToLowerInvariant()
    & $Python -m trkh.tools.audit_deferred_reweight_prior_a0 `
        --replay-summary $SummaryPath `
        --expected-summary-sha256 $SummaryHash `
        --replay-output $ReplayPath
    if ($LASTEXITCODE -ne 0) {
        throw "Deferred-reweight A0 independent replay failed with exit code $LASTEXITCODE"
    }
}
finally {
    Pop-Location
}
