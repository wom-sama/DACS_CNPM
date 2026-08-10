param(
    [string]$Python = "D:\DataAI\.venv\Scripts\python.exe",
    [ValidateSet("Preflight", "Formal", "Finalize")]
    [string]$Phase = "Preflight",
    [string]$OutputDir = "runs\audit_wildcat_negative_evidence_a0_20260717",
    [ValidateSet("pass", "fail")]
    [string]$VisualReviewResult = "fail",
    [string]$VisualReviewNote = ""
)

$ErrorActionPreference = "Stop"
$env:CUBLAS_WORKSPACE_CONFIG = ":4096:8"
$RepoRoot = Split-Path -Parent $PSScriptRoot
Set-Location -LiteralPath $RepoRoot

if (-not (Test-Path -LiteralPath $Python -PathType Leaf)) {
    throw "Python executable not found: $Python"
}

function Invoke-LockedPython {
    param([string[]]$Arguments)
    & $Python @Arguments
    if ($LASTEXITCODE -ne 0) {
        throw "WILDCAT command failed with exit code ${LASTEXITCODE}: $($Arguments -join ' ')"
    }
}

if ($Phase -in @("Preflight", "Formal")) {
    Invoke-LockedPython -Arguments @(
        "-m", "py_compile",
        "trkh\tools\audit_wildcat_negative_evidence_readiness.py",
        "tests\test_audit_wildcat_negative_evidence_readiness.py"
    )
    Invoke-LockedPython -Arguments @(
        "-m", "pyflakes",
        "trkh\tools\audit_wildcat_negative_evidence_readiness.py",
        "tests\test_audit_wildcat_negative_evidence_readiness.py"
    )
    Invoke-LockedPython -Arguments @(
        "-m", "pytest",
        "tests\test_audit_wildcat_negative_evidence_readiness.py",
        "-q"
    )
}

if ($Phase -eq "Preflight") {
    if (Test-Path -LiteralPath $OutputDir) {
        throw "Preflight must not create or reuse an output directory: $OutputDir"
    }
    Invoke-LockedPython -Arguments @(
        "-m", "trkh.tools.audit_wildcat_negative_evidence_readiness",
        "--preflight-only"
    )
    Write-Host "WILDCAT A0 preflight passed without creating an output directory."
    exit 0
}

if ($Phase -eq "Formal") {
    if (Test-Path -LiteralPath $OutputDir) {
        throw "Formal WILDCAT output already exists; overwrite/rerun is forbidden: $OutputDir"
    }
    Invoke-LockedPython -Arguments @(
        "-m", "trkh.tools.audit_wildcat_negative_evidence_readiness",
        "--output-dir", $OutputDir
    )
    $SummaryPath = Join-Path $OutputDir "summary.json"
    if (-not (Test-Path -LiteralPath $SummaryPath -PathType Leaf)) {
        throw "Formal WILDCAT audit did not create summary.json."
    }
    $Summary = Get-Content -Raw -LiteralPath $SummaryPath | ConvertFrom-Json
    $SummarySha = (Get-FileHash -Algorithm SHA256 -LiteralPath $SummaryPath).Hash.ToLowerInvariant()
    Write-Host "WILDCAT formal status: $($Summary.status)"
    Write-Host "Locked pre-review summary SHA-256: $SummarySha"
    exit 0
}

if ([string]::IsNullOrWhiteSpace($VisualReviewNote)) {
    throw "Finalize requires -VisualReviewNote covering every contact sheet."
}
$SummaryPath = Join-Path $OutputDir "summary.json"
if (-not (Test-Path -LiteralPath $SummaryPath -PathType Leaf)) {
    throw "Formal WILDCAT summary is missing: $SummaryPath"
}
$Summary = Get-Content -Raw -LiteralPath $SummaryPath | ConvertFrom-Json
if ($Summary.status -ne "awaiting_visual_review") {
    throw "WILDCAT summary is not awaiting visual review: $($Summary.status)"
}
$SummarySha = (Get-FileHash -Algorithm SHA256 -LiteralPath $SummaryPath).Hash.ToLowerInvariant()
Invoke-LockedPython -Arguments @(
    "-m", "trkh.tools.audit_wildcat_negative_evidence_readiness",
    "--output-dir", $OutputDir,
    "--finalize-visual-review",
    "--visual-review-result", $VisualReviewResult,
    "--visual-review-note", $VisualReviewNote,
    "--expected-summary-sha256", $SummarySha
)
Write-Host "WILDCAT visual review finalized from summary SHA-256 $SummarySha."
