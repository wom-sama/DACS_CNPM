param(
    [string]$Python = "D:\DataAI\.venv\Scripts\python.exe",
    [ValidateSet("Preflight", "Formal", "Finalize")]
    [string]$Phase = "Preflight",
    [string]$OutputDir = "runs\audit_global_response_normalization_signal_a0_20260717",
    [ValidateSet("pass", "fail")]
    [string]$VisualReviewResult = "pass",
    [string]$VisualReviewNote = ""
)

$ErrorActionPreference = "Stop"
$RepoRoot = Split-Path -Parent $PSScriptRoot
Set-Location $RepoRoot

if (-not (Test-Path -LiteralPath $Python -PathType Leaf)) {
    throw "Khong tim thay Python: $Python"
}

function Invoke-LockedPython {
    param([string[]]$Arguments)
    & $Python @Arguments
    if ($LASTEXITCODE -ne 0) {
        throw "Python failed with exit code ${LASTEXITCODE}: $($Arguments -join ' ')"
    }
}

if ($Phase -in @("Preflight", "Formal")) {
    Invoke-LockedPython -Arguments @(
        "-m", "py_compile",
        "trkh\tools\audit_global_response_normalization_signal.py"
    )
    Invoke-LockedPython -Arguments @(
        "-m", "pytest",
        "tests\test_audit_global_response_normalization_signal.py",
        "-q"
    )
}

if ($Phase -eq "Preflight") {
    if (Test-Path -LiteralPath $OutputDir) {
        throw "Preflight cam tao hoac dung output directory: $OutputDir"
    }
    Invoke-LockedPython -Arguments @(
        "-m", "trkh.tools.audit_global_response_normalization_signal",
        "--preflight-only"
    )
    Write-Host "GRN A0 preflight passed without constructing a loader or output directory."
    exit 0
}

if ($Phase -eq "Formal") {
    if (Test-Path -LiteralPath $OutputDir) {
        throw "Formal output da ton tai; protocol cam overwrite/rerun: $OutputDir"
    }
    Invoke-LockedPython -Arguments @(
        "-m", "trkh.tools.audit_global_response_normalization_signal",
        "--output-dir", $OutputDir
    )
    $SummaryPath = Join-Path $OutputDir "summary.json"
    if (-not (Test-Path -LiteralPath $SummaryPath -PathType Leaf)) {
        throw "Formal GRN A0 khong tao summary: $SummaryPath"
    }
    $SummarySha = (Get-FileHash -Algorithm SHA256 -LiteralPath $SummaryPath).Hash.ToLowerInvariant()
    $Summary = Get-Content -Raw -LiteralPath $SummaryPath | ConvertFrom-Json
    if ($Summary.status -eq "awaiting_visual_review") {
        Write-Host "GRN automated audit completed. Review all grn_signal_contact_sheet_*.png."
        Write-Host "Locked pre-review summary SHA-256: $SummarySha"
    }
    else {
        Write-Host "GRN formal audit completed with terminal status: $($Summary.status)"
        Write-Host "Summary SHA-256: $SummarySha"
    }
    exit 0
}

if ([string]::IsNullOrWhiteSpace($VisualReviewNote)) {
    throw "Finalize yeu cau -VisualReviewNote mo ta ket qua xem day du bon trang."
}
$SummaryPath = Join-Path $OutputDir "summary.json"
if (-not (Test-Path -LiteralPath $SummaryPath -PathType Leaf)) {
    throw "Khong tim thay formal summary: $SummaryPath"
}
$SummarySha = (Get-FileHash -Algorithm SHA256 -LiteralPath $SummaryPath).Hash.ToLowerInvariant()
$Summary = Get-Content -Raw -LiteralPath $SummaryPath | ConvertFrom-Json
if ($Summary.status -ne "awaiting_visual_review") {
    throw "Formal status khong yeu cau visual review: $($Summary.status)"
}
Invoke-LockedPython -Arguments @(
    "-m", "trkh.tools.audit_global_response_normalization_signal",
    "--output-dir", $OutputDir,
    "--finalize-visual-review",
    "--visual-review-result", $VisualReviewResult,
    "--visual-review-note", $VisualReviewNote,
    "--expected-summary-sha256", $SummarySha
)
Write-Host "GRN A0 visual review finalized from summary SHA-256 $SummarySha."
