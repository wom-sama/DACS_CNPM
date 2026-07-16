param(
    [string]$Python = "D:\DataAI\.venv\Scripts\python.exe",
    [string]$OutputDir = "runs\audit_cropr_token_selector_a0_pair_corrected_20260717",
    [string]$ExpectedSummarySha256 = "",
    [string]$VisualReviewNote = "",
    [switch]$RunAudit,
    [switch]$FinalizePass,
    [switch]$FinalizeFail
)

$ErrorActionPreference = "Stop"
$RepoRoot = Split-Path -Parent $PSScriptRoot
Set-Location $RepoRoot

$SelectedModes = @($RunAudit, $FinalizePass, $FinalizeFail) | Where-Object { $_ }
if ($SelectedModes.Count -ne 1) {
    throw "Chon dung mot mode: -RunAudit, -FinalizePass, hoac -FinalizeFail."
}
if (-not (Test-Path -LiteralPath $Python)) {
    throw "Khong tim thay Python: $Python"
}

function Assert-FileSha256 {
    param([string]$Path, [string]$Expected)
    if (-not (Test-Path -LiteralPath $Path -PathType Leaf)) {
        throw "Khong tim thay artifact da khoa: $Path"
    }
    $Observed = (Get-FileHash -Algorithm SHA256 -LiteralPath $Path).Hash.ToLowerInvariant()
    if ($Observed -ne $Expected.ToLowerInvariant()) {
        throw "SHA-256 mismatch: $Path observed=$Observed expected=$Expected"
    }
}

$Protocol = "docs\TRKH_5CLASS_CROPR_TOKEN_SELECTOR_READINESS_PROTOCOL_20260716.md"
$Preflight = "runs\audit_cropr_token_selector_preflight_20260716\summary.json"
$PairManifest = "runs\cropr_a0_pair_manifest_20260716.json"
$ControlLast = "runs\probe_cropr_a0_native_control_5e_20260716\checkpoints\last.pt"
$CandidateLast = "runs\probe_cropr_a0_learned_candidate_5e_20260716\checkpoints\last.pt"
$FailedSummary = "runs\audit_cropr_token_selector_a0_pair_20260717\summary.json"
$FailureRecord = "runs\audit_cropr_token_selector_a0_pair_20260717\failure_record.json"
$FailedManifest = "runs\audit_cropr_token_selector_a0_pair_20260717\artifact_manifest.json"

Assert-FileSha256 $Protocol "42e912a6bdc320d98a9acbb79b6d48c1e372a5df861f279af1c6988b3ff6a847"
Assert-FileSha256 $Preflight "12902d6930ec89fd658c2b86e42a30f07ea44643a2fd67a8a1fddb5b7f1e873c"
Assert-FileSha256 $PairManifest "09a89cbe4cf4f87e6ce2ac52dd6affd80b732357bc53c1bb75979f7a6180cc1e"
Assert-FileSha256 $ControlLast "97c0ed6d922926c670c55799c972bb39d1c19e13f295623e44bac4cb66a7225d"
Assert-FileSha256 $CandidateLast "6fa48de9b4d1a07ded560c9730e80987a535769c6b69d2d7af6940ca9fa7c98a"
Assert-FileSha256 $FailedSummary "6404590b0a25049d41238b019bba6da0ef84aea35184f68c588d922f4c04f9cb"
Assert-FileSha256 $FailureRecord "5dc48089553d747656d6e014df71bcbc8ef0fd4c0adea53b5b61301f1a6c7311"
Assert-FileSha256 $FailedManifest "af4ff08eebccd75d13ec5d7764fe4e0db0966ff44fffa5f0ae68b53f68fcbdc0"

if ($RunAudit) {
    if (Test-Path -LiteralPath $OutputDir) {
        throw "Audit output da ton tai; protocol cam overwrite: $OutputDir"
    }
    $TrackedStatus = git status --short --untracked-files=no
    if ($LASTEXITCODE -ne 0 -or -not [string]::IsNullOrWhiteSpace(($TrackedStatus -join "`n"))) {
        throw "Tracked worktree phai clean truoc khi mo train-only holdout."
    }
    $Head = (git rev-parse HEAD).Trim()
    $Upstream = (git rev-parse '@{upstream}').Trim()
    if ($LASTEXITCODE -ne 0 -or $Head -ne $Upstream) {
        throw "HEAD phai duoc push truoc Cropr pair audit."
    }
    & $Python -m trkh.tools.audit_cropr_token_selector_pair `
        --control-checkpoint $ControlLast `
        --candidate-checkpoint $CandidateLast `
        --pair-manifest $PairManifest `
        --preflight-summary $Preflight `
        --protocol $Protocol `
        --output-dir $OutputDir `
        --batch-size 32 `
        --xai-batch-size 2 `
        --num-workers 4 `
        --seed 42
    if ($LASTEXITCODE -ne 0) {
        throw "Cropr pair audit failed with exit code $LASTEXITCODE."
    }
    Write-Host "Cropr pair quantitative audit completed; visual review remains required."
}

if ($FinalizePass -or $FinalizeFail) {
    if ($ExpectedSummarySha256 -notmatch '^[0-9a-fA-F]{64}$') {
        throw "Finalize yeu cau -ExpectedSummarySha256 gom 64 ky tu hex."
    }
    if ([string]::IsNullOrWhiteSpace($VisualReviewNote)) {
        throw "Finalize yeu cau -VisualReviewNote khong rong."
    }
    $Result = if ($FinalizePass) { "pass" } else { "fail" }
    & $Python -m trkh.tools.audit_cropr_token_selector_pair `
        --output-dir $OutputDir `
        --finalize-visual-review `
        --expected-summary-sha256 $ExpectedSummarySha256 `
        --visual-review-result $Result `
        --visual-review-note $VisualReviewNote
    if ($LASTEXITCODE -ne 0) {
        throw "Cropr visual finalization failed with exit code $LASTEXITCODE."
    }
}
