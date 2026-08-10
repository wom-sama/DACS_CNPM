[CmdletBinding()]
param(
    [ValidateSet("Preflight", "Finalize", "Replay", "All")]
    [string]$Phase = "Preflight",
    [string]$Python = "D:\DataAI\.venv\Scripts\python.exe",
    [string]$GenerationRoot = "D:\DataAI\AIEx\TRKH\runs\audit_saspa_dual_view_synthetic_a0_20260723\f1_tiny_output_v2"
)

Set-StrictMode -Version Latest
$ErrorActionPreference = "Stop"
if (Test-Path variable:PSNativeCommandUseErrorActionPreference) {
    $PSNativeCommandUseErrorActionPreference = $false
}

$RepoRoot = (Resolve-Path (Join-Path $PSScriptRoot "..")).Path
$Module = "trkh.tools.audit_saspa_synthetic_a0_f1_finalize"
$Lock = "docs\TRKH_5CLASS_SASPA_F1_FINALIZATION_LOCK_20260724.json"
$Summary = Join-Path $GenerationRoot "f1_final_summary.json"
$FocusedTests = @(
    "tests\test_audit_saspa_synthetic_a0_f1.py",
    "tests\test_audit_saspa_synthetic_a0_f1_duplicates.py",
    "tests\test_audit_saspa_synthetic_a0_f1_xai.py",
    "tests\test_audit_saspa_synthetic_a0_f1_finalize.py"
)

function Invoke-NativeChecked {
    param(
        [Parameter(Mandatory = $true)][string]$FilePath,
        [Parameter(Mandatory = $true)][string[]]$Arguments
    )
    & $FilePath @Arguments
    if ($LASTEXITCODE -ne 0) {
        throw "Native command failed with exit code ${LASTEXITCODE}: $FilePath $($Arguments -join ' ')"
    }
}

function Assert-PushedTrackedState {
    $Tracked = (& git -C $RepoRoot status --short --untracked-files=no) -join "`n"
    if ($LASTEXITCODE -ne 0) {
        throw "Unable to inspect tracked Git state."
    }
    if (-not [string]::IsNullOrWhiteSpace($Tracked)) {
        throw "Formal SaSPA F1 finalization requires a clean tracked worktree."
    }
    $Head = (& git -C $RepoRoot rev-parse HEAD).Trim()
    $Upstream = (& git -C $RepoRoot rev-parse '@{upstream}').Trim()
    if ($LASTEXITCODE -ne 0 -or $Head -ne $Upstream) {
        throw "Formal SaSPA F1 finalization requires HEAD to equal upstream."
    }
    $RequiredFiles = @(
        "docs/TRKH_5CLASS_SASPA_F1_FINALIZATION_LOCK_20260724.json",
        "trkh/tools/audit_saspa_synthetic_a0_f1_finalize.py",
        "tests/test_audit_saspa_synthetic_a0_f1_finalize.py",
        "scripts/run_trkh_saspa_synthetic_a0_f1_finalize.ps1"
    )
    foreach ($RequiredFile in $RequiredFiles) {
        & git -C $RepoRoot ls-files --error-unmatch $RequiredFile | Out-Null
        if ($LASTEXITCODE -ne 0) {
            throw "Formal SaSPA F1 finalization requires tracked file: $RequiredFile"
        }
    }
}

function Invoke-StaticPreflight {
    Invoke-NativeChecked -FilePath $Python -Arguments @(
        "-m", "py_compile",
        "trkh\tools\audit_saspa_synthetic_a0_f1_finalize.py",
        "tests\test_audit_saspa_synthetic_a0_f1_finalize.py"
    )
    Invoke-NativeChecked -FilePath $Python -Arguments @(
        "-m", "pyflakes",
        "trkh\tools\audit_saspa_synthetic_a0_f1_finalize.py",
        "tests\test_audit_saspa_synthetic_a0_f1_finalize.py"
    )
    Invoke-NativeChecked -FilePath $Python -Arguments @(
        "-m", "pytest",
        $FocusedTests[0], $FocusedTests[1], $FocusedTests[2], $FocusedTests[3], "-q"
    )
    Invoke-NativeChecked -FilePath $Python -Arguments @(
        "-m", $Module,
        "--preflight-only",
        "--lock", $Lock,
        "--generation-root", $GenerationRoot
    )
}

function Invoke-FormalFinalize {
    Assert-PushedTrackedState
    if (Test-Path -LiteralPath $Summary -PathType Leaf) {
        throw "Final summary already exists and may not be overwritten: $Summary"
    }
    Invoke-NativeChecked -FilePath $Python -Arguments @("-m", "pytest", "-q")
    Invoke-NativeChecked -FilePath $Python -Arguments @(
        "-m", $Module,
        "--formal-finalize",
        "--lock", $Lock,
        "--generation-root", $GenerationRoot
    )
}

function Invoke-Replay {
    if (-not (Test-Path -LiteralPath $Summary -PathType Leaf)) {
        throw "Final summary is missing: $Summary"
    }
    Invoke-NativeChecked -FilePath $Python -Arguments @(
        "-m", $Module,
        "--replay-summary", $Summary
    )
}

$OldPythonPath = $env:PYTHONPATH
$env:PYTHONPATH = $RepoRoot
Push-Location $RepoRoot
try {
    switch ($Phase) {
        "Preflight" {
            Invoke-StaticPreflight
        }
        "Finalize" {
            Invoke-StaticPreflight
            Invoke-FormalFinalize
        }
        "Replay" {
            Invoke-Replay
        }
        "All" {
            Invoke-StaticPreflight
            Invoke-FormalFinalize
            Invoke-Replay
        }
    }
}
finally {
    Pop-Location
    if ($null -eq $OldPythonPath) {
        Remove-Item Env:PYTHONPATH -ErrorAction SilentlyContinue
    }
    else {
        $env:PYTHONPATH = $OldPythonPath
    }
}
