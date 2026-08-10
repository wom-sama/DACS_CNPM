[CmdletBinding()]
param(
    [ValidateSet("Preflight", "Audit", "Replay", "All")]
    [string]$Phase = "Preflight",
    [string]$Python = "D:\DataAI\.venv\Scripts\python.exe",
    [string]$GenerationRoot = "D:\DataAI\AIEx\TRKH\runs\audit_saspa_dual_view_synthetic_a0_20260723\f1_tiny_output_v2",
    [string]$Checkpoint = "D:\DataAI\AIEx\TRKH\runs\probe_v8_yolof_pairroute_teacherfocusbinary015_boundarydrop_bboxprior_120b_2e_20260701\checkpoints\best.pt"
)

Set-StrictMode -Version Latest
$ErrorActionPreference = "Stop"
if (Test-Path variable:PSNativeCommandUseErrorActionPreference) {
    $PSNativeCommandUseErrorActionPreference = $false
}

$RepoRoot = (Resolve-Path (Join-Path $PSScriptRoot "..")).Path
$Module = "trkh.tools.audit_saspa_synthetic_a0_f1_xai"
$Lock = "docs\TRKH_5CLASS_SASPA_F1_XAI_EXECUTION_LOCK_20260724.json"
$OutputDir = Join-Path $GenerationRoot "xai_audit"
$Summary = Join-Path $OutputDir "xai_summary.json"
$FocusedTests = @(
    "tests\test_audit_saspa_synthetic_a0_f1.py",
    "tests\test_audit_saspa_synthetic_a0_f1_duplicates.py",
    "tests\test_audit_saspa_synthetic_a0_f1_xai.py"
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
        throw "Formal SaSPA F1 XAI requires a clean tracked worktree."
    }
    $Head = (& git -C $RepoRoot rev-parse HEAD).Trim()
    $Upstream = (& git -C $RepoRoot rev-parse '@{upstream}').Trim()
    if ($LASTEXITCODE -ne 0 -or $Head -ne $Upstream) {
        throw "Formal SaSPA F1 XAI requires HEAD to equal its upstream."
    }
    $RequiredFiles = @(
        "docs/TRKH_5CLASS_SASPA_F1_XAI_EXECUTION_LOCK_20260724.json",
        "trkh/tools/audit_saspa_synthetic_a0_f1_xai.py",
        "tests/test_audit_saspa_synthetic_a0_f1_xai.py",
        "scripts/run_trkh_saspa_synthetic_a0_f1_xai.ps1"
    )
    foreach ($RequiredFile in $RequiredFiles) {
        & git -C $RepoRoot ls-files --error-unmatch $RequiredFile | Out-Null
        if ($LASTEXITCODE -ne 0) {
            throw "Formal SaSPA F1 XAI requires tracked file: $RequiredFile"
        }
    }
}

function Show-ResourceState {
    Write-Host "SaSPA F1 XAI process snapshot (read-only):"
    Get-CimInstance Win32_Process |
        Where-Object { $_.Name -in @("python.exe", "pythonw.exe", "trtexec.exe") } |
        Select-Object ProcessId, CreationDate, Name, CommandLine |
        Format-Table -Wrap -AutoSize
    if (Get-Command nvidia-smi -ErrorAction SilentlyContinue) {
        Invoke-NativeChecked -FilePath "nvidia-smi" -Arguments @(
            "--query-gpu=name,memory.total,memory.used,utilization.gpu,temperature.gpu,power.draw",
            "--format=csv,noheader,nounits"
        )
    }
}

function Invoke-StaticPreflight {
    if (-not (Test-Path -LiteralPath $Python -PathType Leaf)) {
        throw "TRKH Python is missing: $Python"
    }
    Invoke-NativeChecked -FilePath $Python -Arguments @(
        "-m", "py_compile",
        "trkh\tools\audit_saspa_synthetic_a0_f1_xai.py",
        "tests\test_audit_saspa_synthetic_a0_f1_xai.py"
    )
    Invoke-NativeChecked -FilePath $Python -Arguments @(
        "-m", "pyflakes",
        "trkh\tools\audit_saspa_synthetic_a0_f1_xai.py",
        "tests\test_audit_saspa_synthetic_a0_f1_xai.py"
    )
    Invoke-NativeChecked -FilePath $Python -Arguments @(
        "-m", "pytest",
        $FocusedTests[0], $FocusedTests[1], $FocusedTests[2], "-q"
    )
    Invoke-NativeChecked -FilePath $Python -Arguments @(
        "-m", $Module,
        "--preflight-only",
        "--lock", $Lock,
        "--generation-root", $GenerationRoot,
        "--checkpoint", $Checkpoint,
        "--output-dir", $OutputDir
    )
}

function Invoke-FormalAudit {
    Assert-PushedTrackedState
    if (Test-Path -LiteralPath $Summary -PathType Leaf) {
        throw "XAI summary already exists and may not be overwritten: $Summary"
    }
    Show-ResourceState
    Invoke-NativeChecked -FilePath $Python -Arguments @("-m", "pytest", "-q")
    Invoke-NativeChecked -FilePath $Python -Arguments @(
        "-m", $Module,
        "--formal-audit",
        "--lock", $Lock,
        "--generation-root", $GenerationRoot,
        "--checkpoint", $Checkpoint,
        "--output-dir", $OutputDir
    )
    Show-ResourceState
}

function Invoke-Replay {
    if (-not (Test-Path -LiteralPath $Summary -PathType Leaf)) {
        throw "XAI summary is missing: $Summary"
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
        "Audit" {
            Invoke-StaticPreflight
            Invoke-FormalAudit
        }
        "Replay" {
            Invoke-Replay
        }
        "All" {
            Invoke-StaticPreflight
            Invoke-FormalAudit
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
