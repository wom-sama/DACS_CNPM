[CmdletBinding()]
param(
    [ValidateSet("Preflight", "Generate", "Replay", "All")]
    [string]$Phase = "Preflight",
    [string]$TestPython = "D:\DataAI\.venv\Scripts\python.exe",
    [string]$RuntimePython = "D:\DataAI\Tools\venvs\trkh_saspa_a0_torch26\Scripts\python.exe",
    [string]$OutputDir = "D:\DataAI\AIEx\TRKH\runs\audit_saspa_dual_view_synthetic_a0_20260723\f1_tiny_output"
)

Set-StrictMode -Version Latest
$ErrorActionPreference = "Stop"
if (Test-Path variable:PSNativeCommandUseErrorActionPreference) {
    $PSNativeCommandUseErrorActionPreference = $false
}

$RepoRoot = (Resolve-Path (Join-Path $PSScriptRoot "..")).Path
$Module = "trkh.tools.audit_saspa_synthetic_a0_f1"
$Protocol = "docs\TRKH_5CLASS_SASPA_SYNTHETIC_A0_PROTOCOL_20260723.json"
$ImplementationLock = "docs\TRKH_5CLASS_SASPA_F1_IMPLEMENTATION_LOCK_20260724.json"
$Requirements = "requirements\trkh_saspa_a0.txt"
$F0Root = "runs\audit_saspa_dual_view_synthetic_a0_20260723"
$FocusedTests = @(
    "tests\test_audit_saspa_synthetic_a0_f0.py",
    "tests\test_audit_saspa_synthetic_a0_f1.py"
)
$SummaryPath = Join-Path $OutputDir "generation_summary.json"

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

function Show-ResourceState {
    Write-Host "SaSPA A0 F1 host process snapshot (read-only):"
    Get-CimInstance Win32_Process |
        Where-Object { $_.Name -in @("python.exe", "pythonw.exe", "trtexec.exe") } |
        Select-Object ProcessId, CreationDate, Name, CommandLine |
        Format-Table -Wrap -AutoSize
    if (Get-Command nvidia-smi -ErrorAction SilentlyContinue) {
        Invoke-NativeChecked -FilePath "nvidia-smi" -Arguments @(
            "--query-gpu=name,memory.total,memory.used,utilization.gpu,temperature.gpu,power.draw",
            "--format=csv,noheader,nounits"
        )
        & nvidia-smi --query-compute-apps=pid,process_name,used_memory --format=csv,noheader,nounits
        if ($LASTEXITCODE -ne 0) {
            Write-Warning "nvidia-smi compute-process query is unavailable on this Windows driver."
        }
    }
}

function Assert-PushedTrackedState {
    $Tracked = (& git -C $RepoRoot status --short --untracked-files=no) -join "`n"
    if ($LASTEXITCODE -ne 0) {
        throw "Unable to inspect tracked Git state."
    }
    if (-not [string]::IsNullOrWhiteSpace($Tracked)) {
        throw "Formal SaSPA F1 requires a clean tracked worktree."
    }
    $Head = (& git -C $RepoRoot rev-parse HEAD).Trim()
    $Upstream = (& git -C $RepoRoot rev-parse '@{upstream}').Trim()
    if ($LASTEXITCODE -ne 0 -or $Head -ne $Upstream) {
        throw "Formal SaSPA F1 requires HEAD to equal its upstream."
    }
    $RequiredFiles = @(
        "docs/TRKH_5CLASS_SASPA_F1_IMPLEMENTATION_LOCK_20260724.json",
        "docs/TRKH_5CLASS_SASPA_F1_IMPLEMENTATION_LOCK_20260724.md",
        "docs/TRKH_5CLASS_SASPA_F1_IMPLEMENTATION_LOCK_20260724.sha256",
        "trkh/tools/audit_saspa_synthetic_a0_f1.py",
        "tests/test_audit_saspa_synthetic_a0_f1.py",
        "scripts/run_trkh_saspa_synthetic_a0_f1.ps1"
    )
    foreach ($RequiredFile in $RequiredFiles) {
        & git -C $RepoRoot ls-files --error-unmatch $RequiredFile | Out-Null
        if ($LASTEXITCODE -ne 0) {
            throw "Formal SaSPA F1 requires tracked file: $RequiredFile"
        }
    }
}

function Invoke-StaticPreflight {
    if (-not (Test-Path -LiteralPath $TestPython -PathType Leaf)) {
        throw "TRKH test Python is missing: $TestPython"
    }
    if (-not (Test-Path -LiteralPath $RuntimePython -PathType Leaf)) {
        throw "SaSPA runtime is missing. Run scripts\setup_trkh_saspa_a0_env.ps1 first."
    }
    Invoke-NativeChecked -FilePath $TestPython -Arguments @(
        "-m", "py_compile",
        "trkh\tools\audit_saspa_synthetic_a0_f1.py",
        "tests\test_audit_saspa_synthetic_a0_f1.py"
    )
    Invoke-NativeChecked -FilePath $TestPython -Arguments @(
        "-m", "pyflakes",
        "trkh\tools\audit_saspa_synthetic_a0_f1.py",
        "tests\test_audit_saspa_synthetic_a0_f1.py"
    )
    Invoke-NativeChecked -FilePath $TestPython -Arguments @(
        "-m", "pytest",
        $FocusedTests[0], $FocusedTests[1], "-q"
    )
    Invoke-NativeChecked -FilePath $RuntimePython -Arguments @(
        "-m", $Module,
        "--preflight-only",
        "--protocol", $Protocol,
        "--implementation-lock", $ImplementationLock,
        "--requirements", $Requirements,
        "--f0-root", $F0Root
    )
}

function Invoke-FormalGeneration {
    Assert-PushedTrackedState
    if (Test-Path -LiteralPath $OutputDir) {
        throw "F1 output already exists and may not be overwritten: $OutputDir"
    }
    Show-ResourceState
    Invoke-NativeChecked -FilePath $TestPython -Arguments @(
        "-m", "pytest", "-q"
    )
    Invoke-NativeChecked -FilePath $RuntimePython -Arguments @(
        "-m", $Module,
        "--formal-generate",
        "--protocol", $Protocol,
        "--implementation-lock", $ImplementationLock,
        "--requirements", $Requirements,
        "--f0-root", $F0Root,
        "--output-dir", $OutputDir
    )
    Show-ResourceState
}

function Invoke-Replay {
    if (-not (Test-Path -LiteralPath $SummaryPath -PathType Leaf)) {
        throw "SaSPA F1 generation summary is missing: $SummaryPath"
    }
    Invoke-NativeChecked -FilePath $RuntimePython -Arguments @(
        "-m", $Module,
        "--replay-summary", $SummaryPath
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
        "Generate" {
            Invoke-StaticPreflight
            Invoke-FormalGeneration
        }
        "Replay" {
            Invoke-Replay
        }
        "All" {
            Invoke-StaticPreflight
            Invoke-FormalGeneration
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
