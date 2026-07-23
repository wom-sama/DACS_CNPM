[CmdletBinding()]
param(
    [ValidateSet("Prepare", "Preflight", "Audit", "Replay", "All")]
    [string]$Phase = "Preflight",
    [string]$TestPython = "D:\DataAI\.venv\Scripts\python.exe",
    [string]$RuntimePython = "D:\DataAI\Tools\venvs\trkh_saspa_a0_torch26\Scripts\python.exe",
    [string]$GenerationRoot = "D:\DataAI\AIEx\TRKH\runs\audit_saspa_dual_view_synthetic_a0_20260723\f1_tiny_output_v2",
    [string]$CacheDir = "D:\DataAI\Tools\hf_cache\trkh_saspa_dinov2"
)

Set-StrictMode -Version Latest
$ErrorActionPreference = "Stop"
if (Test-Path variable:PSNativeCommandUseErrorActionPreference) {
    $PSNativeCommandUseErrorActionPreference = $false
}

$RepoRoot = (Resolve-Path (Join-Path $PSScriptRoot "..")).Path
$Module = "trkh.tools.audit_saspa_synthetic_a0_f1_duplicates"
$Lock = "docs\TRKH_5CLASS_SASPA_F1_DUPLICATE_AUDIT_LOCK_20260724.json"
$AuditDir = Join-Path $GenerationRoot "duplicate_audit"
$ModelManifest = Join-Path $AuditDir "dinov2_snapshot_manifest.json"
$Report = Join-Path $AuditDir "duplicate_report.json"
$FocusedTests = @(
    "tests\test_audit_saspa_synthetic_a0_f1.py",
    "tests\test_audit_saspa_synthetic_a0_f1_duplicates.py"
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
        throw "Formal duplicate audit requires a clean tracked worktree."
    }
    $Head = (& git -C $RepoRoot rev-parse HEAD).Trim()
    $Upstream = (& git -C $RepoRoot rev-parse '@{upstream}').Trim()
    if ($LASTEXITCODE -ne 0 -or $Head -ne $Upstream) {
        throw "Formal duplicate audit requires HEAD to equal its upstream."
    }
    $RequiredFiles = @(
        "docs/TRKH_5CLASS_SASPA_F1_DUPLICATE_AUDIT_LOCK_20260724.json",
        "trkh/tools/audit_saspa_synthetic_a0_f1_duplicates.py",
        "tests/test_audit_saspa_synthetic_a0_f1_duplicates.py",
        "scripts/run_trkh_saspa_synthetic_a0_f1_duplicates.ps1"
    )
    foreach ($RequiredFile in $RequiredFiles) {
        & git -C $RepoRoot ls-files --error-unmatch $RequiredFile | Out-Null
        if ($LASTEXITCODE -ne 0) {
            throw "Formal duplicate audit requires tracked file: $RequiredFile"
        }
    }
}

function Show-ResourceState {
    Write-Host "SaSPA F1 duplicate-audit process snapshot (read-only):"
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
    if (-not (Test-Path -LiteralPath $TestPython -PathType Leaf)) {
        throw "TRKH test Python is missing: $TestPython"
    }
    if (-not (Test-Path -LiteralPath $RuntimePython -PathType Leaf)) {
        throw "SaSPA runtime Python is missing: $RuntimePython"
    }
    Invoke-NativeChecked -FilePath $TestPython -Arguments @(
        "-m", "py_compile",
        "trkh\tools\audit_saspa_synthetic_a0_f1_duplicates.py",
        "tests\test_audit_saspa_synthetic_a0_f1_duplicates.py"
    )
    Invoke-NativeChecked -FilePath $TestPython -Arguments @(
        "-m", "pyflakes",
        "trkh\tools\audit_saspa_synthetic_a0_f1_duplicates.py",
        "tests\test_audit_saspa_synthetic_a0_f1_duplicates.py"
    )
    Invoke-NativeChecked -FilePath $TestPython -Arguments @(
        "-m", "pytest",
        $FocusedTests[0], $FocusedTests[1], "-q"
    )
    Invoke-NativeChecked -FilePath $RuntimePython -Arguments @(
        "-m", $Module,
        "--preflight-only",
        "--lock", $Lock,
        "--generation-root", $GenerationRoot,
        "--audit-dir", $AuditDir,
        "--model-manifest", $ModelManifest
    )
}

function Invoke-PrepareModel {
    Assert-PushedTrackedState
    if (Test-Path -LiteralPath $ModelManifest -PathType Leaf) {
        Write-Host "DINOv2 snapshot manifest already exists; preserving it."
        return
    }
    Invoke-NativeChecked -FilePath $RuntimePython -Arguments @(
        "-m", $Module,
        "--prepare-model",
        "--lock", $Lock,
        "--generation-root", $GenerationRoot,
        "--audit-dir", $AuditDir,
        "--cache-dir", $CacheDir,
        "--model-manifest", $ModelManifest
    )
}

function Invoke-FormalAudit {
    Assert-PushedTrackedState
    if (-not (Test-Path -LiteralPath $ModelManifest -PathType Leaf)) {
        throw "DINOv2 snapshot manifest is missing: $ModelManifest"
    }
    if (Test-Path -LiteralPath $Report -PathType Leaf) {
        throw "Duplicate report already exists and may not be overwritten: $Report"
    }
    Show-ResourceState
    Invoke-NativeChecked -FilePath $TestPython -Arguments @(
        "-m", "pytest", "-q"
    )
    Invoke-NativeChecked -FilePath $RuntimePython -Arguments @(
        "-m", $Module,
        "--formal-audit",
        "--lock", $Lock,
        "--generation-root", $GenerationRoot,
        "--audit-dir", $AuditDir,
        "--model-manifest", $ModelManifest
    )
    Show-ResourceState
}

function Invoke-Replay {
    if (-not (Test-Path -LiteralPath $Report -PathType Leaf)) {
        throw "Duplicate report is missing: $Report"
    }
    Invoke-NativeChecked -FilePath $RuntimePython -Arguments @(
        "-m", $Module,
        "--replay-report", $Report
    )
}

$OldPythonPath = $env:PYTHONPATH
$env:PYTHONPATH = $RepoRoot
Push-Location $RepoRoot
try {
    switch ($Phase) {
        "Prepare" {
            Invoke-PrepareModel
        }
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
            Invoke-PrepareModel
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
