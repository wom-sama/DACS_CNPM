[CmdletBinding()]
param(
    [ValidateSet("Preflight", "Engineering", "Formal", "Replay", "VisualPass", "VisualFail", "All")]
    [string]$Phase = "All",
    [string]$Python = "D:\DataAI\.venv\Scripts\python.exe",
    [string]$OutputDir = "D:\DataAI\AIEx\TRKH\runs\audit_jpeg_dl_input_quantization_a0_20260721",
    [string]$ExpectedSummarySha256 = ""
)

Set-StrictMode -Version Latest
$ErrorActionPreference = "Stop"
if (Test-Path variable:PSNativeCommandUseErrorActionPreference) {
    $PSNativeCommandUseErrorActionPreference = $false
}

$RepoRoot = (Resolve-Path (Join-Path $PSScriptRoot "..")).Path
$Module = "trkh.tools.audit_jpeg_dl_input_quantization_a0"
$FocusedTest = "tests\test_audit_jpeg_dl_input_quantization_a0.py"
$SummaryPath = Join-Path $OutputDir "summary.json"

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

function Invoke-PythonChecked {
    param([Parameter(Mandatory = $true)][string[]]$Arguments)
    Invoke-NativeChecked -FilePath $Python -Arguments $Arguments
}

function Show-ResourceState {
    Write-Host "JPEG-DL A0 host process snapshot (read-only):"
    Get-CimInstance Win32_Process |
        Where-Object { $_.Name -in @("python.exe", "pythonw.exe", "trtexec.exe") } |
        Select-Object ProcessId, CreationDate, Name, CommandLine |
        Format-Table -AutoSize
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
    $tracked = (& git -C $RepoRoot status --short --untracked-files=no) -join "`n"
    if ($LASTEXITCODE -ne 0) {
        throw "Unable to inspect tracked git state."
    }
    if (-not [string]::IsNullOrWhiteSpace($tracked)) {
        throw "Formal JPEG-DL A0 requires a clean tracked worktree."
    }
    $head = (& git -C $RepoRoot rev-parse HEAD).Trim()
    $upstream = (& git -C $RepoRoot rev-parse '@{upstream}').Trim()
    if ($LASTEXITCODE -ne 0 -or $head -ne $upstream) {
        throw "Formal JPEG-DL A0 requires HEAD to equal its upstream."
    }
    $requiredFiles = @(
        "trkh/tools/audit_jpeg_dl_input_quantization_a0.py",
        "tests/test_audit_jpeg_dl_input_quantization_a0.py",
        "scripts/run_trkh_jpeg_dl_input_quantization_a0.ps1"
    )
    foreach ($requiredFile in $requiredFiles) {
        & git -C $RepoRoot ls-files --error-unmatch $requiredFile | Out-Null
        if ($LASTEXITCODE -ne 0) {
            throw "Formal JPEG-DL A0 requires tracked file: $requiredFile"
        }
    }
}

function Invoke-StaticPreflight {
    if (-not (Test-Path -LiteralPath $Python -PathType Leaf)) {
        throw "Locked Python executable is missing: $Python"
    }
    if (Test-Path -LiteralPath $OutputDir) {
        throw "Locked formal output already exists: $OutputDir"
    }
    Invoke-PythonChecked -Arguments @(
        "-m", "py_compile",
        "trkh\tools\audit_jpeg_dl_input_quantization_a0.py",
        $FocusedTest
    )
    Invoke-PythonChecked -Arguments @(
        "-m", "pyflakes",
        "trkh\tools\audit_jpeg_dl_input_quantization_a0.py",
        $FocusedTest
    )
    Invoke-PythonChecked -Arguments @("-m", "pytest", $FocusedTest, "-q")
    Invoke-PythonChecked -Arguments @(
        "-m", $Module,
        "--preflight-only",
        "--output-dir", $OutputDir
    )
}

function Invoke-EngineeringForward {
    Show-ResourceState
    Invoke-PythonChecked -Arguments @(
        "-m", $Module,
        "--engineering-forward",
        "--output-dir", $OutputDir
    )
}

function Invoke-FormalAudit {
    Assert-PushedTrackedState
    Show-ResourceState
    Invoke-PythonChecked -Arguments @("-m", "pytest", "-q")
    $oldValues = @{
        "TRKH_JPEG_DL_A0_PREFLIGHT" = $env:TRKH_JPEG_DL_A0_PREFLIGHT
        "PYTHONHASHSEED" = $env:PYTHONHASHSEED
        "CUBLAS_WORKSPACE_CONFIG" = $env:CUBLAS_WORKSPACE_CONFIG
        "OMP_NUM_THREADS" = $env:OMP_NUM_THREADS
        "MKL_NUM_THREADS" = $env:MKL_NUM_THREADS
        "TRKH_ALLOW_WINDOWS_MULTIPROCESSING" = $env:TRKH_ALLOW_WINDOWS_MULTIPROCESSING
        "TRKH_ALLOW_WINDOWS_PIN_MEMORY" = $env:TRKH_ALLOW_WINDOWS_PIN_MEMORY
        "TRKH_ALLOW_WINDOWS_PERSISTENT_WORKERS" = $env:TRKH_ALLOW_WINDOWS_PERSISTENT_WORKERS
    }
    $env:TRKH_JPEG_DL_A0_PREFLIGHT = "passed"
    $env:PYTHONHASHSEED = "42"
    $env:CUBLAS_WORKSPACE_CONFIG = ":4096:8"
    $env:OMP_NUM_THREADS = "1"
    $env:MKL_NUM_THREADS = "1"
    $env:TRKH_ALLOW_WINDOWS_MULTIPROCESSING = "1"
    $env:TRKH_ALLOW_WINDOWS_PIN_MEMORY = "1"
    $env:TRKH_ALLOW_WINDOWS_PERSISTENT_WORKERS = "1"
    try {
        Invoke-PythonChecked -Arguments @(
            "-m", $Module,
            "--output-dir", $OutputDir
        )
    }
    finally {
        foreach ($name in $oldValues.Keys) {
            $value = $oldValues[$name]
            if ($null -eq $value) {
                Remove-Item "Env:$name" -ErrorAction SilentlyContinue
            }
            else {
                Set-Item "Env:$name" $value
            }
        }
    }
    Show-ResourceState
}

function Invoke-ExternalReplay {
    if (-not (Test-Path -LiteralPath $SummaryPath -PathType Leaf)) {
        throw "JPEG-DL A0 summary is missing: $SummaryPath"
    }
    Invoke-PythonChecked -Arguments @(
        "-m", $Module,
        "--replay-summary", $SummaryPath
    )
}

function Invoke-VisualFinalization {
    param([Parameter(Mandatory = $true)][ValidateSet("pass", "fail")][string]$Decision)
    if ([string]::IsNullOrWhiteSpace($ExpectedSummarySha256)) {
        throw "Visual finalization requires -ExpectedSummarySha256 from replay output."
    }
    Invoke-PythonChecked -Arguments @(
        "-m", $Module,
        "--replay-summary", $SummaryPath,
        "--finalize-visual-review", $Decision,
        "--expected-summary-sha256", $ExpectedSummarySha256
    )
}

Push-Location $RepoRoot
try {
    switch ($Phase) {
        "Preflight" {
            Invoke-StaticPreflight
        }
        "Engineering" {
            Invoke-StaticPreflight
            Invoke-EngineeringForward
        }
        "Formal" {
            Invoke-StaticPreflight
            Invoke-EngineeringForward
            Invoke-FormalAudit
        }
        "Replay" {
            Invoke-ExternalReplay
        }
        "VisualPass" {
            Invoke-VisualFinalization -Decision "pass"
        }
        "VisualFail" {
            Invoke-VisualFinalization -Decision "fail"
        }
        "All" {
            Invoke-StaticPreflight
            Invoke-EngineeringForward
            Invoke-FormalAudit
            Invoke-ExternalReplay
        }
    }
}
finally {
    Pop-Location
}
