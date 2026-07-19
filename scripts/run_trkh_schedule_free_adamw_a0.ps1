param(
    [string]$Python = "D:\DataAI\.venv\Scripts\python.exe",
    [ValidateSet("Preflight", "Formal", "Replay", "Finalize")]
    [string]$Phase = "Preflight",
    [string]$OutputDir = "runs\audit_schedule_free_adamw_a0_20260720",
    [ValidateSet("pass", "fail")]
    [string]$VisualReviewResult = "fail",
    [string]$VisualReviewNote = "",
    [bool]$PauseWallpaperEngine = $true
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
        throw "Schedule-Free command failed with exit code ${LASTEXITCODE}: $($Arguments -join ' ')"
    }
}

function Get-ComputeProcesses {
    return Get-CimInstance Win32_Process | Where-Object {
        $_.Name -in @("python.exe", "pythonw.exe", "trtexec.exe", "ffmpeg.exe")
    } | Select-Object ProcessId, Name, CreationDate, CommandLine
}

function Show-LocalComputeState {
    $Processes = Get-ComputeProcesses
    if ($Processes) {
        Write-Host "Existing compute processes were detected; none will be stopped:"
        $Processes | Format-Table -AutoSize | Out-Host
    } else {
        Write-Host "No existing Python/TensorRT/FFmpeg process was detected."
    }
    if (Get-Command nvidia-smi -ErrorAction SilentlyContinue) {
        & nvidia-smi --query-gpu=utilization.gpu,memory.used,memory.total,temperature.gpu `
            --format=csv,noheader,nounits
    }
}

function Assert-GpuIsolation {
    $Processes = Get-ComputeProcesses
    if ($Processes) {
        $Processes | Format-Table -AutoSize | Out-Host
        throw "Formal Schedule-Free timing requires no pre-existing Python/TensorRT/FFmpeg process."
    }
    if (-not (Get-Command nvidia-smi -ErrorAction SilentlyContinue)) {
        throw "nvidia-smi is required for the locked Schedule-Free timing gate."
    }
    $Samples = @()
    for ($Index = 0; $Index -lt 3; $Index++) {
        $Raw = & nvidia-smi --query-gpu=utilization.gpu --format=csv,noheader,nounits
        if ($LASTEXITCODE -ne 0) {
            throw "nvidia-smi utilization query failed."
        }
        $Samples += [int]($Raw.Trim())
        if ($Index -lt 2) {
            Start-Sleep -Seconds 1
        }
    }
    $Maximum = ($Samples | Measure-Object -Maximum).Maximum
    Write-Host "GPU isolation utilization samples: $($Samples -join ', ') percent"
    if ($Maximum -gt 10) {
        throw "Formal Schedule-Free timing is blocked because background GPU utilization exceeds 10 percent."
    }
}

function Set-WallpaperEngineState {
    param([ValidateSet("pause", "play")][string]$State)
    $Process = Get-Process -Name "wallpaper64", "wallpaper32" -ErrorAction SilentlyContinue |
        Select-Object -First 1
    if (-not $Process) {
        return $false
    }
    $Executable = $Process.Path
    if (-not $Executable -or -not (Test-Path -LiteralPath $Executable -PathType Leaf)) {
        Write-Warning "Wallpaper Engine is running, but its executable path is unavailable."
        return $false
    }
    Start-Process -FilePath $Executable -ArgumentList @("-control", $State) `
        -WorkingDirectory (Split-Path -Parent $Executable) -Wait -WindowStyle Hidden
    return $true
}

if ($Phase -in @("Preflight", "Formal")) {
    Invoke-LockedPython -Arguments @(
        "-m", "py_compile",
        "trkh\tools\audit_schedule_free_adamw_a0.py",
        "tests\test_audit_schedule_free_adamw_a0.py"
    )
    Invoke-LockedPython -Arguments @(
        "-m", "pyflakes",
        "trkh\tools\audit_schedule_free_adamw_a0.py",
        "tests\test_audit_schedule_free_adamw_a0.py"
    )
    Invoke-LockedPython -Arguments @(
        "-m", "pytest",
        "tests\test_audit_schedule_free_adamw_a0.py",
        "-q"
    )
}

if ($Phase -eq "Preflight") {
    if (Test-Path -LiteralPath $OutputDir) {
        throw "Preflight must not create or reuse an output directory: $OutputDir"
    }
    Show-LocalComputeState
    Invoke-LockedPython -Arguments @(
        "-m", "trkh.tools.audit_schedule_free_adamw_a0",
        "--preflight-only"
    )
    Write-Host "Schedule-Free A0 preflight passed without dataset/model inference or output creation."
    exit 0
}

if ($Phase -eq "Formal") {
    if (Test-Path -LiteralPath $OutputDir) {
        throw "Formal Schedule-Free output already exists; overwrite/rerun is forbidden: $OutputDir"
    }
    Show-LocalComputeState
    $WallpaperPaused = $false
    try {
        if ($PauseWallpaperEngine) {
            $WallpaperPaused = Set-WallpaperEngineState -State "pause"
        }
        Assert-GpuIsolation
        Invoke-LockedPython -Arguments @(
            "-m", "trkh.tools.audit_schedule_free_adamw_a0",
            "--output-dir", $OutputDir
        )
        $SummaryPath = Join-Path $OutputDir "summary.json"
        if (-not (Test-Path -LiteralPath $SummaryPath -PathType Leaf)) {
            throw "Formal Schedule-Free audit did not create summary.json."
        }
        Invoke-LockedPython -Arguments @(
            "-m", "trkh.tools.audit_schedule_free_adamw_a0",
            "--replay-summary", $SummaryPath
        )
        $Summary = Get-Content -Raw -LiteralPath $SummaryPath | ConvertFrom-Json
        $SummarySha = (Get-FileHash -Algorithm SHA256 -LiteralPath $SummaryPath).Hash.ToLowerInvariant()
        Write-Host "Schedule-Free formal status: $($Summary.status)"
        Write-Host "Locked pre-review summary SHA-256: $SummarySha"
    } finally {
        if ($WallpaperPaused) {
            [void](Set-WallpaperEngineState -State "play")
        }
        Show-LocalComputeState
    }
    exit 0
}

$SummaryPath = Join-Path $OutputDir "summary.json"
if (-not (Test-Path -LiteralPath $SummaryPath -PathType Leaf)) {
    throw "Formal Schedule-Free summary is missing: $SummaryPath"
}

if ($Phase -eq "Replay") {
    Invoke-LockedPython -Arguments @(
        "-m", "trkh.tools.audit_schedule_free_adamw_a0",
        "--replay-summary", $SummaryPath
    )
    exit 0
}

if ([string]::IsNullOrWhiteSpace($VisualReviewNote)) {
    throw "Finalize requires -VisualReviewNote covering all four contact sheets."
}
$Summary = Get-Content -Raw -LiteralPath $SummaryPath | ConvertFrom-Json
if ($Summary.status -ne "awaiting_visual_review") {
    throw "Schedule-Free summary is not awaiting visual review: $($Summary.status)"
}
$SummarySha = (Get-FileHash -Algorithm SHA256 -LiteralPath $SummaryPath).Hash.ToLowerInvariant()
Invoke-LockedPython -Arguments @(
    "-m", "trkh.tools.audit_schedule_free_adamw_a0",
    "--output-dir", $OutputDir,
    "--finalize-visual-review",
    "--visual-review-result", $VisualReviewResult,
    "--visual-review-note", $VisualReviewNote,
    "--expected-summary-sha256", $SummarySha
)
Write-Host "Schedule-Free visual review finalized from summary SHA-256 $SummarySha."
