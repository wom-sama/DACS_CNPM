param(
    [string]$Python = "D:\DataAI\.venv\Scripts\python.exe",
    [ValidateSet("Preflight", "Formal", "Replay", "Finalize")]
    [string]$Phase = "Preflight",
    [string]$OutputDir = "runs\audit_axial_color_topology_a0_20260720",
    [ValidateSet("pass", "fail")]
    [string]$VisualReviewResult = "fail",
    [string]$VisualReviewNote = "",
    [bool]$PauseWallpaperEngine = $true
)

$ErrorActionPreference = "Stop"
$RepoRoot = Split-Path -Parent $PSScriptRoot
Set-Location -LiteralPath $RepoRoot

if (-not (Test-Path -LiteralPath $Python -PathType Leaf)) {
    throw "Python executable not found: $Python"
}

function Invoke-LockedPython {
    param([string[]]$Arguments)
    & $Python @Arguments
    if ($LASTEXITCODE -ne 0) {
        throw "Axial A0 command failed with exit code ${LASTEXITCODE}: $($Arguments -join ' ')"
    }
}

function Get-ComputeProcesses {
    return Get-CimInstance Win32_Process | Where-Object {
        $_.Name -in @("python.exe", "pythonw.exe", "trtexec.exe", "ffmpeg.exe")
    } | Select-Object ProcessId, Name, CreationDate, CommandLine
}

function Show-ComputeState {
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
        throw "Formal axial A0 requires no pre-existing Python/TensorRT/FFmpeg process."
    }
    if (-not (Get-Command nvidia-smi -ErrorAction SilentlyContinue)) {
        throw "nvidia-smi is required for the formal GPU isolation gate."
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
    Write-Host "GPU utilization samples: $($Samples -join ', ') percent"
    if (($Samples | Measure-Object -Maximum).Maximum -gt 20) {
        throw "Formal axial A0 is blocked because background GPU utilization exceeds 20 percent."
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
        return $false
    }
    Start-Process -FilePath $Executable -ArgumentList @("-control", $State) `
        -WorkingDirectory (Split-Path -Parent $Executable) -Wait -WindowStyle Hidden
    return $true
}

if ($Phase -in @("Preflight", "Formal")) {
    Invoke-LockedPython -Arguments @(
        "-m", "py_compile",
        "trkh\tools\audit_axial_color_topology_a0.py",
        "tests\test_audit_axial_color_topology_a0.py"
    )
    Invoke-LockedPython -Arguments @(
        "-m", "pyflakes",
        "trkh\tools\audit_axial_color_topology_a0.py",
        "tests\test_audit_axial_color_topology_a0.py"
    )
    Invoke-LockedPython -Arguments @(
        "-m", "pytest",
        "tests\test_audit_axial_color_topology_a0.py",
        "-q"
    )
}

if ($Phase -eq "Preflight") {
    if (Test-Path -LiteralPath $OutputDir) {
        throw "Preflight must not create or reuse output: $OutputDir"
    }
    Show-ComputeState
    Invoke-LockedPython -Arguments @(
        "-m", "trkh.tools.audit_axial_color_topology_a0",
        "--preflight-only",
        "--output-dir", $OutputDir
    )
    Write-Host "Axial A0 preflight passed without dataset/model loading or output creation."
    exit 0
}

if ($Phase -eq "Formal") {
    if (Test-Path -LiteralPath $OutputDir) {
        throw "Formal axial A0 output already exists; rerun is forbidden: $OutputDir"
    }
    Show-ComputeState
    $WallpaperPaused = $false
    try {
        if ($PauseWallpaperEngine) {
            $WallpaperPaused = Set-WallpaperEngineState -State "pause"
        }
        Assert-GpuIsolation
        Invoke-LockedPython -Arguments @(
            "-m", "trkh.tools.audit_axial_color_topology_a0",
            "--output-dir", $OutputDir
        )
        $SummaryPath = Join-Path $OutputDir "summary.json"
        Invoke-LockedPython -Arguments @(
            "-m", "trkh.tools.audit_axial_color_topology_a0",
            "--replay-summary", $SummaryPath
        )
        $Summary = Get-Content -Raw -LiteralPath $SummaryPath | ConvertFrom-Json
        $SummarySha = (Get-FileHash -Algorithm SHA256 -LiteralPath $SummaryPath).Hash.ToLowerInvariant()
        Write-Host "Axial A0 status: $($Summary.status)"
        Write-Host "Summary SHA-256: $SummarySha"
    } finally {
        if ($WallpaperPaused) {
            [void](Set-WallpaperEngineState -State "play")
        }
        Show-ComputeState
    }
    exit 0
}

$SummaryPath = Join-Path $OutputDir "summary.json"
if (-not (Test-Path -LiteralPath $SummaryPath -PathType Leaf)) {
    throw "Axial A0 summary is missing: $SummaryPath"
}

if ($Phase -eq "Replay") {
    Invoke-LockedPython -Arguments @(
        "-m", "trkh.tools.audit_axial_color_topology_a0",
        "--replay-summary", $SummaryPath
    )
    exit 0
}

if ([string]::IsNullOrWhiteSpace($VisualReviewNote)) {
    throw "Finalize requires -VisualReviewNote covering all four contact sheets."
}
$SummarySha = (Get-FileHash -Algorithm SHA256 -LiteralPath $SummaryPath).Hash.ToLowerInvariant()
Invoke-LockedPython -Arguments @(
    "-m", "trkh.tools.audit_axial_color_topology_a0",
    "--output-dir", $OutputDir,
    "--finalize-visual-review",
    "--visual-review-result", $VisualReviewResult,
    "--visual-review-note", $VisualReviewNote,
    "--expected-summary-sha256", $SummarySha
)
Write-Host "Axial A0 visual review finalized from summary SHA-256 $SummarySha."
