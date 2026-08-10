[CmdletBinding()]
param(
    [string]$BasePython = "C:\Users\ADMIN\AppData\Local\Programs\Python\Python311\python.exe",
    [string]$VenvRoot = "D:\DataAI\Tools\venvs\trkh_saspa_a0_torch26",
    [string]$Requirements = ""
)

Set-StrictMode -Version Latest
$ErrorActionPreference = "Stop"
if (Test-Path variable:PSNativeCommandUseErrorActionPreference) {
    $PSNativeCommandUseErrorActionPreference = $false
}

$RepoRoot = (Resolve-Path (Join-Path $PSScriptRoot "..")).Path
if ([string]::IsNullOrWhiteSpace($Requirements)) {
    $Requirements = Join-Path $RepoRoot "requirements\trkh_saspa_a0.txt"
}

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

if (-not (Test-Path -LiteralPath $BasePython -PathType Leaf)) {
    throw "Base Python is missing: $BasePython"
}
if (-not (Test-Path -LiteralPath $Requirements -PathType Leaf)) {
    throw "SaSPA requirements lock is missing: $Requirements"
}

$RuntimePython = Join-Path $VenvRoot "Scripts\python.exe"
if (-not (Test-Path -LiteralPath $RuntimePython -PathType Leaf)) {
    if (Test-Path -LiteralPath $VenvRoot) {
        throw "Refusing to reuse an incomplete runtime directory: $VenvRoot"
    }
    Invoke-NativeChecked -FilePath $BasePython -Arguments @(
        "-m", "venv", $VenvRoot
    )
}

$ConfigPath = Join-Path $VenvRoot "pyvenv.cfg"
$ConfigText = Get-Content -LiteralPath $ConfigPath -Raw
if ($ConfigText -notmatch "include-system-site-packages\s*=\s*false") {
    throw "SaSPA A0 runtime must not inherit global site-packages: $ConfigPath"
}

Invoke-NativeChecked -FilePath $RuntimePython -Arguments @(
    "-m", "pip", "install",
    "--disable-pip-version-check",
    "--no-input",
    "--requirement", $Requirements
)
Invoke-NativeChecked -FilePath $RuntimePython -Arguments @("-m", "pip", "check")

$OldPythonPath = $env:PYTHONPATH
$env:PYTHONPATH = $RepoRoot
try {
    Invoke-NativeChecked -FilePath $RuntimePython -Arguments @(
        "-m", "trkh.tools.audit_saspa_synthetic_a0_f0",
        "--runtime-check-only",
        "--requirements", $Requirements
    )
}
finally {
    if ($null -eq $OldPythonPath) {
        Remove-Item Env:PYTHONPATH -ErrorAction SilentlyContinue
    }
    else {
        $env:PYTHONPATH = $OldPythonPath
    }
}

Write-Host "SaSPA A0 isolated runtime is ready: $RuntimePython"
