param(
    [string]$Python = "D:\DataAI\.venv\Scripts\python.exe",
    [string]$DataYaml = "D:\DataAI\AIEx\newdataset\yolo_f\data.yaml",
    [string]$KeeperCheckpoint = "runs\probe_v8_yolof_pairroute_teacherfocusbinary015_boundarydrop_bboxprior_120b_2e_20260701\checkpoints\best.pt",
    [string]$CandidateCheckpoint = "runs\full_v8_yolof_randominit_30e_20260714_105524\checkpoints\best.pt",
    [string]$FrozenSummary = "runs\full_v8_yolof_randominit_30e_20260714_105524\final_audit_manual\precision_ensemble_readiness_val_20260714\summary.json",
    [string]$XaiCaseCsv = "runs\full_v8_yolof_randominit_30e_20260714_105524\final_audit_manual\precision_ensemble_readiness_val_20260714\xai_priority_cases.csv",
    [string]$RunName = ("precision_ensemble_certified_{0}" -f (Get-Date -Format "yyyyMMdd_HHmmss")),
    [int]$XaiCases = 12,
    [switch]$PreflightOnly
)

$ErrorActionPreference = "Stop"
$ProjectRoot = Split-Path -Parent $PSScriptRoot
Set-Location $ProjectRoot

$ExpectedKeeperSha256 = "1f49d577240c69dc63c30af70db52ec2aa9da65a17aef1c4b1c09ece6c482677"
$ExpectedCandidateSha256 = "f8bd6309b9820d2b2d6db28815a9a01e11ebbaf0bd5eb6c5ce097b6aa081a549"
$ExpectedFrozenSummarySha256 = "6c6397d3561ec917d9716ecfd2c6b65a616433e3a07c45031df898578d07a8c8"

function Resolve-RequiredPath {
    param([string]$Value, [string]$Label)
    if ([string]::IsNullOrWhiteSpace($Value)) {
        throw "$Label must not be empty."
    }
    $candidate = if ([System.IO.Path]::IsPathRooted($Value)) {
        $Value
    }
    else {
        Join-Path $ProjectRoot $Value
    }
    $fullPath = [System.IO.Path]::GetFullPath($candidate)
    if (-not (Test-Path -LiteralPath $fullPath -PathType Leaf)) {
        throw "$Label not found: $fullPath"
    }
    return $fullPath
}

function Assert-FileSha256 {
    param([string]$Path, [string]$Expected, [string]$Label)
    $observed = (Get-FileHash -LiteralPath $Path -Algorithm SHA256).Hash.ToLowerInvariant()
    if ($observed -ne $Expected.ToLowerInvariant()) {
        throw "$Label SHA-256 mismatch: expected=$Expected observed=$observed path=$Path"
    }
    return $observed
}

function Write-JsonNoBom {
    param([object]$Value, [string]$Path, [int]$Depth = 10)
    $parent = Split-Path -Parent $Path
    if (-not [string]::IsNullOrWhiteSpace($parent)) {
        New-Item -ItemType Directory -Path $parent -Force | Out-Null
    }
    $json = $Value | ConvertTo-Json -Depth $Depth
    $utf8 = New-Object System.Text.UTF8Encoding($false)
    [System.IO.File]::WriteAllText($Path, $json, $utf8)
}

$script:StepRecords = New-Object System.Collections.ArrayList

function Invoke-PythonStep {
    param([string]$Name, [string[]]$Arguments)
    Write-Host "`n=== $Name ===" -ForegroundColor Cyan
    $started = Get-Date
    $previousErrorActionPreference = $ErrorActionPreference
    $exitCode = 1
    try {
        $ErrorActionPreference = "Continue"
        & $PythonPath @Arguments
        $exitCode = $LASTEXITCODE
    }
    finally {
        $ErrorActionPreference = $previousErrorActionPreference
    }
    $seconds = ((Get-Date) - $started).TotalSeconds
    [void]$script:StepRecords.Add([ordered]@{
        name = $Name
        exit_code = $exitCode
        seconds = [math]::Round($seconds, 3)
    })
    if ($exitCode -ne 0) {
        throw "$Name failed with exit code $exitCode."
    }
}

function Assert-PassedSummary {
    param([string]$Path, [string]$Label)
    if (-not (Test-Path -LiteralPath $Path -PathType Leaf)) {
        throw "$Label summary not found: $Path"
    }
    $summary = Get-Content -LiteralPath $Path -Raw | ConvertFrom-Json
    if (-not [bool]$summary.all_gates_passed) {
        throw "$Label did not pass all gates: $Path"
    }
    return $summary
}

function Assert-XaiSummary {
    param(
        [string]$Path,
        [string]$Member,
        [string]$PackageSha256,
        [string]$SourceSha256
    )
    if (-not (Test-Path -LiteralPath $Path -PathType Leaf)) {
        throw "XAI summary not found: $Path"
    }
    $summary = Get-Content -LiteralPath $Path -Raw | ConvertFrom-Json
    $provenance = $summary.ensemble_member_provenance
    if (-not [bool]$provenance.enabled) {
        throw "XAI ensemble provenance is not enabled for $Member."
    }
    if ([string]$provenance.selected_member -ne $Member) {
        throw "XAI selected member mismatch: expected=$Member observed=$($provenance.selected_member)"
    }
    if ([string]$provenance.package_checkpoint_sha256 -ne $PackageSha256) {
        throw "XAI package SHA-256 mismatch for $Member."
    }
    if ([string]$provenance.member_source_checkpoint_sha256 -ne $SourceSha256) {
        throw "XAI source-member SHA-256 mismatch for $Member."
    }
    if ([bool]$provenance.aggregate_attention_produced) {
        throw "XAI must not label an aggregate map as ensemble attention."
    }
    if (-not [bool]$provenance.xai_autograd.in_memory_only -or [bool]$provenance.xai_autograd.optimizer_step_performed) {
        throw "XAI autograd provenance is not inference-only for $Member."
    }
    if ([int]$summary.selected_cases -ne $XaiCases) {
        throw "XAI case count mismatch for ${Member}: expected=$XaiCases observed=$($summary.selected_cases)"
    }
    return $summary
}

if ($RunName -notmatch '^[A-Za-z0-9_.-]+$') {
    throw "RunName may contain only letters, digits, dot, underscore, and dash: $RunName"
}
if ($XaiCases -lt 1 -or $XaiCases -gt 64) {
    throw "XaiCases must be in [1, 64]."
}

$PythonPath = Resolve-RequiredPath -Value $Python -Label "Python"
$DataYamlPath = Resolve-RequiredPath -Value $DataYaml -Label "Data YAML"
$KeeperPath = Resolve-RequiredPath -Value $KeeperCheckpoint -Label "Keeper checkpoint"
$CandidatePath = Resolve-RequiredPath -Value $CandidateCheckpoint -Label "Candidate checkpoint"
$FrozenSummaryPath = Resolve-RequiredPath -Value $FrozenSummary -Label "Frozen validation summary"
$XaiCaseCsvPath = Resolve-RequiredPath -Value $XaiCaseCsv -Label "XAI priority CSV"

[void](Assert-FileSha256 -Path $KeeperPath -Expected $ExpectedKeeperSha256 -Label "Keeper checkpoint")
[void](Assert-FileSha256 -Path $CandidatePath -Expected $ExpectedCandidateSha256 -Label "Candidate checkpoint")
[void](Assert-FileSha256 -Path $FrozenSummaryPath -Expected $ExpectedFrozenSummarySha256 -Label "Frozen validation summary")

$RunDir = Join-Path $ProjectRoot ("runs\" + $RunName)
$PackageDir = Join-Path $RunDir "package"
$PackagePath = Join-Path $PackageDir "precision_ensemble.pt"
$PytorchAuditDir = Join-Path $RunDir "audit_pytorch_full_val"
$OnnxDir = Join-Path $RunDir "deploy_onnx_cpu"
$OnnxPath = Join-Path $OnnxDir "model_fp32.onnx"
$OnnxAuditDir = Join-Path $RunDir "audit_onnx_cpu_full_val"
$KeeperXaiDir = Join-Path $RunDir "xai_keeper_val"
$CandidateXaiDir = Join-Path $RunDir "xai_candidate_val"
$StatusPath = Join-Path $RunDir "pipeline_status.json"
$LatestPointerPath = Join-Path $ProjectRoot "runs\latest_precision_ensemble.json"

$env:PYTHONPATH = $ProjectRoot
$env:OMP_NUM_THREADS = "4"

Invoke-PythonStep -Name "module_preflight" -Arguments @(
    "-c",
    "import trkh.tools.build_precision_ensemble_checkpoint, trkh.tools.audit_precision_ensemble_checkpoint, trkh.tools.audit_precision_ensemble_onnx, trkh.evaluation.xai_audit, trkh.inference.deploy; print('precision ensemble modules ok')"
)

if ($PreflightOnly) {
    [ordered]@{
        status = "ok"
        mode = "preflight_only"
        run_dir = $RunDir
        keeper_checkpoint = $KeeperPath
        candidate_checkpoint = $CandidatePath
        frozen_validation_summary = $FrozenSummaryPath
        data_yaml = $DataYamlPath
        xai_cases = $XaiCases
        certified_backends = @("pytorch_fp32", "onnxruntime_cpu_fp32")
        rejected_backends = @("onnxruntime_cuda", "tensorrt_fp16", "tensorrt_fp32_no_tf32_o0")
        test_split_opened = $false
        native_stderr_policy = "direct invocation; no 2>&1 pipeline; LASTEXITCODE checked"
    } | ConvertTo-Json -Depth 6
    exit 0
}

if (Test-Path -LiteralPath $RunDir) {
    throw "Run directory already exists. Use a new RunName: $RunDir"
}
New-Item -ItemType Directory -Path $RunDir | Out-Null

try {
    Invoke-PythonStep -Name "build_locked_precision_package" -Arguments @(
        "-m", "trkh.tools.build_precision_ensemble_checkpoint",
        "--keeper", $KeeperPath,
        "--candidate", $CandidatePath,
        "--frozen-summary", $FrozenSummaryPath,
        "--expected-keeper-sha256", $ExpectedKeeperSha256,
        "--expected-candidate-sha256", $ExpectedCandidateSha256,
        "--expected-summary-sha256", $ExpectedFrozenSummarySha256,
        "--output-dir", $PackageDir
    )
    $PackageSha256 = (Get-FileHash -LiteralPath $PackagePath -Algorithm SHA256).Hash.ToLowerInvariant()

    Invoke-PythonStep -Name "audit_pytorch_full_validation" -Arguments @(
        "-m", "trkh.tools.audit_precision_ensemble_checkpoint",
        "--checkpoint", $PackagePath,
        "--keeper", $KeeperPath,
        "--candidate", $CandidatePath,
        "--frozen-summary", $FrozenSummaryPath,
        "--data", $DataYamlPath,
        "--output-dir", $PytorchAuditDir,
        "--expected-checkpoint-sha256", $PackageSha256,
        "--expected-keeper-sha256", $ExpectedKeeperSha256,
        "--expected-candidate-sha256", $ExpectedCandidateSha256,
        "--expected-summary-sha256", $ExpectedFrozenSummarySha256,
        "--batch-size", "1",
        "--device", "cuda"
    )
    [void](Assert-PassedSummary -Path (Join-Path $PytorchAuditDir "summary.json") -Label "PyTorch package audit")

    Invoke-PythonStep -Name "export_onnx_cpu_contract" -Arguments @(
        "-m", "trkh.inference.deploy",
        "--checkpoint", $PackagePath,
        "--data", $DataYamlPath,
        "--class-name-mode", "raw",
        "--expected-num-classes", "5",
        "--output-dir", $OnnxDir,
        "--batch-size", "1",
        "--num-workers", "0",
        "--benchmark-batch-size", "1",
        "--split", "val",
        "--skip-accuracy",
        "--skip-benchmark",
        "--skip-trt-engine"
    )
    $OnnxSha256 = (Get-FileHash -LiteralPath $OnnxPath -Algorithm SHA256).Hash.ToLowerInvariant()

    Invoke-PythonStep -Name "audit_onnx_cpu_full_validation" -Arguments @(
        "-m", "trkh.tools.audit_precision_ensemble_onnx",
        "--checkpoint", $PackagePath,
        "--onnx", $OnnxPath,
        "--frozen-summary", $FrozenSummaryPath,
        "--data", $DataYamlPath,
        "--output-dir", $OnnxAuditDir,
        "--expected-checkpoint-sha256", $PackageSha256,
        "--expected-onnx-sha256", $OnnxSha256,
        "--expected-summary-sha256", $ExpectedFrozenSummarySha256,
        "--reference-batch-size", "1",
        "--probability-tolerance", "1e-5",
        "--device", "cpu",
        "--onnx-provider", "cpu"
    )
    [void](Assert-PassedSummary -Path (Join-Path $OnnxAuditDir "summary.json") -Label "ONNX Runtime CPU audit")

    foreach ($member in @("keeper", "candidate")) {
        $xaiDir = if ($member -eq "keeper") { $KeeperXaiDir } else { $CandidateXaiDir }
        $sourceSha = if ($member -eq "keeper") { $ExpectedKeeperSha256 } else { $ExpectedCandidateSha256 }
        Invoke-PythonStep -Name ("xai_{0}_validation" -f $member) -Arguments @(
            "-m", "trkh.evaluation.xai_audit",
            "--checkpoint", $PackagePath,
            "--ensemble-member", $member,
            "--data", $DataYamlPath,
            "--split", "val",
            "--class-name-mode", "raw",
            "--expected-num-classes", "5",
            "--output-dir", $xaiDir,
            "--case-csv", $XaiCaseCsvPath,
            "--max-cases", [string]$XaiCases,
            "--batch-size", "1",
            "--num-workers", "0",
            "--method", "all",
            "--robustness-probes",
            "--disable-amp",
            "--focus-class-index", "1",
            "--bbox-token-prior-source", "bbox"
        )
        [void](Assert-XaiSummary -Path (Join-Path $xaiDir "xai_audit_summary.json") -Member $member -PackageSha256 $PackageSha256 -SourceSha256 $sourceSha)
    }

    $status = [ordered]@{
        status = "completed"
        mode = "precision_ensemble_certified_pipeline"
        completed_at = (Get-Date).ToString("o")
        checkpoint = $PackagePath
        checkpoint_sha256 = $PackageSha256
        onnx = $OnnxPath
        onnx_sha256 = $OnnxSha256
        frozen_validation_summary = $FrozenSummaryPath
        frozen_validation_summary_sha256 = $ExpectedFrozenSummarySha256
        certified_backends = [ordered]@{
            pytorch_fp32 = $true
            onnxruntime_cpu_fp32 = $true
        }
        rejected_backends = [ordered]@{
            onnxruntime_cuda = "targeted parity mismatch and probability gate failure"
            tensorrt_fp16 = "full-validation argmax and probability gate failure"
            tensorrt_fp32_no_tf32_o0 = "argmax exact but probability error exceeds 0.005"
        }
        video_contract = "PyTorch image-only video uses an unverified full-frame metadata fallback."
        test_split_opened = $false
        xai_cases_per_member = $XaiCases
        steps = $script:StepRecords
    }
    Write-JsonNoBom -Value $status -Path $StatusPath
    Write-JsonNoBom -Value ([ordered]@{
        checkpoint = $PackagePath
        checkpoint_sha256 = $PackageSha256
        onnx = $OnnxPath
        onnx_sha256 = $OnnxSha256
        pipeline_status = $StatusPath
        updated_at = (Get-Date).ToString("o")
    }) -Path $LatestPointerPath
    $status | ConvertTo-Json -Depth 10
}
catch {
    $failure = [ordered]@{
        status = "failed"
        failed_at = (Get-Date).ToString("o")
        error = $_.Exception.Message
        steps = $script:StepRecords
    }
    Write-JsonNoBom -Value $failure -Path $StatusPath
    throw
}
