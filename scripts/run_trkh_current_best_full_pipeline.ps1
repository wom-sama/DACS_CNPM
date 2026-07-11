param(
    [string]$Python = "D:\DataAI\.venv\Scripts\python.exe",
    [string]$DataYaml = "D:\DataAI\AIEx\newdataset\yolo_f\data.yaml",
    [string]$RunName = ("full_v8_yolof_current_best_30e_{0}" -f (Get-Date -Format "yyyyMMdd_HHmmss")),
    [string]$ResumeCheckpoint = "runs\probe_v8_yolof_pairroute_teacherfocusbinary015_boundarydrop_bboxprior_120b_2e_20260701\checkpoints\best.pt",
    [string]$TeacherCsv = "runs\weighted_ensemble_no_pretrain_6expert_val_gate_20260701\teacher_probs_train_sampleindex_focus005.csv",
    [string]$VerifierJson = "runs\patch_evidence_mil_yolof_keeper_01only_exportparams_20260705\pair_verifier_model_params.json",
    [int]$Epochs = 30,
    [int]$Patience = 3,
    [double]$MinimumRawValMacroF1 = 0.8829248547554016,
    [double]$MinimumRawValClass1F1 = 0.678260862827301,
    [double]$ValidationGateTolerance = 1e-6,
    [switch]$RunFinalTest,
    [switch]$PreflightOnly
)

$ErrorActionPreference = "Stop"
$ProjectRoot = Split-Path -Parent $PSScriptRoot
Set-Location $ProjectRoot

function Resolve-ProjectPath {
    param(
        [string]$Value,
        [string]$Label
    )
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
    if (-not (Test-Path -LiteralPath $fullPath)) {
        throw "$Label not found: $fullPath"
    }
    return $fullPath
}

function Write-JsonNoBom {
    param(
        [object]$Value,
        [string]$Path,
        [int]$Depth = 8
    )
    $parent = Split-Path -Parent $Path
    if (-not [string]::IsNullOrWhiteSpace($parent)) {
        New-Item -ItemType Directory -Path $parent -Force | Out-Null
    }
    $json = $Value | ConvertTo-Json -Depth $Depth
    $utf8 = New-Object System.Text.UTF8Encoding($false)
    [System.IO.File]::WriteAllText($Path, $json, $utf8)
}

$PythonPath = Resolve-ProjectPath -Value $Python -Label "Python"
$DataYamlPath = Resolve-ProjectPath -Value $DataYaml -Label "Data YAML"
$ResumeCheckpointPath = Resolve-ProjectPath -Value $ResumeCheckpoint -Label "Resume checkpoint"
$TeacherCsvPath = Resolve-ProjectPath -Value $TeacherCsv -Label "Teacher CSV"
$VerifierJsonPath = Resolve-ProjectPath -Value $VerifierJson -Label "Verifier JSON"
$TrainLauncher = Resolve-ProjectPath `
    -Value "scripts\run_trkh_5class_attention_views_v8.ps1" `
    -Label "TRKH train launcher"

if ($RunName -notmatch '^[A-Za-z0-9_.-]+$') {
    throw "RunName may contain only letters, digits, dot, underscore, and dash: $RunName"
}
if ($Epochs -lt 1 -or $Epochs -gt 30) {
    throw "Epochs must be in [1, 30]."
}
if ($Patience -lt 1) {
    throw "Patience must be >= 1."
}

$RunDir = Join-Path $ProjectRoot ("runs\" + $RunName)
$AuditRoot = Join-Path $RunDir "final_audit"
$Checkpoint = Join-Path $RunDir "checkpoints\best.pt"
$PipelineStatusPath = Join-Path $RunDir "pipeline_status.json"
$LatestPointerPath = Join-Path $ProjectRoot "runs\latest_full_pipeline.json"

if (-not $PreflightOnly -and (Test-Path -LiteralPath $RunDir)) {
    throw "Run directory already exists. Use a new RunName: $RunDir"
}

$env:PYTHONPATH = $ProjectRoot
$env:TRKH_AMP_DTYPE = "bf16"
$env:OMP_NUM_THREADS = "4"
$env:TRKH_ALLOW_WINDOWS_MULTIPROCESSING = "1"

$script:StepRecords = New-Object System.Collections.ArrayList

function Add-StepRecord {
    param(
        [string]$Name,
        [string]$Status,
        [int]$ExitCode,
        [double]$Seconds,
        [string]$Note = ""
    )
    [void]$script:StepRecords.Add([ordered]@{
        name = $Name
        status = $Status
        exit_code = $ExitCode
        seconds = [math]::Round($Seconds, 3)
        note = $Note
    })
}

function Invoke-PythonStep {
    param(
        [string]$Name,
        [string[]]$Arguments
    )
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
    if ($exitCode -ne 0) {
        Add-StepRecord -Name $Name -Status "failed" -ExitCode $exitCode -Seconds $seconds
        throw "$Name failed with exit code $exitCode."
    }
    Add-StepRecord -Name $Name -Status "completed" -ExitCode 0 -Seconds $seconds
}

$TrainParameters = @{
    Python = $PythonPath
    DataYaml = $DataYamlPath
    RunName = $RunName
    ResumeCheckpoint = $ResumeCheckpointPath
    Epochs = $Epochs
    SchedulerTotalEpochs = $Epochs
    Patience = $Patience
    BatchSize = 32
    GradAccumSteps = 2
    NumWorkers = 4
    EvalNumWorkers = 2
    MaxTrainBatches = 0
    MaxValBatches = 0
    Seed = 42
    LearningRate = 8e-5
    MinLearningRate = 1e-6
    WarmupEpochs = 1
    WeightDecay = 0.05
    BackboneLrScale = 1.0
    ImageSize = 256
    CropMarginRatio = 0.05
    BboxTokenPriorSource = "bbox"
    HardSampleManifest = ""
    HardSampleRepeatFactor = 1.0
    DistillationTeacherCsv = $TeacherCsvPath
    DistillationWeight = 0.0
    DistillationFocusClassWeight = 1.0
    TeacherFocusBinaryLossWeight = 0.015
    TeacherFocusBinaryFocusClass = 1
    TeacherFocusBinaryClasses = "0,1,2,4"
    TeacherFocusBinaryTeacherMinConfidence = 0.0
    TeacherFocusBinaryErrorPower = 0.0
    BBoxSpatialFusion = $true
    BBoxSpatialFusionHiddenDim = 64
    BBoxSpatialFusionDropout = 0.05
    BBoxSpatialFusionLogitScale = 0.20
    PairwiseMarginRouting = $true
    PairwiseMarginPairs = "0-1,1-2,2-3,4-rest"
    PairwiseMarginRouteMaxProbabilityMargin = 0.20
    MetricLearningLossWeight = 0.04
    MetricLearningTemperature = 0.16
    MetricLearningSources = "head,patch"
    BBoxForegroundDropoutLossWeight = 0.08
    BBoxForegroundDropoutConsistencyWeight = 0.03
    BBoxForegroundDropoutProbability = 0.45
    BBoxForegroundDropoutMinAreaRatio = 0.04
    BBoxForegroundDropoutMaxAreaRatio = 0.14
    BBoxForegroundDropoutMode = "boundary_band"
    BBoxForegroundDropoutFill = "mean"
    BBoxForegroundDropoutTemperature = 1.2
    AttentionViewLossWeight = 0.35
    AttentionCropProbability = 0.40
    AttentionDropProbability = 0.20
    AttentionViewStartEpoch = 2
    AttentionViewScoreSource = "learned_attention"
    AttentionViewForegroundWeight = 0.40
    AttentionDropMinAreaRatio = 0.06
    AttentionDropMaxAreaRatio = 0.16
    SkipFinalTest = $true
    TraceArchitecture = $true
}

$VerifierArguments = @(
    "--patch-evidence-linear-verifier-json", $VerifierJsonPath,
    "--patch-evidence-linear-verifier-pair", "0-1",
    "--patch-evidence-linear-verifier-min-pair-probability", "0.02",
    "--patch-evidence-linear-verifier-max-pair-margin", "0.40",
    "--patch-evidence-linear-verifier-confidence-threshold", "0.60",
    "--patch-evidence-linear-verifier-logit-boost", "0.01",
    "--patch-evidence-linear-verifier-protect-right-min-probability", "0.0"
)

if ($PreflightOnly) {
    $TrainParameters["PreflightOnly"] = $true
    $previousErrorActionPreference = $ErrorActionPreference
    try {
        $ErrorActionPreference = "Continue"
        & $TrainLauncher @TrainParameters
        $preflightExitCode = $LASTEXITCODE
    }
    finally {
        $ErrorActionPreference = $previousErrorActionPreference
    }
    if ($preflightExitCode -ne 0) {
        throw "Train launcher preflight failed with exit code $preflightExitCode."
    }
    Invoke-PythonStep -Name "pipeline_module_imports" -Arguments @(
        "-c",
        "import trkh.evaluation.evaluate, trkh.evaluation.xai_audit, trkh.evaluation.robustness_eval, trkh.tools.audit_prediction_forensics, trkh.tools.audit_class_confusions, trkh.tools.build_boundary_review_manifest, trkh.tools.summarize_xai_transitions, trkh.tools.audit_trkh_artifact_retention"
    )
    [ordered]@{
        status = "ok"
        mode = "preflight_only"
        run_name = $RunName
        python = $PythonPath
        data_yaml = $DataYamlPath
        resume_checkpoint = $ResumeCheckpointPath
        verifier_json = $VerifierJsonPath
        epochs = $Epochs
        minimum_raw_val_macro_f1 = $MinimumRawValMacroF1
        minimum_raw_val_class_1_f1 = $MinimumRawValClass1F1
        validation_gate_tolerance = $ValidationGateTolerance
        final_test_requested = [bool]$RunFinalTest
        native_stderr_policy = "direct invocation; no 2>&1 pipeline; LASTEXITCODE checked"
    } | ConvertTo-Json -Depth 5
    exit 0
}

function Invoke-Evaluation {
    param(
        [string]$Split,
        [string]$OutputDir,
        [string]$PaperName,
        [bool]$WithVerifier
    )
    $arguments = @(
        "-m", "trkh.evaluation.evaluate",
        "--checkpoint", $Checkpoint,
        "--data", $DataYamlPath,
        "--class-name-mode", "raw",
        "--expected-num-classes", "5",
        "--split", $Split,
        "--batch-size", "64",
        "--num-workers", "2",
        "--amp",
        "--bbox-token-prior-source", "bbox",
        "--paper-name", $PaperName,
        "--family", "TRKH",
        "--output-dir", $OutputDir
    )
    if ($WithVerifier) {
        $arguments += $VerifierArguments
    }
    Invoke-PythonStep -Name ("evaluate_{0}_{1}" -f $Split, $(if ($WithVerifier) { "softboost001" } else { "raw" })) -Arguments $arguments
}

function Invoke-Forensics {
    param(
        [string]$Name,
        [string]$Predictions,
        [string]$OutputDir
    )
    Invoke-PythonStep -Name $Name -Arguments @(
        "-m", "trkh.tools.audit_prediction_forensics",
        "--predictions", $Predictions,
        "--output-dir", $OutputDir,
        "--pairs", "0-1,1-2,2-3,4-rest",
        "--focus-class-index", "1",
        "--image-stats-mode", "basic",
        "--max-image-stats", "500",
        "--top-k-images", "24"
    )
}

function Invoke-ConfusionAudit {
    param(
        [string]$Name,
        [string]$Predictions,
        [string]$OutputDir
    )
    Invoke-PythonStep -Name $Name -Arguments @(
        "-m", "trkh.tools.audit_class_confusions",
        "--predictions", $Predictions,
        "--output-dir", $OutputDir,
        "--data", $DataYamlPath,
        "--focus-class-index", "1",
        "--top-k-images", "24"
    )
}

function Invoke-BoundaryAudit {
    param(
        [string]$Name,
        [string]$Split,
        [string]$Predictions,
        [string]$OutputDir
    )
    $arguments = @(
        "-m", "trkh.tools.build_boundary_review_manifest",
        "--predictions", $Predictions,
        "--output-dir", $OutputDir,
        "--split", $Split,
        "--focus-class-index", "1",
        "--high-confidence-threshold", "0.55",
        "--low-margin-threshold", "0.06",
        "--image-stats-mode", "basic",
        "--max-total", "120",
        "--max-per-reason", "30",
        "--max-per-pair", "30"
    )
    if ($Split -eq "test") {
        $arguments += "--allow-test"
    }
    Invoke-PythonStep -Name $Name -Arguments $arguments
}

function Invoke-XaiAudit {
    param(
        [string]$Name,
        [string]$Split,
        [string]$OutputDir,
        [bool]$WithVerifier
    )
    $arguments = @(
        "-m", "trkh.evaluation.xai_audit",
        "--checkpoint", $Checkpoint,
        "--data", $DataYamlPath,
        "--class-name-mode", "raw",
        "--expected-num-classes", "5",
        "--split", $Split,
        "--output-dir", $OutputDir,
        "--max-cases", "24",
        "--mistake-cases", "12",
        "--low-confidence-cases", "6",
        "--close-margin-cases", "6",
        "--per-class-cases", "2",
        "--batch-size", "32",
        "--num-workers", "0",
        "--method", "all",
        "--query-tokens", "cls_register_mean",
        "--bbox-token-prior-source", "bbox",
        "--robustness-probes",
        "--review-high-confidence", "0.45",
        "--top-k", "5"
    )
    if ($WithVerifier) {
        $arguments += $VerifierArguments
    }
    Invoke-PythonStep -Name $Name -Arguments $arguments
    Invoke-PythonStep -Name ($Name + "_transitions") -Arguments @(
        "-m", "trkh.tools.summarize_xai_transitions",
        "--xai-summary-json", (Join-Path $OutputDir "xai_audit_summary.json"),
        "--output-dir", (Join-Path $OutputDir "transition_summary")
    )
}

$PipelineState = [ordered]@{
    status = "running"
    run_name = $RunName
    started_at = (Get-Date).ToString("o")
    final_test_requested = [bool]$RunFinalTest
    native_stderr_policy = "direct invocation; no 2>&1 pipeline; LASTEXITCODE checked"
    steps = $script:StepRecords
}

try {
    Write-Host "`n=== train_current_best ===" -ForegroundColor Cyan
    $trainStarted = Get-Date
    $previousErrorActionPreference = $ErrorActionPreference
    $trainExitCode = 1
    try {
        $ErrorActionPreference = "Continue"
        & $TrainLauncher @TrainParameters
        $trainExitCode = $LASTEXITCODE
    }
    finally {
        $ErrorActionPreference = $previousErrorActionPreference
    }
    $trainSeconds = ((Get-Date) - $trainStarted).TotalSeconds
    if ($trainExitCode -ne 0) {
        Add-StepRecord -Name "train_current_best" -Status "failed" -ExitCode $trainExitCode -Seconds $trainSeconds
        throw "Training failed with exit code $trainExitCode."
    }
    Add-StepRecord -Name "train_current_best" -Status "completed" -ExitCode 0 -Seconds $trainSeconds

    if (-not (Test-Path -LiteralPath $Checkpoint)) {
        throw "Training completed without best checkpoint: $Checkpoint"
    }
    $TrainSummaryPath = Join-Path $RunDir "summary.json"
    if (-not (Test-Path -LiteralPath $TrainSummaryPath)) {
        throw "Training summary missing: $TrainSummaryPath"
    }
    $TrainSummary = Get-Content -LiteralPath $TrainSummaryPath -Raw | ConvertFrom-Json
    if ($null -ne $TrainSummary.test_summary) {
        throw "Leakage guard failed: trainer test_summary must be null before explicit final test."
    }
    if ($null -eq $TrainSummary.architecture_trace -or $TrainSummary.architecture_trace.status -ne "completed") {
        throw "Architecture trace did not complete."
    }
    New-Item -ItemType Directory -Path $AuditRoot -Force | Out-Null

    $ValRaw = Join-Path $AuditRoot "val_raw"
    $ValSoft = Join-Path $AuditRoot "val_softboost001"
    Invoke-Evaluation -Split "val" -OutputDir $ValRaw -PaperName "TRKH-current-best-full-val-raw" -WithVerifier $false
    Invoke-Evaluation -Split "val" -OutputDir $ValSoft -PaperName "TRKH-current-best-full-val-softboost001" -WithVerifier $true

    Invoke-Forensics -Name "forensics_val_raw" -Predictions (Join-Path $ValRaw "predictions_detailed.csv") -OutputDir (Join-Path $AuditRoot "forensics_val_raw")
    Invoke-Forensics -Name "forensics_val_softboost001" -Predictions (Join-Path $ValSoft "predictions_detailed.csv") -OutputDir (Join-Path $AuditRoot "forensics_val_softboost001")
    Invoke-ConfusionAudit -Name "confusions_val_raw" -Predictions (Join-Path $ValRaw "predictions_detailed.csv") -OutputDir (Join-Path $AuditRoot "confusions_val_raw")
    Invoke-ConfusionAudit -Name "confusions_val_softboost001" -Predictions (Join-Path $ValSoft "predictions_detailed.csv") -OutputDir (Join-Path $AuditRoot "confusions_val_softboost001")
    Invoke-BoundaryAudit -Name "boundary_val_raw" -Split "val" -Predictions (Join-Path $ValRaw "predictions_detailed.csv") -OutputDir (Join-Path $AuditRoot "boundary_val_raw")
    Invoke-BoundaryAudit -Name "boundary_val_softboost001" -Split "val" -Predictions (Join-Path $ValSoft "predictions_detailed.csv") -OutputDir (Join-Path $AuditRoot "boundary_val_softboost001")

    Invoke-PythonStep -Name "robustness_val_raw" -Arguments @(
        "-m", "trkh.evaluation.robustness_eval",
        "--checkpoint", $Checkpoint,
        "--data", $DataYamlPath,
        "--class-name-mode", "raw",
        "--expected-num-classes", "5",
        "--batch-size", "32",
        "--num-workers", "0",
        "--output-dir", (Join-Path $AuditRoot "robustness_val_raw"),
        "--num-fail-cases", "12"
    )

    Invoke-XaiAudit -Name "xai_val_raw" -Split "val" -OutputDir (Join-Path $AuditRoot "xai_val_raw") -WithVerifier $false
    Invoke-XaiAudit -Name "xai_val_softboost001" -Split "val" -OutputDir (Join-Path $AuditRoot "xai_val_softboost001") -WithVerifier $true

    $ValRawMetricsPath = Join-Path $ValRaw "metrics_detailed.json"
    $ValSoftMetricsPath = Join-Path $ValSoft "metrics_detailed.json"
    $ValRawMetrics = Get-Content -LiteralPath $ValRawMetricsPath -Raw | ConvertFrom-Json
    $ValSoftMetrics = Get-Content -LiteralPath $ValSoftMetricsPath -Raw | ConvertFrom-Json
    $ValRawClass1 = $ValRawMetrics.per_class | Where-Object { [int]$_.class_index -eq 1 } | Select-Object -First 1
    $ValSoftClass1 = $ValSoftMetrics.per_class | Where-Object { [int]$_.class_index -eq 1 } | Select-Object -First 1
    if ($null -eq $ValRawClass1 -or $null -eq $ValSoftClass1) {
        throw "Validation metrics do not contain class index 1."
    }
    $ValidationGatePassed = (
        [double]$ValRawMetrics.macro_f1 + $ValidationGateTolerance -ge $MinimumRawValMacroF1 -and
        [double]$ValRawClass1.f1 + $ValidationGateTolerance -ge $MinimumRawValClass1F1
    )
    $ValidationGate = [ordered]@{
        passed = [bool]$ValidationGatePassed
        policy = "raw checkpoint must match or exceed both selected raw keeper thresholds before test/deploy promotion"
        thresholds = [ordered]@{
            macro_f1 = $MinimumRawValMacroF1
            class_1_f1 = $MinimumRawValClass1F1
            numeric_tolerance = $ValidationGateTolerance
        }
        raw = [ordered]@{
            macro_f1 = [double]$ValRawMetrics.macro_f1
            class_1_precision = [double]$ValRawClass1.precision
            class_1_recall = [double]$ValRawClass1.recall
            class_1_f1 = [double]$ValRawClass1.f1
        }
        softboost001_diagnostic = [ordered]@{
            macro_f1 = [double]$ValSoftMetrics.macro_f1
            class_1_precision = [double]$ValSoftClass1.precision
            class_1_recall = [double]$ValSoftClass1.recall
            class_1_f1 = [double]$ValSoftClass1.f1
        }
        test_requested = [bool]$RunFinalTest
        test_permitted = [bool]($RunFinalTest -and $ValidationGatePassed)
    }
    Write-JsonNoBom -Value $ValidationGate -Path (Join-Path $AuditRoot "validation_promotion_gate.json")

    if ($RunFinalTest -and $ValidationGatePassed) {
        $TestRaw = Join-Path $AuditRoot "test_final_raw"
        $TestSoft = Join-Path $AuditRoot "test_final_softboost001"
        Invoke-Evaluation -Split "test" -OutputDir $TestRaw -PaperName "TRKH-current-best-full-test-final-raw" -WithVerifier $false
        Invoke-Evaluation -Split "test" -OutputDir $TestSoft -PaperName "TRKH-current-best-full-test-final-softboost001" -WithVerifier $true
        Invoke-Forensics -Name "forensics_test_final_raw" -Predictions (Join-Path $TestRaw "predictions_detailed.csv") -OutputDir (Join-Path $AuditRoot "forensics_test_final_raw")
        Invoke-Forensics -Name "forensics_test_final_softboost001" -Predictions (Join-Path $TestSoft "predictions_detailed.csv") -OutputDir (Join-Path $AuditRoot "forensics_test_final_softboost001")
        Invoke-ConfusionAudit -Name "confusions_test_final_raw" -Predictions (Join-Path $TestRaw "predictions_detailed.csv") -OutputDir (Join-Path $AuditRoot "confusions_test_final_raw")
        Invoke-ConfusionAudit -Name "confusions_test_final_softboost001" -Predictions (Join-Path $TestSoft "predictions_detailed.csv") -OutputDir (Join-Path $AuditRoot "confusions_test_final_softboost001")
        Invoke-BoundaryAudit -Name "boundary_test_final_softboost001" -Split "test" -Predictions (Join-Path $TestSoft "predictions_detailed.csv") -OutputDir (Join-Path $AuditRoot "boundary_test_final_softboost001")
        Invoke-XaiAudit -Name "xai_test_final_softboost001" -Split "test" -OutputDir (Join-Path $AuditRoot "xai_test_final_softboost001") -WithVerifier $true
    }

    Invoke-PythonStep -Name "artifact_retention_audit" -Arguments @(
        "-m", "trkh.tools.audit_trkh_artifact_retention",
        "--runs-root", (Join-Path $ProjectRoot "runs"),
        "--output-dir", (Join-Path $AuditRoot "artifact_retention")
    )

    $PipelineSummary = [ordered]@{
        status = if ($ValidationGatePassed) { "completed" } else { "validation_gate_rejected" }
        run_name = $RunName
        checkpoint = $Checkpoint
        train_summary = $TrainSummaryPath
        audit_root = $AuditRoot
        validation_raw = $ValRaw
        validation_softboost001 = $ValSoft
        validation_gate = $ValidationGate
        final_test_included = [bool]($RunFinalTest -and $ValidationGatePassed)
        final_test_skip_reason = if ($RunFinalTest -and -not $ValidationGatePassed) { "raw validation promotion gate failed" } else { $null }
        test_final_raw = if ($RunFinalTest -and $ValidationGatePassed) { Join-Path $AuditRoot "test_final_raw" } else { $null }
        test_final_softboost001 = if ($RunFinalTest -and $ValidationGatePassed) { Join-Path $AuditRoot "test_final_softboost001" } else { $null }
        expected_engine = Join-Path $RunDir "deploy\model_fp32_fp16.engine"
        completed_at = (Get-Date).ToString("o")
    }
    Write-JsonNoBom -Value $PipelineSummary -Path (Join-Path $AuditRoot "pipeline_summary.json")
    if ($ValidationGatePassed) {
        Write-JsonNoBom -Value $PipelineSummary -Path $LatestPointerPath
    }

    $PipelineState.status = $PipelineSummary.status
    $PipelineState.finished_at = (Get-Date).ToString("o")
    $PipelineState.checkpoint = $Checkpoint
    $PipelineState.audit_root = $AuditRoot
    Write-JsonNoBom -Value $PipelineState -Path $PipelineStatusPath
    $PipelineSummary | ConvertTo-Json -Depth 6
}
catch {
    $PipelineState.status = "failed"
    $PipelineState.finished_at = (Get-Date).ToString("o")
    $PipelineState.error = $_.Exception.Message
    Write-JsonNoBom -Value $PipelineState -Path $PipelineStatusPath
    throw
}
