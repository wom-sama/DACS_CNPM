param(
    [string]$Python = "D:\DataAI\.venv\Scripts\python.exe",
    [string]$RawDataYaml = "D:\DataAI\AIEx\newdataset\yolo_f\data.yaml",
    [string]$Declaration = "runs\audit_cidt_readiness_full_train_20260714\predictions_all_conditions.csv",
    [string]$FoldRoot = "runs\yolof_cidt_fold0_trainonly_20260716",
    [string]$PreflightOutput = "runs\audit_inattentive_fusion_preflight_20260716",
    [string]$ControlRunName = "probe_inattentive_fusion_a0_control_5e_20260716",
    [string]$CandidateRunName = "probe_inattentive_fusion_a0_candidate_5e_20260716",
    [switch]$PreflightOnly,
    [switch]$ReplayEngineeringCorrection,
    [switch]$FinalizeVisualReview,
    [ValidateSet("pass", "fail")]
    [string]$VisualReviewResult = "pass",
    [string]$VisualReviewNote = "",
    [string]$ExpectedSummarySha256 = "",
    [switch]$RunPair
)

$ErrorActionPreference = "Stop"
$RepoRoot = Split-Path -Parent $PSScriptRoot
Set-Location $RepoRoot
$env:PYTHONPATH = $RepoRoot
$env:TRKH_ALLOW_WINDOWS_MULTIPROCESSING = "1"
$env:OMP_NUM_THREADS = "4"

$SelectedModes = @($PreflightOnly, $ReplayEngineeringCorrection, $FinalizeVisualReview, $RunPair) | Where-Object { $_ }
if ($SelectedModes.Count -ne 1) {
    throw "Chon dung mot mode: -PreflightOnly, -ReplayEngineeringCorrection, -FinalizeVisualReview, hoac -RunPair."
}
if (-not (Test-Path -LiteralPath $Python)) {
    throw "Khong tim thay Python: $Python"
}

$FoldDataYaml = Join-Path $FoldRoot "data.yaml"
$FoldSummary = Join-Path $FoldRoot "summary.json"
$Protocol = "docs\TRKH_5CLASS_INATTENTIVE_TOKEN_FUSION_READINESS_PROTOCOL_20260716.md"
$RequiredFiles = @($RawDataYaml, $Declaration, $FoldDataYaml, $FoldSummary, $Protocol)
foreach ($Path in $RequiredFiles) {
    if (-not (Test-Path -LiteralPath $Path)) {
        throw "Khong tim thay input da khoa: $Path"
    }
}

function Assert-FileSha256 {
    param([string]$Path, [string]$Expected)
    $Observed = (Get-FileHash -Algorithm SHA256 -LiteralPath $Path).Hash.ToLowerInvariant()
    if ($Observed -ne $Expected.ToLowerInvariant()) {
        throw "SHA-256 mismatch: $Path observed=$Observed expected=$Expected"
    }
}

Assert-FileSha256 -Path $RawDataYaml -Expected "716e33df24c63a9e9920f97b685199707fb84ab4c7154544f5dd9a3e00d884ef"
Assert-FileSha256 -Path $Declaration -Expected "2e0993752d58d99ea429bfefe1e2bfe6fa949e45aea1a26cc4bdfee97d4db21c"
Assert-FileSha256 -Path $FoldDataYaml -Expected "4195ba89995d4eee483cae31626717381df910f56657086e7116322ca971388e"
Assert-FileSha256 -Path $FoldSummary -Expected "2f938b11573073ed957c2522e1a170e6043522f305a787620d1af5f6dbd66763"
Assert-FileSha256 -Path $Protocol -Expected "d3f8de6fbef7ad5d88ca33e5db71f9c5ca587f32c1ad5ee4797fd400aaf3ee4b"

if ($PreflightOnly) {
    if (Test-Path -LiteralPath $PreflightOutput) {
        throw "Preflight output da ton tai; protocol cam overwrite: $PreflightOutput"
    }
    & $Python -m trkh.tools.audit_inattentive_token_fusion_preflight `
        --fold-data $FoldDataYaml `
        --fold-summary $FoldSummary `
        --raw-data $RawDataYaml `
        --declaration $Declaration `
        --protocol $Protocol `
        --output-dir $PreflightOutput `
        --batch-size 32 `
        --fp32-batch-size 2 `
        --audit-batch-size 32 `
        --num-workers 4 `
        --seed 42 `
        --warmup-iterations 2 `
        --timed-iterations 5
    if ($LASTEXITCODE -ne 0) {
        throw "Inattentive-token fusion preflight rejected or failed with exit code $LASTEXITCODE."
    }
    Write-Host "Automated preflight passed. Review every fusion_overlay_page_*.png before finalization."
}

if ($ReplayEngineeringCorrection) {
    if ($ExpectedSummarySha256 -notmatch '^[0-9a-fA-F]{64}$') {
        throw "ReplayEngineeringCorrection yeu cau -ExpectedSummarySha256 gom 64 ky tu hex."
    }
    & $Python -m trkh.tools.audit_inattentive_token_fusion_preflight `
        --output-dir $PreflightOutput `
        --replay-engineering-correction `
        --expected-summary-sha256 $ExpectedSummarySha256 `
        --batch-size 32 `
        --fp32-batch-size 2 `
        --num-workers 4 `
        --seed 42
    if ($LASTEXITCODE -ne 0) {
        throw "Inattentive-token fusion engineering correction rejected with exit code $LASTEXITCODE."
    }
}

if ($FinalizeVisualReview) {
    if ([string]::IsNullOrWhiteSpace($VisualReviewNote)) {
        throw "FinalizeVisualReview yeu cau -VisualReviewNote."
    }
    if ($ExpectedSummarySha256 -notmatch '^[0-9a-fA-F]{64}$') {
        throw "FinalizeVisualReview yeu cau -ExpectedSummarySha256 gom 64 ky tu hex."
    }
    & $Python -m trkh.tools.audit_inattentive_token_fusion_preflight `
        --output-dir $PreflightOutput `
        --finalize-visual-review `
        --visual-review-result $VisualReviewResult `
        --visual-review-note $VisualReviewNote `
        --expected-summary-sha256 $ExpectedSummarySha256
    if ($LASTEXITCODE -ne 0) {
        throw "Inattentive-token fusion visual finalization rejected with exit code $LASTEXITCODE."
    }
}

if ($RunPair) {
    $PreflightSummary = Join-Path $PreflightOutput "summary.json"
    if (-not (Test-Path -LiteralPath $PreflightSummary)) {
        throw "Khong tim thay preflight summary: $PreflightSummary"
    }
    $Preflight = Get-Content -Raw -LiteralPath $PreflightSummary | ConvertFrom-Json
    if (-not [bool]$Preflight.gate.formal_pair_permission) {
        throw "Preflight va visual review khong cap formal_pair_permission."
    }
    if ([bool]$Preflight.validation_used -or [bool]$Preflight.test_used) {
        throw "Preflight khong duoc phep su dung official validation/test."
    }
    $TrackedStatus = git status --short --untracked-files=no
    if ($LASTEXITCODE -ne 0 -or -not [string]::IsNullOrWhiteSpace(($TrackedStatus -join "`n"))) {
        throw "Tracked worktree phai clean truoc formal pair."
    }
    $Head = (git rev-parse HEAD).Trim()
    $Upstream = (git rev-parse '@{upstream}').Trim()
    if ($LASTEXITCODE -ne 0 -or $Head -ne $Upstream) {
        throw "HEAD phai duoc push truoc formal pair."
    }
    $EvidenceHead = [string]$Preflight.git.head
    if ($null -ne $Preflight.postflight_replay) {
        if (-not [bool]$Preflight.postflight_replay.passed -or -not [bool]$Preflight.postflight_replay.runtime_files_unchanged) {
            throw "Postflight replay khong hop le hoac runtime files da thay doi."
        }
        $EvidenceHead = [string]$Preflight.postflight_replay.git.head
    }
    if ($EvidenceHead -ne $Head) {
        throw "Preflight/correction evidence phai duoc tao tu chinh HEAD da push dang chay."
    }

    $ControlRunDir = Join-Path "runs" $ControlRunName
    $CandidateRunDir = Join-Path "runs" $CandidateRunName
    $PairManifestPath = Join-Path "runs" "inattentive_fusion_a0_pair_manifest_20260716.json"
    if (
        (Test-Path -LiteralPath $ControlRunDir) -or
        (Test-Path -LiteralPath $CandidateRunDir) -or
        (Test-Path -LiteralPath $PairManifestPath)
    ) {
        throw "Formal pair artifact da ton tai; protocol cam overwrite/rerun."
    }

    $Launcher = Join-Path $PSScriptRoot "run_trkh_5class_attention_views_v8.ps1"
    $Shared = @{
        Python = $Python
        DataYaml = $FoldDataYaml
        ResumeCheckpoint = ""
        ImageSize = 256
        StemArchitecture = "conv_pool"
        Epochs = 5
        SchedulerTotalEpochs = 5
        Patience = 3
        BatchSize = 32
        GradAccumSteps = 2
        NumWorkers = 4
        EvalNumWorkers = 2
        MaxTrainBatches = 0
        MaxValBatches = 0
        Seed = 42
        DisableBalancedEpochSampling = $true
        LearningRate = 2.5e-4
        MinLearningRate = 1e-6
        WarmupEpochs = 1
        WeightDecay = 0.05
        ClassificationLoss = "ldam_focal"
        FocalLossGamma = 1.0
        FocalLossMix = 0.10
        LabelSmoothing = 0.02
        LdamMaxMargin = 0.30
        LdamScale = 18.0
        MetricLearningLossWeight = 0.04
        MetricLearningTemperature = 0.16
        AttentionViewLossWeight = 0.0
        AttentionCropProbability = 0.0
        AttentionDropProbability = 0.0
        TeacherFocusBinaryLossWeight = 0.0
        DistillationTeacherCsv = ""
        SampleWeightManifest = ""
        HardSampleManifest = ""
        AmbiguousSoftTargetManifest = ""
        TargetedMarginManifest = ""
        FocusNeighborBinaryManifest = ""
        ClassificationSourceContext = $false
        PairedViewTrain = $false
        AuxiliaryTrainData = ""
        MixStyle = $false
        TokenPruning = $true
        EarlyTokenMaskKeepRate = 1.0
        BBoxSpatialFusion = $true
        GatedRelativePositionAttention = $false
        VisualContrastAttention = $false
        FovealAggregatedAttention = $false
        DeformableSpatialAttention = $false
        CrossCovarianceAttention = $false
        DynamicGraphMixer = $false
        SoftMoePatchAdapter = $false
        DeepClassPrompt = $false
        LayerTokenFusion = $false
        PatchStyleRecalibration = $false
        BlockLocalPatchMixer = $false
        LocallyEnhancedFfn = $false
        ConcurrentLocalGlobalCoupling = $false
        BiLevelRoutingAttention = $false
        DataCartography = $true
        TraceArchitecture = $true
        SkipFinalTest = $true
    }

    & $Launcher @Shared -RunName $ControlRunName -InattentiveTokenFusion $false
    if ($LASTEXITCODE -ne 0) {
        throw "Fusion hard-prune control failed with exit code $LASTEXITCODE"
    }
    & $Launcher @Shared -RunName $CandidateRunName -InattentiveTokenFusion $true
    if ($LASTEXITCODE -ne 0) {
        throw "Fusion candidate failed with exit code $LASTEXITCODE"
    }

    $PairManifest = [ordered]@{
        method = "inattentive_token_fusion_a0_pair"
        protocol = $Protocol
        git_head = $Head
        fold_data = (Resolve-Path -LiteralPath $FoldDataYaml).Path
        preflight_summary = (Resolve-Path -LiteralPath $PreflightSummary).Path
        control_run = (Resolve-Path -LiteralPath $ControlRunDir).Path
        candidate_run = (Resolve-Path -LiteralPath $CandidateRunDir).Path
        sole_causal_difference = "inattentive_token_fusion_false_vs_true"
        official_validation_used = $false
        test_used = $false
        current_best_command_updated = $false
    }
    $PairManifest | ConvertTo-Json -Depth 5 | Set-Content -LiteralPath $PairManifestPath -Encoding utf8
    Write-Host "Inattentive-token fusion A0 pair completed: $PairManifestPath"
}
