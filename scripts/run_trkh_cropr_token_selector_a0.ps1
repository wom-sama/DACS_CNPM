param(
    [string]$Python = "D:\DataAI\.venv\Scripts\python.exe",
    [string]$FitOnlyDataYaml = "configs\trkh_cropr_a0_fitonly_20260716.yaml",
    [string]$HoldoutFoldDataYaml = "runs\yolof_cidt_fold0_trainonly_20260716\data.yaml",
    [string]$FoldSummary = "runs\yolof_cidt_fold0_trainonly_20260716\summary.json",
    [string]$Declaration = "runs\audit_cidt_readiness_full_train_20260714\predictions_all_conditions.csv",
    [string]$PreflightOutput = "runs\audit_cropr_token_selector_preflight_20260716",
    [string]$ControlRunName = "probe_cropr_a0_native_control_5e_20260716",
    [string]$CandidateRunName = "probe_cropr_a0_learned_candidate_5e_20260716",
    [switch]$PreflightOnly,
    [switch]$RunPair
)

$ErrorActionPreference = "Stop"
$RepoRoot = Split-Path -Parent $PSScriptRoot
Set-Location $RepoRoot

$SelectedModes = @($PreflightOnly, $RunPair) | Where-Object { $_ }
if ($SelectedModes.Count -ne 1) {
    throw "Chon dung mot mode: -PreflightOnly hoac -RunPair."
}
if (-not (Test-Path -LiteralPath $Python)) {
    throw "Khong tim thay Python: $Python"
}

$Protocol = "docs\TRKH_5CLASS_CROPR_TOKEN_SELECTOR_READINESS_PROTOCOL_20260716.md"
$RequiredFiles = @(
    $FitOnlyDataYaml,
    $HoldoutFoldDataYaml,
    $FoldSummary,
    $Declaration,
    $Protocol
)
foreach ($Path in $RequiredFiles) {
    if (-not (Test-Path -LiteralPath $Path)) {
        throw "Khong tim thay Cropr A0 input da khoa: $Path"
    }
}

function Assert-FileSha256 {
    param([string]$Path, [string]$Expected)
    $Observed = (Get-FileHash -Algorithm SHA256 -LiteralPath $Path).Hash.ToLowerInvariant()
    if ($Observed -ne $Expected.ToLowerInvariant()) {
        throw "SHA-256 mismatch: $Path observed=$Observed expected=$Expected"
    }
}

Assert-FileSha256 -Path $FitOnlyDataYaml -Expected "5306a58516ecbc85cfb775c734413bdd24561e3ddbfbe0a7245c90461a00ff79"
Assert-FileSha256 -Path $HoldoutFoldDataYaml -Expected "4195ba89995d4eee483cae31626717381df910f56657086e7116322ca971388e"
Assert-FileSha256 -Path $FoldSummary -Expected "2f938b11573073ed957c2522e1a170e6043522f305a787620d1af5f6dbd66763"
Assert-FileSha256 -Path $Declaration -Expected "2e0993752d58d99ea429bfefe1e2bfe6fa949e45aea1a26cc4bdfee97d4db21c"
Assert-FileSha256 -Path $Protocol -Expected "42e912a6bdc320d98a9acbb79b6d48c1e372a5df861f279af1c6988b3ff6a847"

if ($PreflightOnly) {
    if (Test-Path -LiteralPath $PreflightOutput) {
        throw "Preflight output da ton tai; protocol cam overwrite: $PreflightOutput"
    }
    & $Python -m trkh.tools.audit_cropr_token_selector_preflight `
        --fit-only-data $FitOnlyDataYaml `
        --fold-data $HoldoutFoldDataYaml `
        --fold-summary $FoldSummary `
        --declaration $Declaration `
        --protocol $Protocol `
        --output-dir $PreflightOutput `
        --batch-size 32 `
        --fp32-batch-size 2 `
        --num-workers 4 `
        --seed 42 `
        --warmup-iterations 2 `
        --timed-iterations 5
    if ($LASTEXITCODE -ne 0) {
        throw "Cropr A0 preflight rejected or failed with exit code $LASTEXITCODE."
    }
    Write-Host "Cropr A0 engineering preflight passed."
}

if ($RunPair) {
    $PreflightSummary = Join-Path $PreflightOutput "summary.json"
    if (-not (Test-Path -LiteralPath $PreflightSummary)) {
        throw "Khong tim thay preflight summary: $PreflightSummary"
    }
    $Preflight = Get-Content -Raw -LiteralPath $PreflightSummary | ConvertFrom-Json
    if (-not [bool]$Preflight.gate.formal_pair_permission) {
        throw "Preflight khong cap formal_pair_permission."
    }
    if ([bool]$Preflight.validation_used -or [bool]$Preflight.test_used) {
        throw "Preflight da truy cap validation/test trai protocol."
    }
    $PreflightFitOnlyHash = [string]$Preflight.sources.fit_only_data.sha256
    if ($PreflightFitOnlyHash -ne "5306a58516ecbc85cfb775c734413bdd24561e3ddbfbe0a7245c90461a00ff79") {
        throw "Preflight khong khoa dung fit-only data YAML."
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
    if ([string]$Preflight.git.head -ne $Head -or [string]$Preflight.git.upstream -ne $Head) {
        throw "Preflight phai duoc tao tu chinh HEAD da push dang chay."
    }

    $ControlRunDir = Join-Path "runs" $ControlRunName
    $CandidateRunDir = Join-Path "runs" $CandidateRunName
    $PairManifestPath = Join-Path "runs" "cropr_a0_pair_manifest_20260716.json"
    if (
        (Test-Path -LiteralPath $ControlRunDir) -or
        (Test-Path -LiteralPath $CandidateRunDir) -or
        (Test-Path -LiteralPath $PairManifestPath)
    ) {
        throw "Formal Cropr artifact da ton tai; protocol cam overwrite/rerun."
    }

    $Launcher = Join-Path $PSScriptRoot "run_trkh_5class_attention_views_v8.ps1"
    $Shared = @{
        Python = $Python
        DataYaml = $FitOnlyDataYaml
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
        InattentiveTokenFusion = $false
        CroprTokenSelector = $true
        TokenPruneForegroundWeight = 0.35
        EarlyTokenMaskKeepRate = 1.0
        BBoxSpatialFusion = $true
        GatedRelativePositionAttention = $false
        VisualContrastAttention = $false
        FovealAggregatedAttention = $false
        DeformableSpatialAttention = $false
        BiLevelRoutingAttention = $false
        CrossCovarianceAttention = $false
        DynamicGraphMixer = $false
        SoftMoePatchAdapter = $false
        DeepClassPrompt = $false
        LayerTokenFusion = $false
        PatchStyleRecalibration = $false
        BlockLocalPatchMixer = $false
        LocallyEnhancedFfn = $false
        ConcurrentLocalGlobalCoupling = $false
        LateMemberBranch = $false
        DataCartography = $true
        TraceArchitecture = $true
        SkipFinalTest = $true
    }

    & $Launcher @Shared -RunName $ControlRunName -CroprTokenSelectorRouting $false
    if ($LASTEXITCODE -ne 0) {
        throw "Cropr native-routing control failed with exit code $LASTEXITCODE"
    }
    & $Launcher @Shared -RunName $CandidateRunName -CroprTokenSelectorRouting $true
    if ($LASTEXITCODE -ne 0) {
        throw "Cropr learned-routing candidate failed with exit code $LASTEXITCODE"
    }

    $ControlLast = Join-Path $ControlRunDir "checkpoints\last.pt"
    $CandidateLast = Join-Path $CandidateRunDir "checkpoints\last.pt"
    $ControlOccurrences = Join-Path $ControlRunDir "data_cartography_train_occurrence_hashes.json"
    $CandidateOccurrences = Join-Path $CandidateRunDir "data_cartography_train_occurrence_hashes.json"
    foreach ($Path in @($ControlLast, $CandidateLast, $ControlOccurrences, $CandidateOccurrences)) {
        if (-not (Test-Path -LiteralPath $Path)) {
            throw "Formal pair thieu artifact bat buoc: $Path"
        }
    }
    $ControlOccurrencePayload = Get-Content -Raw -LiteralPath $ControlOccurrences | ConvertFrom-Json
    $CandidateOccurrencePayload = Get-Content -Raw -LiteralPath $CandidateOccurrences | ConvertFrom-Json
    if ($ControlOccurrencePayload.epochs.Count -ne 5 -or $CandidateOccurrencePayload.epochs.Count -ne 5) {
        throw "Formal pair phai hoan tat dung 5 occurrence epochs."
    }
    $ControlEpochs = $ControlOccurrencePayload.epochs | ConvertTo-Json -Depth 6 -Compress
    $CandidateEpochs = $CandidateOccurrencePayload.epochs | ConvertTo-Json -Depth 6 -Compress
    if ($ControlEpochs -ne $CandidateEpochs) {
        throw "Candidate khong replay bit-exact ordered sample occurrence hashes cua control."
    }

    $PairManifest = [ordered]@{
        method = "cropr_token_selector_a0_pair"
        protocol = (Resolve-Path -LiteralPath $Protocol).Path
        git_head = $Head
        fit_only_data = (Resolve-Path -LiteralPath $FitOnlyDataYaml).Path
        holdout_fold_data = (Resolve-Path -LiteralPath $HoldoutFoldDataYaml).Path
        preflight_summary = (Resolve-Path -LiteralPath $PreflightSummary).Path
        control_run = (Resolve-Path -LiteralPath $ControlRunDir).Path
        candidate_run = (Resolve-Path -LiteralPath $CandidateRunDir).Path
        control_last_checkpoint_sha256 = (Get-FileHash -Algorithm SHA256 -LiteralPath $ControlLast).Hash.ToLowerInvariant()
        candidate_last_checkpoint_sha256 = (Get-FileHash -Algorithm SHA256 -LiteralPath $CandidateLast).Hash.ToLowerInvariant()
        occurrence_epochs = $ControlOccurrencePayload.epochs
        sole_causal_difference = "cropr_token_selector_routing_false_vs_true"
        holdout_loaded_during_training = $false
        official_validation_used = $false
        test_used = $false
        current_best_command_updated = $false
    }
    $PairManifest | ConvertTo-Json -Depth 8 | Set-Content -LiteralPath $PairManifestPath -Encoding utf8
    Write-Host "Cropr A0 pair completed: $PairManifestPath"
}
