[CmdletBinding()]
param(
    [ValidateSet("Preflight", "Smoke", "Probe", "Full")]
    [string]$Mode = "Preflight",

    [ValidateSet("Direct", "Hybrid", "Both")]
    [string]$Arm = "Both",

    [string]$Python = "D:\DataAI\.venv\Scripts\python.exe",
    [string]$DataYaml = "D:\DataAI\AIEx\newdataset\yolo_f\data.yaml",
    [string]$KeeperCheckpoint = "D:\DataAI\AIEx\TRKH\runs\probe_v8_yolof_pairroute_teacherfocusbinary015_boundarydrop_bboxprior_120b_2e_20260701\checkpoints\best.pt",
    [string]$DinoCheckpoint = "C:\Users\ADMIN\.cache\huggingface\hub\models--timm--vit_small_patch16_dinov3.lvd1689m\snapshots\3bf4720a82ec2066db88137180ff1f83a675cef0\model.safetensors",
    [string]$OutputDir = "",
    [string]$RunTag = "20260729",

    [ValidateSet(8, 12, 16, 24)]
    [int]$BatchSize = 24,
    [int]$NumWorkers = 4,
    [int]$EvalNumWorkers = 2,
    [int]$Seed = 42,

    [string]$GateManifest = "",
    [switch]$ConfirmFull,
    [switch]$AllowHistoricalYoloProtocol
)

Set-StrictMode -Version Latest
$ErrorActionPreference = "Stop"

if (-not $AllowHistoricalYoloProtocol) {
    throw @"
This A0 launcher is retained only to reproduce the historical yolo_f protocol.
The canonical dataset is now class_f and its labels are not compatible with the
old keeper or teacher cache. Use:

  python -m trkh.recipes.pretrained_classf_b0 --help

Pass -AllowHistoricalYoloProtocol only for an explicitly labelled historical
reproduction. Its checkpoints and metrics must not initialize or select a
canonical class_f model.
"@
}

$ProtocolId = "TRKH_PRETRAINED_DINOV3_A0_20260729"
$DinoModelName = "vit_small_patch16_dinov3.lvd1689m"
$DinoSha256 = "2a1ec16ae28ffa07bc0ead0241ee7df9fc26451fe6f9f839b7b3afa0a906b040"
$DinoSourceUrl = "https://huggingface.co/timm/vit_small_patch16_dinov3.lvd1689m"
$DinoSourceRevision = "3bf4720a82ec2066db88137180ff1f83a675cef0"
$DinoSourceLicense = "dinov3-license"
$KeeperSha256 = "1f49d577240c69dc63c30af70db52ec2aa9da65a17aef1c4b1c09ece6c482677"
$DataYamlSha256 = "716e33df24c63a9e9920f97b685199707fb84ab4c7154544f5dd9a3e00d884ef"
$ExpectedClassNames = @(
    "Xoai_Song_Chua_KhoDap",
    "Xoai_Song_ChuaNhe_CoNguyCo",
    "Xoai_Chin_NgotThanh_DeDap",
    "Xoai_ChinGia_NgotGat_KhongVanChuyen",
    "Xoai_Hu_KhongAnDuoc"
)

$RepoRoot = [System.IO.Path]::GetFullPath((Join-Path $PSScriptRoot ".."))
if ([string]::IsNullOrWhiteSpace($OutputDir)) {
    $OutputRoot = Join-Path $RepoRoot "runs"
}
elseif ([System.IO.Path]::IsPathRooted($OutputDir)) {
    $OutputRoot = [System.IO.Path]::GetFullPath($OutputDir)
}
else {
    $OutputRoot = [System.IO.Path]::GetFullPath((Join-Path $RepoRoot $OutputDir))
}

if ($RunTag -notmatch "^[A-Za-z0-9_-]+$") {
    throw "RunTag may contain only letters, numbers, underscore and hyphen."
}
if ($NumWorkers -lt 0 -or $EvalNumWorkers -lt 0) {
    throw "Worker counts must be non-negative."
}

$AccumulationByBatch = @{
    8 = 6
    12 = 4
    16 = 3
    24 = 2
}
$GradAccumSteps = [int]$AccumulationByBatch[$BatchSize]
$EffectiveBatchSize = $BatchSize * $GradAccumSteps
if ($EffectiveBatchSize -ne 48) {
    throw "A0 requires effective batch 48; got $BatchSize x $GradAccumSteps."
}

function Write-JsonFile {
    param(
        [Parameter(Mandatory = $true)]$Value,
        [Parameter(Mandatory = $true)][string]$Path
    )
    $parent = Split-Path -Parent $Path
    if (-not [string]::IsNullOrWhiteSpace($parent)) {
        New-Item -ItemType Directory -Path $parent -Force | Out-Null
    }
    $Value | ConvertTo-Json -Depth 12 | Set-Content -LiteralPath $Path -Encoding UTF8
}

function Assert-FileSha256 {
    param(
        [Parameter(Mandatory = $true)][string]$Path,
        [Parameter(Mandatory = $true)][string]$Expected,
        [Parameter(Mandatory = $true)][string]$Label
    )
    if (-not (Test-Path -LiteralPath $Path -PathType Leaf)) {
        throw "$Label not found: $Path"
    }
    $observed = (Get-FileHash -LiteralPath $Path -Algorithm SHA256).Hash.ToLowerInvariant()
    if ($observed -ne $Expected.ToLowerInvariant()) {
        throw "$Label SHA-256 mismatch: observed=$observed expected=$Expected path=$Path"
    }
    return $observed
}

function Get-A0RunName {
    param(
        [Parameter(Mandatory = $true)][string]$SelectedArm,
        [Parameter(Mandatory = $true)][string]$Stage
    )
    return "pretrained_dinov3_a0_$($SelectedArm.ToLowerInvariant())_$($Stage.ToLowerInvariant())_$RunTag"
}

function Get-SelectedArms {
    if ($Arm -eq "Both") {
        return @("Direct", "Hybrid")
    }
    return @($Arm)
}

function Assert-StageMarker {
    param(
        [Parameter(Mandatory = $true)][string]$SelectedArm,
        [Parameter(Mandatory = $true)][string]$RequiredStage
    )
    $requiredRun = Get-A0RunName -SelectedArm $SelectedArm -Stage $RequiredStage
    $markerPath = Join-Path (Join-Path $OutputRoot $requiredRun) "a0_stage_complete.json"
    if (-not (Test-Path -LiteralPath $markerPath -PathType Leaf)) {
        throw "Missing successful $RequiredStage marker for ${SelectedArm}: $markerPath"
    }
    $marker = Get-Content -LiteralPath $markerPath -Raw | ConvertFrom-Json
    if (
        [string]$marker.protocol -ne $ProtocolId -or
        [string]$marker.arm -ne $SelectedArm.ToLowerInvariant() -or
        [string]$marker.stage -ne $RequiredStage.ToLowerInvariant() -or
        [int]$marker.exit_code -ne 0 -or
        [string]$marker.git_commit -ne $script:GitCommit -or
        [bool]$marker.test_locked -ne $true -or
        [int]$marker.batch_size -ne $BatchSize -or
        [int]$marker.grad_accum_steps -ne $GradAccumSteps -or
        [int]$marker.effective_batch_size -ne $EffectiveBatchSize
    ) {
        throw "Invalid $RequiredStage marker: $markerPath"
    }
}

function Assert-FullGateManifest {
    param([Parameter(Mandatory = $true)][string]$Path)

    if (-not (Test-Path -LiteralPath $Path -PathType Leaf)) {
        throw "Full gate manifest not found: $Path"
    }
    $gate = Get-Content -LiteralPath $Path -Raw | ConvertFrom-Json
    $expectedDirect = Get-A0RunName -SelectedArm "Direct" -Stage "Probe"
    $expectedHybrid = Get-A0RunName -SelectedArm "Hybrid" -Stage "Probe"

    if ([string]$gate.protocol -ne $ProtocolId) {
        throw "Gate protocol mismatch."
    }
    if ([string]$gate.decision -ne "promotable_to_full") {
        throw "Gate decision must be promotable_to_full."
    }
    if ([string]$gate.git_commit -ne $script:GitCommit) {
        throw "Gate Git commit does not match the current code commit."
    }
    if ([string]$gate.direct_probe_run -ne $expectedDirect) {
        throw "Gate direct_probe_run mismatch: expected $expectedDirect"
    }
    if ([string]$gate.hybrid_probe_run -ne $expectedHybrid) {
        throw "Gate hybrid_probe_run mismatch: expected $expectedHybrid"
    }
    if ([int]$gate.full_validation_support -ne 2606) {
        throw "Gate must use all 2606 validation samples."
    }
    if ([string]$gate.bootstrap_unit -ne "source_image_group") {
        throw "Gate bootstrap_unit must be source_image_group."
    }
    if ([int]$gate.bootstrap_resamples -lt 10000) {
        throw "Gate requires at least 10000 paired source-group bootstrap resamples."
    }
    if ([bool]$gate.test_locked -ne $true) {
        throw "Gate must attest that test remained locked."
    }
    if (
        [int]$gate.batch_size -ne $BatchSize -or
        [int]$gate.grad_accum_steps -ne $GradAccumSteps -or
        [int]$gate.effective_batch_size -ne $EffectiveBatchSize
    ) {
        throw "Gate/probe batch geometry does not match the requested full run."
    }

    $macroDelta = [double]$gate.hybrid_minus_direct.macro_f1
    $class1Delta = [double]$gate.hybrid_minus_direct.class1_f1
    $precisionDelta = [double]$gate.hybrid_minus_direct.class1_precision
    $recallDelta = [double]$gate.hybrid_minus_direct.class1_recall
    $macroCiLow = [double]$gate.paired_source_group_bootstrap_95ci.macro_f1[0]
    $class1CiLow = [double]$gate.paired_source_group_bootstrap_95ci.class1_f1[0]

    if ($macroDelta -lt 0.0) {
        throw "Full gate failed: hybrid macro-F1 point delta is negative ($macroDelta)."
    }
    if ($macroCiLow -lt -0.003) {
        throw "Full gate failed: macro-F1 CI lower bound $macroCiLow < -0.003."
    }
    if ($class1Delta -lt 0.010) {
        throw "Full gate failed: class-1-F1 delta $class1Delta < 0.010."
    }
    if ($class1CiLow -le 0.0) {
        throw "Full gate failed: class-1-F1 CI lower bound must be > 0."
    }
    if ($precisionDelta -lt -0.020 -or $recallDelta -lt -0.020) {
        throw "Full gate failed: class-1 precision/recall protection was violated."
    }
    return (Resolve-Path -LiteralPath $Path).Path
}

switch ($Mode) {
    "Preflight" {
        $StageName = "Preflight"
        $Epochs = 1
        $SchedulerTotalEpochs = 5
        $WarmupEpochs = 0
        $Patience = 2
        $MaxTrainBatches = 4
        $MaxValBatches = 2
    }
    "Smoke" {
        $StageName = "Smoke"
        $Epochs = 1
        $SchedulerTotalEpochs = 5
        $WarmupEpochs = 0
        $Patience = 2
        $MaxTrainBatches = 4
        $MaxValBatches = 2
    }
    "Probe" {
        $StageName = "Probe"
        $Epochs = 5
        $SchedulerTotalEpochs = 5
        $WarmupEpochs = 1
        $Patience = 5
        $MaxTrainBatches = 120
        $MaxValBatches = 0
    }
    "Full" {
        $StageName = "Full"
        $Epochs = 30
        $SchedulerTotalEpochs = 30
        $WarmupEpochs = 2
        $Patience = 6
        $MaxTrainBatches = 0
        $MaxValBatches = 0
    }
}

$LearningRate = 1.5e-4
$PretrainedBackboneLrScale = 0.1
$PretrainedBackboneLearningRate = $LearningRate * $PretrainedBackboneLrScale
$MinLearningRate = 1e-6
$resolvedPython = (Resolve-Path -LiteralPath $Python).Path
$resolvedDataYaml = (Resolve-Path -LiteralPath $DataYaml).Path
$resolvedKeeper = (Resolve-Path -LiteralPath $KeeperCheckpoint).Path
$resolvedDino = (Resolve-Path -LiteralPath $DinoCheckpoint).Path

if ([System.IO.Path]::GetExtension($resolvedDino).ToLowerInvariant() -ne ".safetensors") {
    throw "DINO checkpoint must be addressed through the revision-pinned model.safetensors path."
}

$null = Assert-FileSha256 -Path $resolvedDataYaml -Expected $DataYamlSha256 -Label "data.yaml"
$null = Assert-FileSha256 -Path $resolvedKeeper -Expected $KeeperSha256 -Label "keeper checkpoint"
$null = Assert-FileSha256 -Path $resolvedDino -Expected $DinoSha256 -Label "DINOv3 checkpoint"

$env:PYTHONPATH = $RepoRoot
$env:TRKH_AMP_DTYPE = "bf16"
$env:OMP_NUM_THREADS = "4"
$env:TRKH_ALLOW_WINDOWS_MULTIPROCESSING = "1"
Remove-Item Env:TRKH_ALLOW_WINDOWS_PIN_MEMORY -ErrorAction SilentlyContinue
Remove-Item Env:TRKH_ALLOW_WINDOWS_PERSISTENT_WORKERS -ErrorAction SilentlyContinue

$script:GitBranch = ((& git -C $RepoRoot branch --show-current) | Out-String).Trim()
if ($LASTEXITCODE -ne 0) {
    throw "Unable to read Git branch."
}
if (-not $script:GitBranch.StartsWith("research/pretrained-")) {
    throw "A0 may run only on a research/pretrained-* branch; current=$script:GitBranch"
}
$script:GitCommit = ((& git -C $RepoRoot rev-parse HEAD) | Out-String).Trim().ToLowerInvariant()
if ($LASTEXITCODE -ne 0 -or $script:GitCommit -notmatch "^[0-9a-f]{40}$") {
    throw "Unable to resolve a valid Git commit."
}
$GitStatus = @(& git -C $RepoRoot status --short)
if ($LASTEXITCODE -ne 0) {
    throw "Unable to inspect Git status."
}
if (($Mode -eq "Probe" -or $Mode -eq "Full") -and $GitStatus.Count -gt 0) {
    throw "$Mode requires a clean pretrained worktree. Commit the coherent implementation first."
}

$DirectRunName = Get-A0RunName -SelectedArm "Direct" -Stage $StageName
$HybridRunName = Get-A0RunName -SelectedArm "Hybrid" -Stage $StageName

$CommonArgs = @(
    "--data", $resolvedDataYaml,
    "--class-name-mode", "raw",
    "--expected-num-classes", "5",
    "--output-dir", $OutputRoot,
    "--research-track", "pretrained",
    "--pretrained",
    "--timm-model-name", $DinoModelName,
    "--pretrained-checkpoint-path", $resolvedDino,
    "--pretrained-checkpoint-sha256", $DinoSha256,
    "--pretrained-source-url", $DinoSourceUrl,
    "--pretrained-source-revision", $DinoSourceRevision,
    "--pretrained-source-license", $DinoSourceLicense,
    "--image-size", "256",
    "--batch-size", "$BatchSize",
    "--grad-accum-steps", "$GradAccumSteps",
    "--epochs", "$Epochs",
    "--scheduler-total-epochs", "$SchedulerTotalEpochs",
    "--patience", "$Patience",
    "--learning-rate", "$LearningRate",
    "--backbone-lr-scale", "$PretrainedBackboneLrScale",
    "--min-learning-rate", "$MinLearningRate",
    "--warmup-epochs", "$WarmupEpochs",
    "--warmup-start-factor", "0.1",
    "--weight-decay", "0.05",
    "--grad-clip-norm", "0.7",
    "--max-nonfinite-grad-steps", "4",
    "--model-ema",
    "--model-ema-decay", "0.995",
    "--num-workers", "$NumWorkers",
    "--eval-num-workers", "$EvalNumWorkers",
    "--train-image-cache-mb", "0",
    "--eval-image-cache-mb", "0",
    "--seed", "$Seed",
    "--disable-class-weights",
    "--balanced-epoch-multiplier", "1.0",
    "--balanced-epoch-tolerance", "0.10",
    "--no-pretrained-distillation",
    "--distillation-weight", "0",
    "--teacher-focus-binary-loss-weight", "0",
    "--teacher-focus-margin-loss-weight", "0",
    "--teacher-pairwise-margin-loss-weight", "0",
    "--max-train-batches", "$MaxTrainBatches",
    "--max-val-batches", "$MaxValBatches",
    "--skip-final-test"
)

$DirectRecipeArgs = @(
    "--classification-loss", "ldam_focal",
    "--focal-loss-gamma", "1.0",
    "--focal-loss-mix", "0.1",
    "--label-smoothing", "0.02",
    "--ldam-max-margin", "0.3",
    "--ldam-scale", "18",
    "--best-metric", "fair_macro_f1",
    "--fair-f1-gap-target", "0.05",
    "--fair-f1-gap-penalty", "1.5",
    "--fair-f1-min-weight", "0.25",
    "--crop-margin-ratio", "0.05",
    "--class-crop-margin-scale-threshold", "1.5",
    "--class-crop-margin-max-ratio", "0.16",
    "--resize-mode", "pad",
    "--train-scale-min", "0.88",
    "--train-scale-crop-probability", "0.35",
    "--brightness", "0.04",
    "--contrast", "0.04",
    "--saturation", "0.02",
    "--hue", "0.01",
    "--illumination-normalization",
    "--illumination-normalization-strength", "0.35",
    "--background-suppression-mode", "desaturate_blur",
    "--background-suppression-probability", "0.8",
    "--background-suppression-margin", "0.08",
    "--background-suppression-blur-radius", "7",
    "--local-exposure-probability", "0.15",
    "--local-exposure-strength", "0.25",
    "--obstacle-probability", "0.04",
    "--obstacle-max-area", "0.08",
    "--random-erasing-probability", "0",
    "--random-affine-degrees", "3",
    "--random-affine-translate", "0.02",
    "--random-affine-scale-min", "0.96",
    "--horizontal-flip-probability", "0.5",
    "--vertical-flip-probability", "0",
    "--rotate90-probability", "0.03",
    "--lighting-probability", "0",
    "--disable-class-aware-augmentation",
    "--disable-rare-class-repeat",
    "--batch-mix-probability", "0",
    "--mosaic-probability", "0",
    "--mixup-probability", "0",
    "--cutmix-probability", "0",
    "--copy-paste-probability", "0",
    "--targeted-copy-paste-probability", "0"
)

$DirectArgs = @(
    "--run-name", $DirectRunName,
    "--model-type", "timm_classifier"
) + $CommonArgs + $DirectRecipeArgs

$HybridArgs = @(
    "--run-name", $HybridRunName,
    "--model-type", "vit_registers_pretrained_hybrid"
) + $CommonArgs + @(
    "--resume", $resolvedKeeper,
    "--resume-weight-source", "selected",
    "--resume-reset-epoch",
    "--resume-reset-optimizer",
    "--resume-reset-scheduler",
    "--resume-reset-scaler",
    "--pretrained-semantic-expected-prefix-tokens", "5",
    "--pretrained-semantic-expected-embed-dim", "384",
    "--pretrained-semantic-expected-patch-count", "256",
    "--pretrained-semantic-dropout", "0.05",
    "--pretrained-semantic-initial-scale", "0.0",
    "--pretrained-semantic-max-scale", "0.25",
    "--pretrained-backbone-gradient-checkpointing"
)

if ($HybridArgs -contains "--resume-use-cli-config") {
    throw "Protocol violation: hybrid must inherit keeper config without --resume-use-cli-config."
}
if ($HybridArgs -contains "--distillation-teacher-csv") {
    throw "Protocol violation: A0 is label-only and must not receive a teacher CSV."
}
foreach ($candidateArgs in @($DirectArgs, $HybridArgs)) {
    if (@($candidateArgs | Where-Object { $_ -eq "--skip-final-test" }).Count -ne 1) {
        throw "Every A0 arm must contain exactly one --skip-final-test."
    }
}

$RequiredCliFlags = @(
    "--research-track",
    "--pretrained-checkpoint-path",
    "--pretrained-checkpoint-sha256",
    "--pretrained-source-url",
    "--pretrained-source-revision",
    "--pretrained-source-license",
    "--pretrained-semantic-expected-prefix-tokens",
    "--pretrained-backbone-gradient-checkpointing",
    "--resume-use-cli-config",
    "--resume-weight-source",
    "--skip-final-test"
)
$HelpText = ((& $resolvedPython -m trkh.training.train --help 2>&1) | Out-String)
if ($LASTEXITCODE -ne 0) {
    throw "Training parser help failed."
}
foreach ($requiredFlag in $RequiredCliFlags) {
    if (-not $HelpText.Contains($requiredFlag)) {
        throw "Training parser does not expose required flag: $requiredFlag"
    }
}

$CompileTargets = @(
    (Join-Path $RepoRoot "trkh\core\config.py"),
    (Join-Path $RepoRoot "trkh\models\pretrained_semantic_branch.py"),
    (Join-Path $RepoRoot "trkh\models\model.py"),
    (Join-Path $RepoRoot "trkh\training\train.py"),
    (Join-Path $RepoRoot "trkh\evaluation\evaluate.py")
)
& $resolvedPython -m compileall -q @CompileTargets
if ($LASTEXITCODE -ne 0) {
    throw "Python compile preflight failed."
}

$FocusedTests = @(
    (Join-Path $RepoRoot "tests\test_pretrained_semantic_branch.py"),
    (Join-Path $RepoRoot "tests\test_pretrained_hybrid_model.py"),
    (Join-Path $RepoRoot "tests\test_timm_classifier_model.py")
)
foreach ($testPath in $FocusedTests) {
    if (-not (Test-Path -LiteralPath $testPath -PathType Leaf)) {
        throw "Focused test missing: $testPath"
    }
}
& $resolvedPython -m pytest -q @FocusedTests
if ($LASTEXITCODE -ne 0) {
    throw "Focused pretrained tests failed."
}

$ValidationImageDir = Join-Path (Split-Path -Parent $resolvedDataYaml) "images\val"
if (-not (Test-Path -LiteralPath $ValidationImageDir -PathType Container)) {
    throw "Validation image directory not found: $ValidationImageDir"
}
$ValidationCount = @(
    Get-ChildItem -LiteralPath $ValidationImageDir -File -Recurse |
        Where-Object { $_.Extension.ToLowerInvariant() -in @(".jpg", ".jpeg", ".png", ".bmp", ".webp") }
).Count
if ($ValidationCount -ne 2577) {
    throw "A0 requires 2577 yolo_f validation source images; observed=$ValidationCount"
}

$PreflightRoot = Join-Path $OutputRoot "pretrained_dinov3_a0_preflight_$RunTag"
New-Item -ItemType Directory -Path $PreflightRoot -Force | Out-Null
$Timestamp = (Get-Date).ToUniversalTime().ToString("yyyyMMdd_HHmmss")
$RequestPath = Join-Path $PreflightRoot "$($Mode.ToLowerInvariant())_request_$Timestamp.json"
$Request = [ordered]@{
    protocol = $ProtocolId
    mode = $Mode.ToLowerInvariant()
    arm = $Arm.ToLowerInvariant()
    repo_root = $RepoRoot
    data_yaml = $resolvedDataYaml
    keeper_checkpoint = $resolvedKeeper
    dino_checkpoint = $resolvedDino
    dino_sha256 = $DinoSha256
    dino_model_name = $DinoModelName
    dino_source_url = $DinoSourceUrl
    dino_source_revision = $DinoSourceRevision
    dino_source_license = $DinoSourceLicense
    expected_class_names = $ExpectedClassNames
    direct_train_args = $DirectArgs
    hybrid_train_args = $HybridArgs
    learning_rate = $LearningRate
    pretrained_backbone_lr_scale = $PretrainedBackboneLrScale
    pretrained_backbone_learning_rate = $PretrainedBackboneLearningRate
}
Write-JsonFile -Value $Request -Path $RequestPath

$PythonPreflight = @'
import gc
import json
import sys
from pathlib import Path

import timm
import torch
import yaml

from trkh.core.config import load_data_spec
from trkh.data.dataset import MangoYOLOCropDataset
from trkh.models.model import (
    build_model_from_checkpoint,
    create_model,
    extract_bbox_from_model_output,
)
from trkh.core.utils import build_optimizer_param_groups
from trkh.training.train import (
    _load_model_state_allowing_extensions,
    build_configs,
    parse_args,
)

request = json.loads(Path(sys.argv[1]).read_text(encoding="utf-8-sig"))
expected_classes = list(request["expected_class_names"])

with Path(request["data_yaml"]).open("r", encoding="utf-8") as handle:
    data_document = yaml.safe_load(handle)
raw_names = data_document.get("names", [])
if isinstance(raw_names, dict):
    class_names = [raw_names[key] for key in sorted(raw_names, key=lambda value: int(value))]
else:
    class_names = list(raw_names)
if class_names != expected_classes:
    raise RuntimeError(f"data.yaml class order drifted: {class_names!r}")

data_spec = load_data_spec(
    request["data_yaml"],
    class_name_mode="raw",
    expected_num_classes=5,
)
validation_dataset = MangoYOLOCropDataset.from_data_spec(
    data_spec=data_spec,
    split="val",
    transform=None,
    crop_to_primary_object=True,
    classification_target=True,
    classification_object_crops=True,
)
validation_report = validation_dataset.quality_report()
if len(validation_dataset) != 2606:
    raise RuntimeError(
        f"A0 requires 2606 validation object crops; observed={len(validation_dataset)}"
    )
if int(validation_report.get("image_file_count", 0)) != 2577:
    raise RuntimeError(f"Validation source-image drift: {validation_report!r}")
for field in (
    "missing_image_count",
    "empty_label_count",
    "invalid_line_count",
    "invalid_bbox_count",
    "invalid_class_count",
):
    if int(validation_report.get(field, 0)) != 0:
        raise RuntimeError(f"Validation dataset integrity failed: {field}={validation_report[field]}")
del validation_dataset

direct_args = parse_args(request["direct_train_args"])
hybrid_args = parse_args(request["hybrid_train_args"])
if direct_args.model_type != "timm_classifier":
    raise RuntimeError("Direct parser contract changed.")
if hybrid_args.model_type != "vit_registers_pretrained_hybrid":
    raise RuntimeError("Hybrid parser contract changed.")
if hybrid_args.resume_use_cli_config:
    raise RuntimeError("Hybrid parser unexpectedly enabled resume_use_cli_config.")
if hybrid_args.resume_weight_source != "selected":
    raise RuntimeError("Hybrid must inherit the selected/EMA keeper weights.")
if not direct_args.skip_final_test or not hybrid_args.skip_final_test:
    raise RuntimeError("Test lock was not preserved by the parser.")
if direct_args.distillation_teacher_csv is not None or hybrid_args.distillation_teacher_csv is not None:
    raise RuntimeError("A0 must remain label-only.")

direct_config, _, _ = build_configs(direct_args)
direct_model = create_model(num_classes=5, model_config=direct_config)
direct_provenance = getattr(direct_model, "pretrained_provenance", {})
direct_digest = (
    direct_provenance.get("initialization", {})
    .get("checkpoint", {})
    .get("sha256", "")
)
if direct_digest != request["dino_sha256"]:
    raise RuntimeError(f"Direct model did not consume the locked DINO file: {direct_digest}")
direct_parameters = sum(parameter.numel() for parameter in direct_model.parameters())
direct_groups = build_optimizer_param_groups(
    direct_model,
    weight_decay=0.05,
    learning_rate=float(request["learning_rate"]),
    backbone_lr_scale=float(request["pretrained_backbone_lr_scale"]),
)
direct_group_names = {group["name"] for group in direct_groups}
direct_backbone_lrs = {
    float(group["lr"])
    for group in direct_groups
    if group["name"].startswith("backbone_") and "lr" in group
}
if not any(name.startswith("head_") for name in direct_group_names):
    raise RuntimeError(f"Direct timm classifier has no head optimizer group: {direct_group_names}")
if direct_backbone_lrs != {float(request["pretrained_backbone_learning_rate"])}:
    raise RuntimeError(f"Direct DINO LR split is not active: {direct_backbone_lrs}")
del direct_model
gc.collect()

checkpoint = torch.load(request["keeper_checkpoint"], map_location="cpu", weights_only=False)
if list(checkpoint.get("class_names", [])) != expected_classes:
    raise RuntimeError("Keeper class order does not match A0.")
if str(Path(checkpoint.get("data_yaml", "")).resolve()) != str(Path(request["data_yaml"]).resolve()):
    raise RuntimeError("Keeper data.yaml does not match the locked A0 dataset.")

hybrid_config = dict(checkpoint["model_config"])
hybrid_config.update(
    {
        "model_type": "vit_registers_pretrained_hybrid",
        "research_track": "pretrained",
        "pretrained": True,
        "timm_model_name": request["dino_model_name"],
        "pretrained_checkpoint_path": request["dino_checkpoint"],
        "pretrained_checkpoint_sha256": request["dino_sha256"],
        "pretrained_source_url": request["dino_source_url"],
        "pretrained_source_revision": request["dino_source_revision"],
        "pretrained_source_license": request["dino_source_license"],
        "pretrained_semantic_expected_prefix_tokens": 5,
        "pretrained_semantic_expected_embed_dim": 384,
        "pretrained_semantic_expected_patch_count": 256,
        "pretrained_semantic_dropout": 0.05,
        "pretrained_semantic_initial_scale": 0.0,
        "pretrained_semantic_max_scale": 0.25,
        "pretrained_backbone_gradient_checkpointing": True,
    }
)
hybrid_model = create_model(num_classes=5, model_config=hybrid_config)
resume_state = checkpoint["model_state"]
partial_load = _load_model_state_allowing_extensions(
    hybrid_model,
    resume_state,
    allow_extensions=True,
)
if not partial_load:
    raise RuntimeError("Hybrid resume did not report the newly added pretrained branch.")
missing = list(partial_load.get("allowed_missing_keys", []))
if not missing or any(not key.startswith("pretrained_semantic_branch.") for key in missing):
    raise RuntimeError(f"Unexpected hybrid resume gap: {missing!r}")
if partial_load.get("unexpected_keys"):
    raise RuntimeError(f"Unexpected keeper keys: {partial_load['unexpected_keys']!r}")

hybrid_provenance = getattr(hybrid_model, "pretrained_provenance", {})
hybrid_digest = (
    hybrid_provenance.get("semantic_branch", {})
    .get("source", {})
    .get("checkpoint", {})
    .get("sha256", "")
)
if hybrid_digest != request["dino_sha256"]:
    raise RuntimeError(f"Hybrid model did not consume the locked DINO file: {hybrid_digest}")
gate = float(hybrid_model.pretrained_semantic_branch.effective_gate().detach().cpu().item())
if abs(gate) > 1e-12:
    raise RuntimeError(f"A0 keeper identity gate is not zero: {gate}")
keeper_model = build_model_from_checkpoint(checkpoint).eval()
hybrid_model.eval()
torch.manual_seed(20260729)
identity_images = torch.randn(
    1,
    3,
    int(checkpoint["model_config"].get("image_size", 256)),
    int(checkpoint["model_config"].get("image_size", 256)),
)
with torch.no_grad():
    keeper_logits, _ = extract_bbox_from_model_output(keeper_model(identity_images))
    hybrid_logits, _ = extract_bbox_from_model_output(hybrid_model(identity_images))
torch.testing.assert_close(hybrid_logits, keeper_logits, rtol=0.0, atol=0.0)
del keeper_model, identity_images, keeper_logits, hybrid_logits
hybrid_parameters = sum(parameter.numel() for parameter in hybrid_model.parameters())
hybrid_groups = build_optimizer_param_groups(
    hybrid_model,
    weight_decay=0.05,
    learning_rate=float(request["learning_rate"]),
    backbone_lr_scale=float(request["pretrained_backbone_lr_scale"]),
)
hybrid_group_names = {group["name"] for group in hybrid_groups}
hybrid_backbone_lrs = {
    float(group["lr"])
    for group in hybrid_groups
    if group["name"].startswith("backbone_") and "lr" in group
}
if not any(name.startswith("head_") for name in hybrid_group_names):
    raise RuntimeError(f"Hybrid has no keeper/adapter optimizer group: {hybrid_group_names}")
if hybrid_backbone_lrs != {float(request["pretrained_backbone_learning_rate"])}:
    raise RuntimeError(f"Hybrid DINO LR split is not active: {hybrid_backbone_lrs}")
del hybrid_model
gc.collect()

if not torch.cuda.is_available():
    raise RuntimeError("CUDA is unavailable on the target machine.")
if not torch.cuda.is_bf16_supported():
    raise RuntimeError("Target CUDA device does not report BF16 support.")

result = {
    "torch_version": torch.__version__,
    "timm_version": timm.__version__,
    "cuda_available": True,
    "cuda_device": torch.cuda.get_device_name(0),
    "cuda_total_memory_mib": round(torch.cuda.get_device_properties(0).total_memory / 2**20, 1),
    "bf16_supported": True,
    "class_names": class_names,
    "direct_parameter_count": direct_parameters,
    "hybrid_parameter_count": hybrid_parameters,
    "direct_optimizer_groups": sorted(direct_group_names),
    "hybrid_optimizer_groups": sorted(hybrid_group_names),
    "matched_head_learning_rate": float(request["learning_rate"]),
    "matched_dino_backbone_learning_rate": float(request["pretrained_backbone_learning_rate"]),
    "hybrid_keeper_resume_missing_key_count": len(missing),
    "hybrid_initial_effective_gate": gate,
    "dino_sha256_consumed_by_both_models": request["dino_sha256"],
}
print("TRKH_PREFLIGHT_JSON=" + json.dumps(result, ensure_ascii=False, sort_keys=True))
'@

$PythonPreflightPath = Join-Path $PreflightRoot "preflight_runtime_$Timestamp.py"
[System.IO.File]::WriteAllText(
    $PythonPreflightPath,
    $PythonPreflight,
    [System.Text.UTF8Encoding]::new($false)
)
$PreviousErrorActionPreference = $ErrorActionPreference
$ErrorActionPreference = "Continue"
$ModelPreflightOutput = @(& $resolvedPython $PythonPreflightPath $RequestPath 2>&1)
$ModelPreflightExitCode = $LASTEXITCODE
$ErrorActionPreference = $PreviousErrorActionPreference
foreach ($line in $ModelPreflightOutput) {
    Write-Host $line
}
if ($ModelPreflightExitCode -ne 0) {
    throw "Exact local DINO/keeper model-construction preflight failed."
}
$Sentinel = "TRKH_PREFLIGHT_JSON="
$ModelJsonLine = @(
    $ModelPreflightOutput |
        Where-Object { ([string]$_).StartsWith($Sentinel) }
) | Select-Object -Last 1
if ($null -eq $ModelJsonLine) {
    throw "Model preflight did not emit its JSON result."
}
$ModelPreflight = ([string]$ModelJsonLine).Substring($Sentinel.Length) | ConvertFrom-Json

$GpuSnapshot = @()
$GpuProcesses = @()
if ($null -ne (Get-Command nvidia-smi -ErrorAction SilentlyContinue)) {
    $GpuSnapshot = @(& nvidia-smi --query-gpu=name,memory.total,memory.free,driver_version --format=csv,noheader)
    $GpuProcesses = @(& nvidia-smi --query-compute-apps=pid,process_name,used_memory --format=csv,noheader)
}

$ManifestPath = Join-Path $PreflightRoot "$($Mode.ToLowerInvariant())_manifest_$Timestamp.json"
$Manifest = [ordered]@{
    protocol = $ProtocolId
    created_utc = (Get-Date).ToUniversalTime().ToString("o")
    mode = $Mode.ToLowerInvariant()
    selected_arm = $Arm.ToLowerInvariant()
    research_track = "pretrained"
    test_locked = $true
    label_only = $true
    git = [ordered]@{
        branch = $script:GitBranch
        commit = $script:GitCommit
        status_short = $GitStatus
        clean = ($GitStatus.Count -eq 0)
    }
    inputs = [ordered]@{
        data_yaml = $resolvedDataYaml
        data_yaml_sha256 = $DataYamlSha256
        validation_support = $ValidationCount
        class_names = $ExpectedClassNames
        keeper_checkpoint = $resolvedKeeper
        keeper_sha256 = $KeeperSha256
        dino_model_name = $DinoModelName
        dino_checkpoint = $resolvedDino
        dino_checkpoint_sha256 = $DinoSha256
        dino_source_url = $DinoSourceUrl
        dino_source_revision = $DinoSourceRevision
        dino_source_license = $DinoSourceLicense
    }
    runtime = [ordered]@{
        amp_dtype = "bf16"
        batch_size = $BatchSize
        grad_accum_steps = $GradAccumSteps
        effective_batch_size = $EffectiveBatchSize
        num_workers = $NumWorkers
        eval_num_workers = $EvalNumWorkers
        epochs = $Epochs
        scheduler_total_epochs = $SchedulerTotalEpochs
        warmup_epochs = $WarmupEpochs
        learning_rate = $LearningRate
        pretrained_backbone_lr_scale = $PretrainedBackboneLrScale
        pretrained_backbone_learning_rate = $PretrainedBackboneLearningRate
        max_train_batches = $MaxTrainBatches
        max_val_batches = $MaxValBatches
        python = $resolvedPython
        model_preflight = $ModelPreflight
        nvidia_smi_gpu = $GpuSnapshot
        nvidia_smi_compute_processes = $GpuProcesses
    }
    invariants = [ordered]@{
        hybrid_resume_inherits_keeper_config = $true
        hybrid_resume_use_cli_config = $false
        hybrid_initial_scale = 0.0
        teacher_csv = $null
        teacher_losses_zero = $true
        full_validation_required_for_selection = ($Mode -eq "Probe" -or $Mode -eq "Full")
    }
    direct_train_args = $DirectArgs
    hybrid_train_args = $HybridArgs
    request_manifest = $RequestPath
}
Write-JsonFile -Value $Manifest -Path $ManifestPath
Write-Host "A0 preflight passed. Manifest: $ManifestPath"

if ($Mode -eq "Preflight") {
    Write-Host "Preflight mode does not start training."
    exit 0
}

$SelectedArms = @(Get-SelectedArms)
if ($Mode -eq "Probe") {
    foreach ($selected in $SelectedArms) {
        Assert-StageMarker -SelectedArm $selected -RequiredStage "Smoke"
    }
}
if ($Mode -eq "Full") {
    if (-not $ConfirmFull.IsPresent) {
        throw "Full requires the explicit -ConfirmFull switch."
    }
    foreach ($selected in @("Direct", "Hybrid")) {
        Assert-StageMarker -SelectedArm $selected -RequiredStage "Probe"
    }
    if ([string]::IsNullOrWhiteSpace($GateManifest)) {
        throw "Full requires -GateManifest from the paired source-group probe audit."
    }
    $ResolvedGateManifest = Assert-FullGateManifest -Path $GateManifest
    Write-Host "Full gate passed: $ResolvedGateManifest"
}

function Invoke-A0Arm {
    param([Parameter(Mandatory = $true)][string]$SelectedArm)

    if ($SelectedArm -eq "Direct") {
        $runName = $DirectRunName
        $trainArgs = $DirectArgs
    }
    elseif ($SelectedArm -eq "Hybrid") {
        $runName = $HybridRunName
        $trainArgs = $HybridArgs
    }
    else {
        throw "Unsupported arm: $SelectedArm"
    }

    $runDir = Join-Path $OutputRoot $runName
    if (Test-Path -LiteralPath $runDir) {
        throw "Run directory already exists; choose a new RunTag instead of overwriting evidence: $runDir"
    }
    New-Item -ItemType Directory -Path $runDir -Force | Out-Null

    $commandTokens = @($resolvedPython, "-m", "trkh.training.train") + $trainArgs
    $commandLine = ($commandTokens | ForEach-Object {
        $token = [string]$_
        if ($token.Contains(" ")) { '"' + $token.Replace('"', '\"') + '"' } else { $token }
    }) -join " "
    $launchPath = Join-Path $runDir "a0_launch_request.json"
    Write-JsonFile -Path $launchPath -Value ([ordered]@{
        protocol = $ProtocolId
        stage = $StageName.ToLowerInvariant()
        arm = $SelectedArm.ToLowerInvariant()
        created_utc = (Get-Date).ToUniversalTime().ToString("o")
        git_commit = $script:GitCommit
        preflight_manifest = $ManifestPath
        test_locked = $true
        command = $commandLine
        train_args = $trainArgs
    })

    Write-Host "Starting $SelectedArm ${StageName}: $runName"
    Push-Location $RepoRoot
    try {
        & $resolvedPython -m trkh.training.train @trainArgs
        $exitCode = $LASTEXITCODE
    }
    finally {
        Pop-Location
    }
    if ($exitCode -ne 0) {
        throw "$SelectedArm $StageName failed with exit code $exitCode. Evidence retained at $runDir"
    }

    $markerPath = Join-Path $runDir "a0_stage_complete.json"
    Write-JsonFile -Path $markerPath -Value ([ordered]@{
        protocol = $ProtocolId
        stage = $StageName.ToLowerInvariant()
        arm = $SelectedArm.ToLowerInvariant()
        run_name = $runName
        completed_utc = (Get-Date).ToUniversalTime().ToString("o")
        exit_code = 0
        git_commit = $script:GitCommit
        preflight_manifest = $ManifestPath
        test_locked = $true
        batch_size = $BatchSize
        grad_accum_steps = $GradAccumSteps
        effective_batch_size = $EffectiveBatchSize
    })
    Write-Host "Completed $SelectedArm $StageName. Marker: $markerPath"
}

foreach ($selected in $SelectedArms) {
    Invoke-A0Arm -SelectedArm $selected
}

Write-Host "A0 $StageName completed for arm selection: $Arm"
