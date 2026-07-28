import re
from pathlib import Path


def test_v8_launcher_exposes_independent_scheduler_horizon() -> None:
    script = (
        Path(__file__).resolve().parents[1]
        / "scripts"
        / "run_trkh_5class_attention_views_v8.ps1"
    ).read_text(encoding="utf-8")

    assert "[int]$SchedulerTotalEpochs = 0" in script
    assert '"--scheduler-total-epochs", "$SchedulerTotalEpochsEffective"' in script
    assert "scheduler_total_epochs = $SchedulerTotalEpochsEffective" in script
    assert '"--scheduler-total-epochs", "$Epochs"' not in script


def test_v8_launcher_exposes_explicit_resume_modes() -> None:
    script = (
        Path(__file__).resolve().parents[1]
        / "scripts"
        / "run_trkh_5class_attention_views_v8.ps1"
    ).read_text(encoding="utf-8")

    assert '[ValidateSet("Scratch", "WarmStart", "StatefulResume")]' in script
    assert '[string]$ResumeMode = "WarmStart"' in script
    assert '"Scratch" {' in script
    assert '"WarmStart" {' in script
    assert '"StatefulResume" {' in script
    assert 'ResumeMode=Scratch tu choi ResumeCheckpoint khong rong' in script
    assert '$PSBoundParameters.ContainsKey("ResumeCheckpoint")' in script
    assert '$ResumeCheckpoint = ""' in script
    assert 'resume_mode = $ResumeMode' in script
    assert "StatefulResume must use the same run's checkpoints\\last.pt" in script

    scratch_block = script.split('"Scratch" {', 1)[1].split('"WarmStart" {', 1)[0]
    warm_start_block = script.split('"WarmStart" {', 1)[1].split('"StatefulResume" {', 1)[0]
    stateful_block = script.split('"StatefulResume" {', 1)[1].split('default {', 1)[0]
    assert '"--disable-resume"' in scratch_block
    for flag in (
        '"--resume-reset-epoch"',
        '"--resume-reset-optimizer"',
        '"--resume-reset-scheduler"',
        '"--resume-reset-scaler"',
    ):
        assert flag in warm_start_block
        assert flag not in stateful_block


def test_direct_scratch_callers_select_scratch_resume_mode() -> None:
    scripts_dir = Path(__file__).resolve().parents[1] / "scripts"
    callers_with_empty_resume = {}
    for path in scripts_dir.glob("*.ps1"):
        script = path.read_text(encoding="utf-8")
        if (
            "run_trkh_5class_attention_views_v8.ps1" in script
            and re.search(r'ResumeCheckpoint\s*=\s*""', script)
        ):
            callers_with_empty_resume[path.name] = script

    expected_callers = {
        "run_trkh_5class_concurrent_local_global_scratch.ps1",
        "run_trkh_5class_leff_scratch.ps1",
        "run_trkh_5class_stagewise_mbconv_scratch.ps1",
        "run_trkh_bi_level_routing_attention_a1.ps1",
        "run_trkh_cropr_token_selector_a0.ps1",
        "run_trkh_deformable_spatial_attention_a1.ps1",
        "run_trkh_diverse_branch_stem_a0.ps1",
        "run_trkh_foveal_aggregated_attention_a1.ps1",
        "run_trkh_inattentive_token_fusion_a0.ps1",
        "run_trkh_yolof_oof_fold_experts.ps1",
    }
    assert set(callers_with_empty_resume) == expected_callers

    oof_script = callers_with_empty_resume.pop("run_trkh_yolof_oof_fold_experts.ps1")
    assert (
        'ResumeMode = if ([string]::IsNullOrWhiteSpace($ResumeCheckpoint)) '
        '{ "Scratch" } else { "WarmStart" }'
    ) in oof_script
    for script in callers_with_empty_resume.values():
        assert 'ResumeMode = "Scratch"' in script
