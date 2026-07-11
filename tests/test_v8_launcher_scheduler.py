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
