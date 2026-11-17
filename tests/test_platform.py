import pytest

from bayesian.config import Settings
from bayesian.demo import run_demo
from bayesian.platform import BayesianPlatform


def test_pipeline_promotes_and_reports():
    # Full demo-size data: at n=600 the MAP estimate is noisy enough
    # that the accuracy / coverage gates are unstable.
    platform = BayesianPlatform(Settings())
    result = platform.run_pipeline(version="v1")
    assert result.promoted is True, result.promotion_message
    assert result.evaluation["accuracy"] >= 0.75
    assert result.evaluation["auc"] >= 0.80
    assert result.evaluation["coverage_90"] >= 0.80
    assert result.manifest["config_hash"]
    assert result.registry.audit.verify() is True


def test_bandit_comparison_thompson_wins():
    platform = BayesianPlatform(Settings())
    out = platform.run_bandit_comparison(n_rounds=300)
    assert out["regret_saved"] > 0


def test_drift_scenario_triggers_recalibration():
    platform = BayesianPlatform(Settings())
    result = platform.run_pipeline(version="v1")
    scenario = platform.run_drift_scenario(result)
    assert scenario["drift"]["action"] == "recalibrate_and_refit"
    assert scenario["recalibration_after_drift"] is not None


def test_demo_runs_end_to_end(capsys):
    summary = run_demo(Settings())
    out = capsys.readouterr().out
    assert "accuracy" in out
    assert "ECE before calibration" in out
    assert "Promotion" in out
    assert summary["promoted"] is True
    assert len(summary["served_sample"]) == 5
