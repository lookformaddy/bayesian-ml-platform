import pytest

from bayesian.drift import drift_report, ks_statistic, population_stability_index


def test_psi_identical_distributions_is_zero():
    probs = [i / 100 for i in range(100)]
    assert population_stability_index(probs, probs) == pytest.approx(0.0)


def test_psi_shifted_distribution_is_large():
    expected = [0.5] * 100
    actual = [0.95] * 100
    assert population_stability_index(expected, actual) > 0.25


def test_ks_identical_is_zero():
    assert ks_statistic([0.1, 0.5, 0.9], [0.1, 0.5, 0.9]) == pytest.approx(0.0)


def test_ks_disjoint_is_one():
    assert ks_statistic([0.0] * 10, [1.0] * 10) == pytest.approx(1.0)


def test_drift_report_no_action_when_identical():
    mean = [0.0, 0.0]
    cov = [[1.0, 0.0], [0.0, 1.0]]
    probs = [0.3, 0.5, 0.7] * 20
    report = drift_report(mean, cov, mean, cov, probs, probs)
    assert report.action == "no_action"
    assert report.kl_level == "ok"
    assert report.psi_level == "ok"


def test_drift_report_critical_on_large_shift():
    mean0 = [0.0, 0.0]
    mean1 = [3.0, 3.0]
    cov = [[1.0, 0.0], [0.0, 1.0]]
    report = drift_report(
        mean0, cov, mean1, cov, [0.5] * 100, [0.95] * 100
    )
    assert report.action == "recalibrate_and_refit"
    assert report.kl_level == "critical"
    d = report.to_dict()
    assert d["kl_divergence"] > 2.0
