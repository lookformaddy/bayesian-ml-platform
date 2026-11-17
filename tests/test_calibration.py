import pytest

from bayesian.calibration import (
    BinnedCalibrator,
    PlattCalibrator,
    brier_score,
    calibrate_and_report,
    expected_calibration_error,
    log_loss,
    reliability_bins,
)


def _overconfident_probs(n=400):
    # Deliberately overconfident: probs pushed toward 0/1, labels at 70%.
    probs, y = [], []
    for i in range(n):
        p = 0.95 if i % 2 == 0 else 0.05
        label = 1 if (i * 7) % 10 < 7 else 0
        probs.append(p if i % 2 == 0 else 0.05)
        y.append(label)
    return probs, y


def test_ece_perfect_calibration_is_zero():
    probs = [0.0] * 50 + [1.0] * 50
    y = [0] * 50 + [1] * 50
    assert expected_calibration_error(probs, y) == pytest.approx(0.0)


def test_ece_worst_case_is_high():
    probs = [1.0] * 100
    y = [0] * 100
    assert expected_calibration_error(probs, y) == pytest.approx(1.0)


def test_brier_and_logloss_basics():
    assert brier_score([1.0, 0.0], [1, 0]) == pytest.approx(0.0)
    assert brier_score([0.5, 0.5], [1, 0]) == pytest.approx(0.25)
    assert log_loss([1.0], [1]) < 1e-6


def test_reliability_bins_sum_to_n():
    probs = [i / 100 for i in range(100)]
    y = [1 if p > 0.5 else 0 for p in probs]
    bins = reliability_bins(probs, y, n_bins=10)
    assert sum(b["count"] for b in bins) == 100


def test_platt_improves_overconfident_case():
    probs, y = _overconfident_probs()
    fixed, report = calibrate_and_report(probs, y, method="platt")
    assert report.ece_after <= report.ece_before
    assert all(0.0 <= p <= 1.0 for p in fixed)


def test_binned_calibrator_monotone():
    probs = [i / 200 for i in range(200)]
    y = [1 if (i * 13) % 10 < 5 else 0 for i in range(200)]
    cal = BinnedCalibrator(n_bins=5).fit(probs, y)
    assert all(cal.mapping[i] <= cal.mapping[i + 1] for i in range(4))
    out = cal.transform([0.05, 0.5, 0.95])
    assert out[0] <= out[1] <= out[2]


def test_unfitted_calibrator_raises():
    with pytest.raises(ValueError):
        PlattCalibrator().transform([0.5])
    with pytest.raises(ValueError):
        BinnedCalibrator().transform([0.5])
