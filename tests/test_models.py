import math

import pytest

from bayesian.evaluation import evaluate, roc_auc
from bayesian.models import (
    BayesianLinearRegression,
    BayesianLogisticRegression,
    PointLogisticRegression,
)


def test_logistic_fit_predicts_better_than_chance(split):
    model = BayesianLogisticRegression(seed=42).fit(split.X_train, split.y_train)
    probs = model.predict_proba(split.X_test)
    assert all(0.0 <= p <= 1.0 for p in probs)
    assert roc_auc(probs, split.y_test) > 0.80
    preds = model.predict(split.X_test)
    acc = sum(p == t for p, t in zip(preds, split.y_test)) / len(split.y_test)
    assert acc > 0.70


def test_posterior_covariance_is_symmetric_psd(model):
    cov = model.cov_
    d = len(cov)
    for i in range(d):
        for j in range(d):
            assert abs(cov[i][j] - cov[j][i]) < 1e-9
        assert cov[i][i] > 0


def test_predictive_uncertainty_intervals_sane(model, split):
    unc = model.predict_proba_with_uncertainty(split.X_test[:20])
    assert len(unc) == 20
    for u in unc:
        assert 0.0 <= u["lo_90"] <= u["mean"] <= u["hi_90"] <= 1.0
        assert u["std"] >= 0.0
        assert u["width_90"] >= 0.0


def test_uncertainty_grows_away_from_data(model):
    # Far-out-of-distribution rows should carry wider credible intervals.
    near = model.predict_proba_with_uncertainty([[0.0] * 6])[0]
    far = model.predict_proba_with_uncertainty([[8.0] * 6])[0]
    assert far["width_90"] > near["width_90"]


def test_probit_vs_mc_agree(model, split):
    probit = model.predict_proba(split.X_test[:30])
    mc = model.predict_proba_with_uncertainty(split.X_test[:30])
    for p, u in zip(probit, mc):
        assert abs(p - u["mean"]) < 0.15


def test_unfitted_model_raises():
    model = BayesianLogisticRegression()
    with pytest.raises(ValueError):
        model.predict_proba([[0.0] * 6])


def test_bayesian_linear_regression_closed_form():
    # y = 1 + 2*x0 exactly (no noise beyond beta); posterior mean should
    # recover the weights closely with a tight prior on noise.
    X = [[float(i)] for i in range(10)]
    y = [1.0 + 2.0 * i for i in range(10)]
    model = BayesianLinearRegression(alpha=0.01, beta=100.0).fit(X, y)
    assert abs(model.mean_[0] - 1.0) < 0.05
    assert abs(model.mean_[1] - 2.0) < 0.05
    preds = model.predict_with_uncertainty([[10.0]])
    assert abs(preds[0]["mean"] - 21.0) < 0.5
    assert preds[0]["std"] > 0


def test_point_model_agrees_with_bayesian_map(model, point_model, split):
    p_bayes = model.predict_proba(split.X_test[:50])
    p_point = point_model.predict_proba(split.X_test[:50])
    for a, b in zip(p_bayes, p_point):
        assert abs(a - b) < 0.05


def test_evaluate_report_fields(model, split):
    from bayesian.data import true_probabilities

    probs = model.predict_proba(split.X_test)
    intervals = model.predict_proba_with_uncertainty(split.X_test)
    report = evaluate(probs, intervals, split.y_test, true_probs=true_probabilities(split.X_test))
    assert 0.0 <= report.accuracy <= 1.0
    assert 0.0 <= report.coverage_90 <= 1.0
    assert report.n == len(split.y_test)
    assert report.confusion["tp"] + report.confusion["fn"] == sum(split.y_test)


def test_coverage_is_honest():
    """90% credible intervals should contain the true probability
    roughly 90% of the time (allowing sampling noise). Uses the full
    demo-size data: at n=600 the MAP estimate is noisy enough that
    coverage is unstable, so the honest check runs at n=1200."""
    from bayesian.data import generate_dataset, train_test_split, true_probabilities

    data = generate_dataset(n=1200, seed=42)
    split = train_test_split(data, test_fraction=0.25, seed=42)
    fitted = BayesianLogisticRegression(seed=42).fit(split.X_train, split.y_train)
    intervals = fitted.predict_proba_with_uncertainty(split.X_test)
    report = evaluate(
        fitted.predict_proba(split.X_test),
        intervals,
        split.y_test,
        true_probs=true_probabilities(split.X_test),
    )
    assert report.coverage_90 >= 0.80
