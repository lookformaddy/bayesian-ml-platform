"""MCMC cross-check tests — small data / short chains to stay fast."""
import pytest

from bayesian.mcmc import MetropolisHastings
from bayesian.models import BayesianLogisticRegression


@pytest.fixture()
def small_split():
    from bayesian.data import generate_dataset, train_test_split

    data = generate_dataset(n=200, seed=7)
    return train_test_split(data, test_fraction=0.25, seed=7)


def test_mcmc_mean_matches_laplace(small_split):
    model = BayesianLogisticRegression(seed=7).fit(
        small_split.X_train, small_split.y_train
    )
    result = MetropolisHastings(
        n_samples=600, burn_in=200, proposal_std=0.12, seed=7
    ).sample(small_split.X_train, small_split.y_train, init=model.mean_)
    assert 0.05 < result.acceptance_rate < 0.95
    assert result.ess_min > 20
    for m_lap, m_mcmc in zip(model.mean_, result.posterior_mean):
        assert abs(m_lap - m_mcmc) < 0.35


def test_mcmc_is_deterministic(small_split):
    sampler = MetropolisHastings(n_samples=200, burn_in=50, seed=3)
    r1 = sampler.sample(small_split.X_train, small_split.y_train)
    r2 = sampler.sample(small_split.X_train, small_split.y_train)
    assert r1.posterior_mean == r2.posterior_mean
    assert r1.acceptance_rate == r2.acceptance_rate


def test_mcmc_result_serializes(small_split):
    result = MetropolisHastings(n_samples=100, burn_in=20, seed=3).sample(
        small_split.X_train, small_split.y_train
    )
    d = result.to_dict()
    assert d["n_samples"] == 100
    assert 0.0 <= d["acceptance_rate"] <= 1.0
