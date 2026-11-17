from bayesian.bandit import run_epsilon_greedy, run_thompson_sampling


def test_thompson_beats_epsilon_greedy_regret():
    rates = [0.20, 0.80, 0.40]
    ts = run_thompson_sampling(rates, n_rounds=500, seed=1)
    eg = run_epsilon_greedy(rates, n_rounds=500, seed=1)
    assert ts.cumulative_regret < eg.cumulative_regret
    assert ts.total_reward > eg.total_reward


def test_thompson_finds_best_arm():
    rates = [0.20, 0.80, 0.40]
    result = run_thompson_sampling(rates, n_rounds=500, seed=1)
    best_arm = result.pulls_per_arm.index(max(result.pulls_per_arm))
    assert best_arm == 1
    assert result.estimated_rates[1] > 0.65


def test_bandit_deterministic():
    rates = [0.3, 0.6]
    r1 = run_thompson_sampling(rates, n_rounds=100, seed=5)
    r2 = run_thompson_sampling(rates, n_rounds=100, seed=5)
    assert r1.to_dict() == r2.to_dict()


def test_epsilon_greedy_explores_all_arms():
    result = run_epsilon_greedy([0.3, 0.5, 0.4], n_rounds=300, seed=2)
    assert all(p > 0 for p in result.pulls_per_arm)
    assert sum(result.pulls_per_arm) == 300
