import math

import pytest

from bayesian.decision import decide, expected_cost_of_policy, optimal_threshold


def test_optimal_threshold_asymmetric_costs():
    # Missing a positive costs 5x a false alarm -> threshold 1/6.
    assert optimal_threshold(1.0, 5.0) == pytest.approx(1.0 / 6.0)
    assert optimal_threshold(1.0, 1.0) == pytest.approx(0.5)


def test_optimal_threshold_rejects_nonpositive_costs():
    with pytest.raises(ValueError):
        optimal_threshold(0.0, 5.0)


def test_decide_act_above_threshold():
    d = decide(0.8, 0.05, cost_fp=1.0, cost_fn=5.0, abstain_std_threshold=0.18)
    assert d.action == "act"
    assert d.expected_cost == pytest.approx(0.2)


def test_decide_no_act_below_threshold():
    d = decide(0.05, 0.05, cost_fp=1.0, cost_fn=5.0, abstain_std_threshold=0.18)
    assert d.action == "no_act"
    assert d.expected_cost == pytest.approx(0.25)


def test_decide_abstains_on_high_uncertainty():
    d = decide(0.9, 0.30, cost_fp=1.0, cost_fn=5.0, abstain_std_threshold=0.18)
    assert d.action == "abstain"
    assert math.isnan(d.expected_cost)


def test_cost_policy_beats_0_5_baseline():
    # With FN cost 5x FP cost, the optimal threshold should not cost more
    # than the naive 0.5 threshold on this synthetic set.
    probs = [0.2, 0.3, 0.4, 0.6, 0.8, 0.9]
    y = [0, 1, 1, 0, 1, 1]
    out = expected_cost_of_policy(probs, y, cost_fp=1.0, cost_fn=5.0)
    assert out["total_cost"] <= out["baseline_0.5_cost"]
    assert out["threshold"] == pytest.approx(1.0 / 6.0, abs=1e-4)
