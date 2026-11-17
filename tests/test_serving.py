def test_serve_returns_uncertainty(server, split):
    result = server.predict_one(split.X_test[0])
    assert result.fallback is False
    assert result.model == "bayesian"
    assert 0.0 <= result.probability <= 1.0
    assert result.std is not None and result.std >= 0.0
    assert result.lo_90 <= result.probability <= result.hi_90
    assert result.action in ("act", "no_act", "abstain")


def test_serve_abstains_on_extreme_input(server):
    # A far-out-of-distribution row near the decision boundary (large
    # norm, cancelling weights) carries high predictive uncertainty and
    # is referred to a human.
    result = server.predict_one([9.0, 11.0, 0.0, 0.0, 0.0, 0.0])
    assert result.action == "abstain"
    assert result.std is not None and result.std > 0.18


def test_serve_fallback_on_latency_budget(server, split):
    result = server.predict_one(split.X_test[0], simulate_slow=True)
    assert result.fallback is True
    assert result.model == "point"
    assert result.std is None
    assert result.lo_90 is None


def test_serve_stats_track_counts(server, split):
    server.predict_one(split.X_test[0])
    server.predict_one([9.0, 11.0, 0.0, 0.0, 0.0, 0.0])
    server.predict_one(split.X_test[1], simulate_slow=True)
    stats = server.stats()
    assert stats["n_served"] == 3
    assert stats["n_abstained"] == 1
    assert stats["n_fallback"] == 1
    assert stats["abstain_rate"] == 1 / 3
