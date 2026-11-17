"""End-to-end demo: train, MCMC-check, calibrate, evaluate, gate, bandit,
drift, and serve — printing the headline numbers."""
from __future__ import annotations

from .config import Settings
from .platform import BayesianPlatform


def run_demo(settings: Settings | None = None) -> dict:
    s = settings or Settings()
    platform = BayesianPlatform(s)
    result = platform.run_pipeline(version="v1")
    result.bandit = platform.run_bandit_comparison(n_rounds=500)
    result.drift = platform.run_drift_scenario(result)

    # Serve a handful of test rows to show abstention in action.
    from .data import generate_dataset, train_test_split

    data = generate_dataset(n=s.n_samples, n_features=s.n_features, seed=s.seed)
    split = train_test_split(data, test_fraction=s.test_fraction, seed=s.seed)
    served = [result.server.predict_one(x).to_dict() for x in split.X_test[:5]]

    ev, cal, mcmc = result.evaluation, result.calibration, result.mcmc
    print("== Bayesian ML Platform — demo ==")
    print(f"accuracy {ev['accuracy']:.4f} | AUC {ev['auc']:.4f} | log-loss {ev['logloss']:.4f}")
    print(f"ECE before calibration {cal['ece_before']:.4f} -> after {cal['ece_after']:.4f}")
    print(f"Brier before {cal['brier_before']:.4f} -> after {cal['brier_after']:.4f}")
    print(
        f"90% credible-interval coverage {ev['coverage_90']:.4f} "
        f"(mean width {ev['mean_width_90']:.3f})"
    )
    print(
        f"MCMC cross-check: acceptance {mcmc['acceptance_rate']:.3f}, "
        f"min ESS {mcmc['ess_min']:.0f}"
    )
    print(
        f"Cost-aware threshold {result.cost_policy['threshold']:.3f}: "
        f"total cost {result.cost_policy['total_cost']} vs "
        f"0.5-baseline {result.cost_policy['baseline_0.5_cost']} "
        f"(saved {result.cost_policy['cost_saved_vs_baseline']})"
    )
    b = result.bandit
    print(
        f"Bandit regret (500 rounds): Thompson {b['thompson_sampling']['cumulative_regret']:.1f} "
        f"vs epsilon-greedy {b['epsilon_greedy']['cumulative_regret']:.1f} "
        f"(saved {b['regret_saved']:.1f})"
    )
    d = result.drift["drift"]
    print(
        f"Drift event: KL {d['kl_divergence']:.3f} ({d['kl_level']}), "
        f"PSI {d['psi']:.3f} ({d['psi_level']}) -> action {d['action']}"
    )
    if result.drift["recalibration_after_drift"]:
        r = result.drift["recalibration_after_drift"]
        print(f"  recalibration after drift: ECE {r['ece_before']:.4f} -> {r['ece_after']:.4f}")
    print(f"Promotion: {result.promotion_message}")
    print(f"Serving stats: {result.server.stats()}")
    for row in served:
        print(
            f"  p={row['probability']:.3f} std={row['std']} "
            f"CI=[{row['lo_90']:.3f}, {row['hi_90']:.3f}] action={row['action']} "
            f"fallback={row['fallback']}"
        )
    summary = result.summary()
    summary["served_sample"] = served
    summary["serving_stats"] = result.server.stats()
    return summary


if __name__ == "__main__":
    run_demo()
