"""Settings for the Bayesian ML Platform.

All thresholds are overridable through ``BMP_*`` environment variables
(see ``.env.example``). Defaults are the production gates used by the demo.
"""
from __future__ import annotations

import os
from dataclasses import dataclass, field


def _env_float(name: str, default: float) -> float:
    try:
        return float(os.environ[name])
    except (KeyError, ValueError):
        return default


def _env_int(name: str, default: int) -> int:
    try:
        return int(os.environ[name])
    except (KeyError, ValueError):
        return default


@dataclass
class Settings:
    # --- data / reproducibility -------------------------------------------
    seed: int = field(default_factory=lambda: _env_int("BMP_SEED", 42))
    n_samples: int = field(default_factory=lambda: _env_int("BMP_N_SAMPLES", 1200))
    n_features: int = field(default_factory=lambda: _env_int("BMP_N_FEATURES", 6))
    test_fraction: float = field(default_factory=lambda: _env_float("BMP_TEST_FRACTION", 0.25))
    data_version: str = "synthetic-v1"

    # --- bayesian logistic regression (Laplace) ---------------------------
    prior_var: float = field(default_factory=lambda: _env_float("BMP_PRIOR_VAR", 4.0))
    max_newton_iters: int = field(default_factory=lambda: _env_int("BMP_MAX_NEWTON_ITERS", 50))
    newton_tol: float = field(default_factory=lambda: _env_float("BMP_NEWTON_TOL", 1e-6))
    n_posterior_samples: int = field(
        default_factory=lambda: _env_int("BMP_N_POSTERIOR_SAMPLES", 400)
    )

    # --- MCMC --------------------------------------------------------------
    mcmc_samples: int = field(default_factory=lambda: _env_int("BMP_MCMC_SAMPLES", 2000))
    mcmc_burn_in: int = field(default_factory=lambda: _env_int("BMP_MCMC_BURN_IN", 500))
    mcmc_proposal_std: float = field(
        default_factory=lambda: _env_float("BMP_MCMC_PROPOSAL_STD", 0.08)
    )

    # --- serving / decisioning ---------------------------------------------
    abstain_std_threshold: float = field(
        default_factory=lambda: _env_float("BMP_ABSTAIN_STD", 0.18)
    )
    cost_false_positive: float = field(default_factory=lambda: _env_float("BMP_COST_FP", 1.0))
    cost_false_negative: float = field(default_factory=lambda: _env_float("BMP_COST_FN", 5.0))
    serve_budget_ms: float = field(default_factory=lambda: _env_float("BMP_SERVE_BUDGET_MS", 100.0))

    # --- promotion gates ----------------------------------------------------
    accuracy_floor: float = field(default_factory=lambda: _env_float("BMP_ACCURACY_FLOOR", 0.75))
    auc_floor: float = field(default_factory=lambda: _env_float("BMP_AUC_FLOOR", 0.80))
    ece_ceiling: float = field(default_factory=lambda: _env_float("BMP_ECE_CEILING", 0.08))
    coverage_floor: float = field(default_factory=lambda: _env_float("BMP_COVERAGE_FLOOR", 0.80))
    min_eval_cases: int = field(default_factory=lambda: _env_int("BMP_MIN_EVAL_CASES", 100))

    # --- drift ----------------------------------------------------------------
    kl_warn: float = field(default_factory=lambda: _env_float("BMP_KL_WARN", 0.5))
    kl_critical: float = field(default_factory=lambda: _env_float("BMP_KL_CRITICAL", 2.0))
    psi_warn: float = field(default_factory=lambda: _env_float("BMP_PSI_WARN", 0.10))
    psi_critical: float = field(default_factory=lambda: _env_float("BMP_PSI_CRITICAL", 0.25))

    def as_dict(self) -> dict:
        return {
            "seed": self.seed,
            "n_samples": self.n_samples,
            "n_features": self.n_features,
            "test_fraction": self.test_fraction,
            "data_version": self.data_version,
            "prior_var": self.prior_var,
            "max_newton_iters": self.max_newton_iters,
            "newton_tol": self.newton_tol,
            "n_posterior_samples": self.n_posterior_samples,
            "mcmc_samples": self.mcmc_samples,
            "mcmc_burn_in": self.mcmc_burn_in,
            "mcmc_proposal_std": self.mcmc_proposal_std,
            "abstain_std_threshold": self.abstain_std_threshold,
            "cost_false_positive": self.cost_false_positive,
            "cost_false_negative": self.cost_false_negative,
            "serve_budget_ms": self.serve_budget_ms,
            "accuracy_floor": self.accuracy_floor,
            "auc_floor": self.auc_floor,
            "ece_ceiling": self.ece_ceiling,
            "coverage_floor": self.coverage_floor,
            "min_eval_cases": self.min_eval_cases,
            "kl_warn": self.kl_warn,
            "kl_critical": self.kl_critical,
            "psi_warn": self.psi_warn,
            "psi_critical": self.psi_critical,
        }
