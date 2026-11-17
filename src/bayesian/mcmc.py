"""Deterministic Metropolis-Hastings MCMC for the logistic-regression
posterior — a cross-check on the Laplace approximation.

Everything runs on a seeded ``random.Random`` so the chain is fully
reproducible. Diagnostics: acceptance rate, posterior mean, and a simple
effective-sample-size estimate from lag-1 autocorrelation.
"""
from __future__ import annotations

import math
import random
from dataclasses import dataclass, field

from .models import _log_posterior_logreg


@dataclass
class McmcResult:
    samples: list[list[float]]
    acceptance_rate: float
    posterior_mean: list[float]
    posterior_std: list[float]
    ess_min: float

    def to_dict(self) -> dict:
        return {
            "n_samples": len(self.samples),
            "acceptance_rate": round(self.acceptance_rate, 4),
            "posterior_mean": [round(v, 4) for v in self.posterior_mean],
            "posterior_std": [round(v, 4) for v in self.posterior_std],
            "ess_min": round(self.ess_min, 1),
        }


def _ess_estimate(chain: list[float]) -> float:
    """Rough ESS from lag-1 autocorrelation (Geyer-style, truncated)."""
    n = len(chain)
    if n < 10:
        return float(n)
    mean = sum(chain) / n
    var = sum((v - mean) ** 2 for v in chain) / n
    if var <= 0:
        return float(n)
    rho1 = (
        sum((chain[i] - mean) * (chain[i + 1] - mean) for i in range(n - 1))
        / ((n - 1) * var)
    )
    rho1 = max(0.0, min(rho1, 0.95))
    return n / (1.0 + 2.0 * rho1)


@dataclass
class MetropolisHastings:
    n_samples: int = 2000
    burn_in: int = 500
    proposal_std: float = 0.08
    prior_var: float = 4.0
    seed: int = 42

    def sample(
        self,
        X: list[list[float]],
        y: list[int],
        init: list[float] | None = None,
    ) -> McmcResult:
        rng = random.Random(self.seed)
        d = len(X[0]) + 1
        current = [0.0] * d if init is None else list(init)
        current_lp = _log_posterior_logreg(current, X, y, self.prior_var)
        kept: list[list[float]] = []
        accepted = 0
        total = self.n_samples + self.burn_in
        for i in range(total):
            proposal = [c + rng.gauss(0.0, self.proposal_std) for c in current]
            prop_lp = _log_posterior_logreg(proposal, X, y, self.prior_var)
            if math.log(max(rng.random(), 1e-12)) < prop_lp - current_lp:
                current, current_lp = proposal, prop_lp
                accepted += 1
            if i >= self.burn_in:
                kept.append(list(current))
        n = len(kept)
        mean = [sum(s[j] for s in kept) / n for j in range(d)]
        std = [
            math.sqrt(sum((s[j] - mean[j]) ** 2 for s in kept) / n) for j in range(d)
        ]
        ess = min(_ess_estimate([s[j] for s in kept]) for j in range(d))
        return McmcResult(
            samples=kept,
            acceptance_rate=accepted / total,
            posterior_mean=mean,
            posterior_std=std,
            ess_min=ess,
        )
