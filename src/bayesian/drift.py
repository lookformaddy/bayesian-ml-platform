"""Posterior and predictive drift monitoring.

Two complementary signals:
- **Posterior drift** — KL divergence between the deployed posterior
  N(mean, cov) and a freshly fit posterior on recent data. A large KL
  means the model's beliefs have moved.
- **Predictive drift** — PSI (and a two-sample KS statistic) on the
  distribution of predictive probabilities, which needs no labels and
  fires before outcomes arrive.
"""
from __future__ import annotations

import math
from dataclasses import dataclass

from .linalg import kl_gaussian


def population_stability_index(
    expected: list[float], actual: list[float], n_bins: int = 10
) -> float:
    """PSI over fixed [0, 1] bins; <0.1 stable, 0.1–0.25 watch, >0.25 shift."""
    psi = 0.0
    eps = 1e-6
    for b in range(n_bins):
        lo, hi = b / n_bins, (b + 1) / n_bins
        e = sum(1 for p in expected if lo <= p < hi or (b == n_bins - 1 and p == 1.0))
        a = sum(1 for p in actual if lo <= p < hi or (b == n_bins - 1 and p == 1.0))
        e_share = max(e / len(expected), eps)
        a_share = max(a / len(actual), eps)
        psi += (a_share - e_share) * math.log(a_share / e_share)
    return psi


def ks_statistic(a: list[float], b: list[float]) -> float:
    """Two-sample Kolmogorov–Smirnov statistic."""
    xs = sorted(set(a) | set(b))
    best = 0.0
    for x in xs:
        ca = sum(1 for v in a if v <= x) / len(a)
        cb = sum(1 for v in b if v <= x) / len(b)
        best = max(best, abs(ca - cb))
    return best


@dataclass
class DriftReport:
    kl_divergence: float
    psi: float
    ks: float
    kl_level: str
    psi_level: str
    action: str

    def to_dict(self) -> dict:
        return {
            "kl_divergence": round(self.kl_divergence, 4),
            "psi": round(self.psi, 4),
            "ks": round(self.ks, 4),
            "kl_level": self.kl_level,
            "psi_level": self.psi_level,
            "action": self.action,
        }


def _level(value: float, warn: float, critical: float) -> str:
    if value >= critical:
        return "critical"
    if value >= warn:
        return "watch"
    return "ok"


def drift_report(
    old_mean: list[float],
    old_cov: list[list[float]],
    new_mean: list[float],
    new_cov: list[list[float]],
    old_probs: list[float],
    new_probs: list[float],
    kl_warn: float = 0.5,
    kl_critical: float = 2.0,
    psi_warn: float = 0.10,
    psi_critical: float = 0.25,
) -> DriftReport:
    kl = kl_gaussian(old_mean, old_cov, new_mean, new_cov)
    psi = population_stability_index(old_probs, new_probs)
    ks = ks_statistic(old_probs, new_probs)
    kl_level = _level(kl, kl_warn, kl_critical)
    psi_level = _level(psi, psi_warn, psi_critical)
    if kl_level == "critical" or psi_level == "critical":
        action = "recalibrate_and_refit"
    elif kl_level == "watch" or psi_level == "watch":
        action = "monitor_closely"
    else:
        action = "no_action"
    return DriftReport(
        kl_divergence=kl,
        psi=psi,
        ks=ks,
        kl_level=kl_level,
        psi_level=psi_level,
        action=action,
    )
