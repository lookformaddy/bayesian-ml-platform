"""Uncertainty-aware serving with a latency budget and point-model fallback.

Every prediction returns the predictive mean, its standard deviation, the
90% credible interval, and a decision: ``act`` / ``no_act`` / ``abstain``
(refer to a human). If the Bayesian path exceeds the latency budget, the
request falls back to the deterministic point model instead of failing —
the response is flagged so downstream systems know the uncertainty fields
are unavailable.
"""
from __future__ import annotations

import time
from dataclasses import dataclass, field

from .config import Settings
from .decision import decide
from .models import BayesianLogisticRegression, PointLogisticRegression


@dataclass
class ServeResult:
    probability: float
    std: float | None
    lo_90: float | None
    hi_90: float | None
    action: str
    reason: str
    latency_ms: float
    fallback: bool
    model: str

    def to_dict(self) -> dict:
        return {
            "probability": round(self.probability, 4),
            "std": None if self.std is None else round(self.std, 4),
            "lo_90": self.lo_90,
            "hi_90": self.hi_90,
            "action": self.action,
            "reason": self.reason,
            "latency_ms": round(self.latency_ms, 2),
            "fallback": self.fallback,
            "model": self.model,
        }


@dataclass
class BayesianServer:
    model: BayesianLogisticRegression
    point_model: PointLogisticRegression
    settings: Settings
    n_served: int = field(default=0, init=False)
    n_abstained: int = field(default=0, init=False)
    n_fallback: int = field(default=0, init=False)

    def predict_one(self, x: list[float], simulate_slow: bool = False) -> ServeResult:
        start = time.perf_counter()
        if simulate_slow:
            # Test hook: pretend the Bayesian path blew the budget.
            elapsed_ms = self.settings.serve_budget_ms + 1.0
        else:
            unc = self.model.predict_proba_with_uncertainty([x])[0]
            elapsed_ms = (time.perf_counter() - start) * 1000.0
        if elapsed_ms > self.settings.serve_budget_ms:
            p = self.point_model.predict_proba([x])[0]
            self.n_fallback += 1
            self.n_served += 1
            return ServeResult(
                probability=p,
                std=None,
                lo_90=None,
                hi_90=None,
                action="act" if p >= 0.5 else "no_act",
                reason="latency budget exceeded; served by deterministic point model",
                latency_ms=elapsed_ms,
                fallback=True,
                model="point",
            )
        decision = decide(
            probability=unc["mean"],
            std=unc["std"],
            cost_fp=self.settings.cost_false_positive,
            cost_fn=self.settings.cost_false_negative,
            abstain_std_threshold=self.settings.abstain_std_threshold,
        )
        if decision.action == "abstain":
            self.n_abstained += 1
        self.n_served += 1
        return ServeResult(
            probability=unc["mean"],
            std=unc["std"],
            lo_90=unc["lo_90"],
            hi_90=unc["hi_90"],
            action=decision.action,
            reason=decision.reason,
            latency_ms=elapsed_ms,
            fallback=False,
            model="bayesian",
        )

    def stats(self) -> dict:
        return {
            "n_served": self.n_served,
            "n_abstained": self.n_abstained,
            "n_fallback": self.n_fallback,
            "abstain_rate": self.n_abstained / self.n_served if self.n_served else 0.0,
            "fallback_rate": self.n_fallback / self.n_served if self.n_served else 0.0,
        }
