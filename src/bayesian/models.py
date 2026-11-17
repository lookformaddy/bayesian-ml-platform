"""Bayesian models, all from scratch in pure stdlib.

- :class:`BayesianLogisticRegression` — Laplace approximation: Newton
  MAP estimate under a Gaussian prior, posterior covariance = inverse
  Hessian at the MAP. Predictive probabilities use the probit
  approximation and (for variance / credible intervals) seeded Monte
  Carlo draws from the posterior.
- :class:`BayesianLinearRegression` — closed-form Gaussian posterior.
- :class:`PointLogisticRegression` — the deterministic point model used
  as the latency-budget fallback in serving.
"""
from __future__ import annotations

import math
import random
from dataclasses import dataclass, field

from .linalg import add_diag, inverse, matvec, multivariate_normal_sample, solve


def _sigmoid(z: float) -> float:
    if z >= 0:
        return 1.0 / (1.0 + math.exp(-z))
    e = math.exp(z)
    return e / (1.0 + e)


def _design_row(x: list[float]) -> list[float]:
    return [1.0] + list(x)


def _log_posterior_logreg(w: list[float], X: list[list[float]], y: list[int], prior_var: float) -> float:
    ll = 0.0
    for xi, yi in zip(X, y):
        z = sum(a * b for a, b in zip(_design_row(xi), w))
        p = _sigmoid(z)
        ll += yi * math.log(max(p, 1e-12)) + (1 - yi) * math.log(max(1.0 - p, 1e-12))
    prior = -0.5 * sum(v * v for v in w) / prior_var
    return ll + prior


@dataclass
class BayesianLogisticRegression:
    prior_var: float = 4.0
    max_iters: int = 50
    tol: float = 1e-6
    n_posterior_samples: int = 400
    seed: int = 42

    mean_: list[float] = field(default_factory=list, init=False)
    cov_: list[list[float]] = field(default_factory=list, init=False)
    n_features_: int = field(default=0, init=False)
    fitted: bool = field(default=False, init=False)

    def fit(self, X: list[list[float]], y: list[int]) -> "BayesianLogisticRegression":
        d = len(X[0]) + 1  # + intercept
        w = [0.0] * d
        prior_prec = 1.0 / self.prior_var
        for _ in range(self.max_iters):
            grad = [0.0] * d
            hess = [[0.0] * d for _ in range(d)]
            for xi, yi in zip(X, y):
                row = _design_row(xi)
                p = _sigmoid(sum(a * b for a, b in zip(row, w)))
                for i in range(d):
                    grad[i] += (yi - p) * row[i]
                s = p * (1.0 - p)
                for i in range(d):
                    for j in range(d):
                        hess[i][j] += s * row[i] * row[j]
            for i in range(d):
                grad[i] -= prior_prec * w[i]
            hess = add_diag(hess, prior_prec)
            try:
                step = solve(hess, grad)
            except ValueError:
                break
            w = [a + b for a, b in zip(w, step)]
            if max(abs(v) for v in step) < self.tol:
                break
        # Posterior covariance = inverse Hessian at the MAP.
        hess = [[0.0] * d for _ in range(d)]
        for xi, yi in zip(X, y):
            row = _design_row(xi)
            p = _sigmoid(sum(a * b for a, b in zip(row, w)))
            s = p * (1.0 - p)
            for i in range(d):
                for j in range(d):
                    hess[i][j] += s * row[i] * row[j]
        hess = add_diag(hess, prior_prec)
        self.mean_ = w
        self.cov_ = inverse(hess)
        self.n_features_ = len(X[0])
        self.fitted = True
        return self

    def _require_fitted(self) -> None:
        if not self.fitted:
            raise ValueError("model is not fitted")

    def logit_mean_var(self, x: list[float]) -> tuple[float, float]:
        """Posterior mean and variance of the logit w^T x."""
        self._require_fitted()
        row = _design_row(x)
        mean = sum(a * b for a, b in zip(row, self.mean_))
        var = sum(
            row[i] * sum(self.cov_[i][j] * row[j] for j in range(len(row)))
            for i in range(len(row))
        )
        return mean, max(var, 0.0)

    def predict_proba(self, X: list[list[float]]) -> list[float]:
        """Predictive probability via the probit approximation.

        p ≈ sigmoid(kappa * m) with kappa = (1 + pi/8 * v)^(-1/2).
        """
        out = []
        for x in X:
            m, v = self.logit_mean_var(x)
            kappa = (1.0 + math.pi / 8.0 * v) ** (-0.5)
            out.append(_sigmoid(kappa * m))
        return out

    def predict_proba_with_uncertainty(
        self, X: list[list[float]], seed: int | None = None
    ) -> list[dict]:
        """Monte-Carlo predictive mean / std / 90% credible interval."""
        self._require_fitted()
        rng = random.Random(self.seed if seed is None else seed)
        samples = multivariate_normal_sample(
            self.mean_, self.cov_, rng, self.n_posterior_samples
        )
        out = []
        for x in X:
            row = _design_row(x)
            ps = [
                _sigmoid(sum(a * b for a, b in zip(row, w))) for w in samples
            ]
            ps_sorted = sorted(ps)
            n = len(ps_sorted)
            lo = ps_sorted[int(0.05 * n)]
            hi = ps_sorted[min(n - 1, int(0.95 * n))]
            mean = sum(ps) / n
            var = sum((p - mean) ** 2 for p in ps) / n
            out.append(
                {
                    "mean": mean,
                    "std": math.sqrt(var),
                    "lo_90": lo,
                    "hi_90": hi,
                    "width_90": hi - lo,
                }
            )
        return out

    def predict(self, X: list[list[float]]) -> list[int]:
        return [1 if p >= 0.5 else 0 for p in self.predict_proba(X)]

    def log_posterior(self, X: list[list[float]], y: list[int]) -> float:
        self._require_fitted()
        return _log_posterior_logreg(self.mean_, X, y, self.prior_var)


@dataclass
class BayesianLinearRegression:
    """Closed-form Gaussian posterior for y = w^T x + eps.

    Prior w ~ N(0, alpha^-1 I), noise eps ~ N(0, beta^-1).
    Posterior: precision = alpha I + beta X^T X,
    mean = beta * Sigma * X^T y.
    """

    alpha: float = 1.0
    beta: float = 1.0

    mean_: list[float] = field(default_factory=list, init=False)
    cov_: list[list[float]] = field(default_factory=list, init=False)
    fitted: bool = field(default=False, init=False)

    def fit(self, X: list[list[float]], y: list[float]) -> "BayesianLinearRegression":
        d = len(X[0]) + 1
        xtx = [[0.0] * d for _ in range(d)]
        xty = [0.0] * d
        for xi, yi in zip(X, y):
            row = _design_row(xi)
            for i in range(d):
                xty[i] += row[i] * yi
                for j in range(d):
                    xtx[i][j] += row[i] * row[j]
        precision = [[self.beta * xtx[i][j] for j in range(d)] for i in range(d)]
        precision = add_diag(precision, self.alpha)
        self.cov_ = inverse(precision)
        self.mean_ = [self.beta * v for v in matvec(self.cov_, xty)]
        self.fitted = True
        return self

    def predict_with_uncertainty(self, X: list[list[float]]) -> list[dict]:
        if not self.fitted:
            raise ValueError("model is not fitted")
        out = []
        for x in X:
            row = _design_row(x)
            mean = sum(a * b for a, b in zip(row, self.mean_))
            var = sum(
                row[i] * sum(self.cov_[i][j] * row[j] for j in range(len(row)))
                for i in range(len(row))
            )
            var = max(var, 0.0) + 1.0 / self.beta  # + observation noise
            out.append({"mean": mean, "std": math.sqrt(var)})
        return out


@dataclass
class PointLogisticRegression:
    """Deterministic point-estimate logistic regression (MAP with the same
    prior, covariance discarded). Used as the latency-budget fallback:
    fast, no posterior sampling, always available."""

    prior_var: float = 4.0
    max_iters: int = 50
    tol: float = 1e-6

    weights_: list[float] = field(default_factory=list, init=False)
    fitted: bool = field(default=False, init=False)

    def fit(self, X: list[list[float]], y: list[int]) -> "PointLogisticRegression":
        base = BayesianLogisticRegression(
            prior_var=self.prior_var, max_iters=self.max_iters, tol=self.tol
        ).fit(X, y)
        self.weights_ = base.mean_
        self.fitted = True
        return self

    def predict_proba(self, X: list[list[float]]) -> list[float]:
        if not self.fitted:
            raise ValueError("model is not fitted")
        return [
            _sigmoid(sum(a * b for a, b in zip(_design_row(x), self.weights_)))
            for x in X
        ]

    def predict(self, X: list[list[float]]) -> list[int]:
        return [1 if p >= 0.5 else 0 for p in self.predict_proba(X)]
