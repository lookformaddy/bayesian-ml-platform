"""Calibration of predictive probabilities.

Reliability metrics (ECE, Brier score, log-loss) plus two recalibration
methods — Platt scaling (a logistic map on the logit, fit by gradient
descent) and binned/isotonic-style recalibration — all from scratch.
"""
from __future__ import annotations

import math
from dataclasses import dataclass, field


def _sigmoid(z: float) -> float:
    if z >= 0:
        return 1.0 / (1.0 + math.exp(-z))
    e = math.exp(z)
    return e / (1.0 + e)


def _logit(p: float) -> float:
    p = min(max(p, 1e-6), 1.0 - 1e-6)
    return math.log(p / (1.0 - p))


def brier_score(probs: list[float], y: list[int]) -> float:
    return sum((p - yi) ** 2 for p, yi in zip(probs, y)) / len(y)


def log_loss(probs: list[float], y: list[int]) -> float:
    total = 0.0
    for p, yi in zip(probs, y):
        p = min(max(p, 1e-12), 1.0 - 1e-12)
        total += yi * math.log(p) + (1 - yi) * math.log(1.0 - p)
    return -total / len(y)


def reliability_bins(
    probs: list[float], y: list[int], n_bins: int = 10
) -> list[dict]:
    bins: list[dict] = []
    for b in range(n_bins):
        lo, hi = b / n_bins, (b + 1) / n_bins
        idx = [i for i, p in enumerate(probs) if lo <= p < hi or (b == n_bins - 1 and p == 1.0)]
        if not idx:
            bins.append({"bin": (lo, hi), "count": 0, "mean_pred": 0.0, "mean_obs": 0.0, "gap": 0.0})
            continue
        mean_pred = sum(probs[i] for i in idx) / len(idx)
        mean_obs = sum(y[i] for i in idx) / len(idx)
        bins.append(
            {
                "bin": (lo, hi),
                "count": len(idx),
                "mean_pred": mean_pred,
                "mean_obs": mean_obs,
                "gap": abs(mean_pred - mean_obs),
            }
        )
    return bins


def expected_calibration_error(probs: list[float], y: list[int], n_bins: int = 10) -> float:
    bins = reliability_bins(probs, y, n_bins)
    n = len(y)
    return sum(b["gap"] * b["count"] for b in bins) / n


@dataclass
class PlattCalibrator:
    """Platt scaling: p_cal = sigmoid(a * logit(p) + b), fit by gradient
    descent on log-loss with a small L2 penalty on ``a`` drifting from 1."""

    a: float = 1.0
    b: float = 0.0
    fitted: bool = field(default=False, init=False)

    def fit(self, probs: list[float], y: list[int], iters: int = 500, lr: float = 0.05) -> "PlattCalibrator":
        logits = [_logit(p) for p in probs]
        a, b = 1.0, 0.0
        n = len(y)
        for _ in range(iters):
            ga = gb = 0.0
            for z, yi in zip(logits, y):
                p = _sigmoid(a * z + b)
                err = p - yi
                ga += err * z
                gb += err
            ga = ga / n + 0.01 * (a - 1.0)
            gb = gb / n
            a -= lr * ga
            b -= lr * gb
        self.a, self.b = a, b
        self.fitted = True
        return self

    def transform(self, probs: list[float]) -> list[float]:
        if not self.fitted:
            raise ValueError("calibrator is not fitted")
        return [_sigmoid(self.a * _logit(p) + self.b) for p in probs]


@dataclass
class BinnedCalibrator:
    """Isotonic-style binned recalibration: each probability bin maps to
    its observed frequency on the calibration set."""

    n_bins: int = 10
    mapping: list[float] = field(default_factory=list, init=False)
    fitted: bool = field(default=False, init=False)

    def fit(self, probs: list[float], y: list[int]) -> "BinnedCalibrator":
        bins = reliability_bins(probs, y, self.n_bins)
        overall = sum(y) / len(y)
        self.mapping = [
            (b["mean_obs"] if b["count"] else overall) for b in bins
        ]
        # Enforce monotonicity (isotonic-style) with a running max.
        for i in range(1, len(self.mapping)):
            self.mapping[i] = max(self.mapping[i], self.mapping[i - 1])
        self.fitted = True
        return self

    def transform(self, probs: list[float]) -> list[float]:
        if not self.fitted:
            raise ValueError("calibrator is not fitted")
        out = []
        for p in probs:
            idx = min(int(p * self.n_bins), self.n_bins - 1)
            out.append(self.mapping[idx])
        return out


@dataclass
class CalibrationReport:
    ece_before: float
    ece_after: float
    brier_before: float
    brier_after: float
    logloss_before: float
    logloss_after: float
    method: str

    def to_dict(self) -> dict:
        return {
            "method": self.method,
            "ece_before": round(self.ece_before, 4),
            "ece_after": round(self.ece_after, 4),
            "brier_before": round(self.brier_before, 4),
            "brier_after": round(self.brier_after, 4),
            "logloss_before": round(self.logloss_before, 4),
            "logloss_after": round(self.logloss_after, 4),
        }


def calibrate_with_reference(
    ref_probs: list[float],
    ref_y: list[int],
    probs: list[float],
    y: list[int],
    method: str = "platt",
) -> tuple[list[float], CalibrationReport]:
    """Fit the calibrator on a reference set (e.g. training predictions)
    and report before/after metrics on a separate evaluation set. The
    shipped model serves the calibrated probabilities, so the evaluation
    report is built from them."""
    cal = PlattCalibrator() if method == "platt" else BinnedCalibrator()
    cal.fit(ref_probs, ref_y)
    fixed = cal.transform(probs)
    report = CalibrationReport(
        ece_before=expected_calibration_error(probs, y),
        ece_after=expected_calibration_error(fixed, y),
        brier_before=brier_score(probs, y),
        brier_after=brier_score(fixed, y),
        logloss_before=log_loss(probs, y),
        logloss_after=log_loss(fixed, y),
        method=method,
    )
    return fixed, report


def calibrate_and_report(
    probs: list[float],
    y: list[int],
    method: str = "platt",
) -> tuple[list[float], CalibrationReport]:
    """Fit a calibrator on half the data, evaluate on the other half, and
    report before/after metrics. (Split keeps the report honest.)"""
    n = len(probs)
    half = n // 2
    cal = PlattCalibrator() if method == "platt" else BinnedCalibrator()
    cal.fit(probs[:half], y[:half])
    raw = probs[half:]
    fixed = cal.transform(raw)
    y_hold = y[half:]
    report = CalibrationReport(
        ece_before=expected_calibration_error(raw, y_hold),
        ece_after=expected_calibration_error(fixed, y_hold),
        brier_before=brier_score(raw, y_hold),
        brier_after=brier_score(fixed, y_hold),
        logloss_before=log_loss(raw, y_hold),
        logloss_after=log_loss(fixed, y_hold),
        method=method,
    )
    return fixed, report
