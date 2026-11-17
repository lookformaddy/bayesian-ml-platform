"""Evaluation: point metrics, ranking metrics, and uncertainty quality.

Uncertainty quality is the part most platforms skip: a Bayesian model
must not only be accurate, it must know what it doesn't know. We measure
that with the empirical coverage of the 90% credible intervals (should be
near 0.90) and the mean interval width (narrower is better at equal
coverage).
"""
from __future__ import annotations

import math
from dataclasses import dataclass


def accuracy(preds: list[int], y: list[int]) -> float:
    return sum(p == t for p, t in zip(preds, y)) / len(y)


def roc_auc(probs: list[float], y: list[int]) -> float:
    """Rank-based AUC (Mann–Whitney U), ties counted as 0.5."""
    pos = [p for p, t in zip(probs, y) if t == 1]
    neg = [p for p, t in zip(probs, y) if t == 0]
    if not pos or not neg:
        return 0.5
    wins = 0.0
    for p in pos:
        for q in neg:
            if p > q:
                wins += 1.0
            elif p == q:
                wins += 0.5
    return wins / (len(pos) * len(neg))


def confusion(preds: list[int], y: list[int]) -> dict:
    tp = sum(1 for p, t in zip(preds, y) if p == 1 and t == 1)
    fp = sum(1 for p, t in zip(preds, y) if p == 1 and t == 0)
    tn = sum(1 for p, t in zip(preds, y) if p == 0 and t == 0)
    fn = sum(1 for p, t in zip(preds, y) if p == 0 and t == 1)
    return {"tp": tp, "fp": fp, "tn": tn, "fn": fn}


def coverage_of_intervals(
    intervals: list[dict],
    true_probs: list[float],
    key_lo: str = "lo_90",
    key_hi: str = "hi_90",
) -> dict:
    """Empirical coverage of credible intervals: fraction of rows whose
    *true* probability falls inside the 90% interval, plus mean width.
    A well-calibrated posterior covers ~90% at a narrow width."""
    covered = 0
    widths = []
    for interval, tp in zip(intervals, true_probs):
        widths.append(interval["width_90"])
        if interval[key_lo] <= tp <= interval[key_hi]:
            covered += 1
    return {
        "coverage_90": covered / len(true_probs) if true_probs else 0.0,
        "mean_width_90": sum(widths) / len(widths) if widths else 0.0,
        "n": len(true_probs),
    }


@dataclass
class EvaluationReport:
    accuracy: float
    auc: float
    brier: float
    logloss: float
    ece: float
    coverage_90: float
    mean_width_90: float
    n: int
    confusion: dict

    def to_dict(self) -> dict:
        return {
            "accuracy": round(self.accuracy, 4),
            "auc": round(self.auc, 4),
            "brier": round(self.brier, 4),
            "logloss": round(self.logloss, 4),
            "ece": round(self.ece, 4),
            "coverage_90": round(self.coverage_90, 4),
            "mean_width_90": round(self.mean_width_90, 4),
            "n": self.n,
            "confusion": self.confusion,
        }


def evaluate(
    probs: list[float],
    intervals: list[dict],
    y: list[int],
    true_probs: list[float] | None = None,
) -> EvaluationReport:
    from .calibration import brier_score, expected_calibration_error, log_loss

    preds = [1 if p >= 0.5 else 0 for p in probs]
    cov = (
        coverage_of_intervals(intervals, true_probs)
        if true_probs is not None
        else {"coverage_90": 0.0, "mean_width_90": 0.0, "n": len(y)}
    )
    return EvaluationReport(
        accuracy=accuracy(preds, y),
        auc=roc_auc(probs, y),
        brier=brier_score(probs, y),
        logloss=log_loss(probs, y),
        ece=expected_calibration_error(probs, y),
        coverage_90=cov["coverage_90"],
        mean_width_90=cov["mean_width_90"],
        n=len(y),
        confusion=confusion(preds, y),
    )
