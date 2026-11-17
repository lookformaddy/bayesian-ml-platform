"""Cost-aware decisioning on top of predictive probabilities.

A prediction is only useful once it becomes a decision. With asymmetric
costs — a missed positive costs ``cost_fn`` and a false alarm costs
``cost_fp`` — the expected-utility-optimal threshold is
``cost_fp / (cost_fp + cost_fn)``, not 0.5. This module also owns the
abstain / refer-to-human rule: when predictive uncertainty is high, the
cheapest action is to hand the case to a human.
"""
from __future__ import annotations

import math
from dataclasses import dataclass


def optimal_threshold(cost_fp: float, cost_fn: float) -> float:
    """Expected-utility threshold for P(y=1) under asymmetric costs."""
    if cost_fp <= 0 or cost_fn <= 0:
        raise ValueError("costs must be positive")
    return cost_fp / (cost_fp + cost_fn)


@dataclass
class Decision:
    action: str  # "act" | "no_act" | "abstain"
    probability: float
    std: float
    threshold: float
    expected_cost: float
    reason: str


def decide(
    probability: float,
    std: float,
    cost_fp: float,
    cost_fn: float,
    abstain_std_threshold: float,
) -> Decision:
    """One uncertainty-aware, cost-aware decision."""
    threshold = optimal_threshold(cost_fp, cost_fn)
    if std > abstain_std_threshold:
        return Decision(
            action="abstain",
            probability=probability,
            std=std,
            threshold=threshold,
            expected_cost=float("nan"),
            reason=f"predictive std {std:.3f} exceeds abstain threshold {abstain_std_threshold:.3f}",
        )
    if probability >= threshold:
        # Act: expected cost is the chance we are wrong (false positive).
        cost = (1.0 - probability) * cost_fp
        return Decision(
            action="act",
            probability=probability,
            std=std,
            threshold=threshold,
            expected_cost=cost,
            reason=f"p={probability:.3f} >= threshold {threshold:.3f}",
        )
    cost = probability * cost_fn
    return Decision(
        action="no_act",
        probability=probability,
        std=std,
        threshold=threshold,
        expected_cost=cost,
        reason=f"p={probability:.3f} < threshold {threshold:.3f}",
    )


def expected_cost_of_policy(
    probs: list[float],
    y: list[int],
    cost_fp: float,
    cost_fn: float,
    threshold: float | None = None,
) -> dict:
    """Realised cost of acting at a threshold, plus the 0.5 baseline."""
    threshold = threshold if threshold is not None else optimal_threshold(cost_fp, cost_fn)
    cost = 0.0
    cost_baseline = 0.0
    for p, yi in zip(probs, y):
        pred = 1 if p >= threshold else 0
        pred_base = 1 if p >= 0.5 else 0
        if pred == 1 and yi == 0:
            cost += cost_fp
        elif pred == 0 and yi == 1:
            cost += cost_fn
        if pred_base == 1 and yi == 0:
            cost_baseline += cost_fp
        elif pred_base == 0 and yi == 1:
            cost_baseline += cost_fn
    return {
        "threshold": round(threshold, 4),
        "total_cost": round(cost, 2),
        "baseline_0.5_cost": round(cost_baseline, 2),
        "cost_saved_vs_baseline": round(cost_baseline - cost, 2),
    }
