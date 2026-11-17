"""Seeded synthetic data for the demo and tests.

The task is binary classification on a small tabular problem (a generic
"flag this case for review" decision, deliberately domain-free). Features
are drawn from a seeded generator with a planted linear signal plus one
interaction term, so a linear model is good but not perfect — exactly the
regime where uncertainty matters.
"""
from __future__ import annotations

import math
import random
from dataclasses import dataclass


@dataclass
class Dataset:
    X: list[list[float]]
    y: list[int]
    feature_names: list[str]
    seed: int
    n_features: int

    def __len__(self) -> int:
        return len(self.y)


@dataclass
class Split:
    X_train: list[list[float]]
    y_train: list[int]
    X_test: list[list[float]]
    y_test: list[int]


# Planted ground-truth weights (intercept first). Kept fixed so every run
# of the demo is comparable; the seed only changes the sampled rows.
TRUE_WEIGHTS = [0.4, 1.1, -0.9, 0.7, 0.5, -0.6, 0.3]
# Strength of the one planted interaction term. The linear model is good
# but not perfect; 0.15 keeps the misspecification small enough that the
# Laplace posterior's 90% intervals stay honest (coverage ≈ 0.9).
INTERACTION = 0.15


def _sigmoid(z: float) -> float:
    if z >= 0:
        return 1.0 / (1.0 + math.exp(-z))
    e = math.exp(z)
    return e / (1.0 + e)


def generate_dataset(n: int = 1200, n_features: int = 6, seed: int = 42) -> Dataset:
    if n_features != 6:
        raise ValueError("this synthetic generator is defined for n_features=6")
    rng = random.Random(seed)
    weights = TRUE_WEIGHTS
    X: list[list[float]] = []
    y: list[int] = []
    for _ in range(n):
        row = [rng.gauss(0.0, 1.0) for _ in range(n_features)]
        # Planted interaction: the signal is mostly linear with one
        # interaction term, so the linear model is strong but imperfect.
        z = weights[0] + sum(w * v for w, v in zip(weights[1:], row))
        z += INTERACTION * row[0] * row[1]
        p = _sigmoid(z)
        X.append(row)
        y.append(1 if rng.random() < p else 0)
    return Dataset(
        X=X,
        y=y,
        feature_names=[f"feature_{i}" for i in range(n_features)],
        seed=seed,
        n_features=n_features,
    )


def generate_shifted_dataset(
    n: int = 600, n_features: int = 6, seed: int = 123, shift: float = 1.0
) -> Dataset:
    """A drifted copy: the feature means move, simulating a distribution
    shift the drift monitors should catch."""
    rng = random.Random(seed)
    weights = TRUE_WEIGHTS
    X: list[list[float]] = []
    y: list[int] = []
    for _ in range(n):
        row = [rng.gauss(shift, 1.0) for _ in range(n_features)]
        z = weights[0] + sum(w * v for w, v in zip(weights[1:], row))
        z += INTERACTION * row[0] * row[1]
        p = _sigmoid(z)
        X.append(row)
        y.append(1 if rng.random() < p else 0)
    return Dataset(
        X=X,
        y=y,
        feature_names=[f"feature_{i}" for i in range(n_features)],
        seed=seed,
        n_features=n_features,
    )


def true_probability(x: list[float]) -> float:
    """Ground-truth P(y=1 | x) under the planted generator — used to
    measure credible-interval coverage (does the 90% interval contain the
    true probability ~90% of the time?)."""
    z = TRUE_WEIGHTS[0] + sum(w * v for w, v in zip(TRUE_WEIGHTS[1:], x))
    z += INTERACTION * x[0] * x[1]
    return _sigmoid(z)


def true_probabilities(X: list[list[float]]) -> list[float]:
    return [true_probability(x) for x in X]


def train_test_split(data: Dataset, test_fraction: float = 0.25, seed: int = 42) -> Split:
    rng = random.Random(seed + 1)
    idx = list(range(len(data)))
    rng.shuffle(idx)
    n_test = max(1, int(round(len(data) * test_fraction)))
    test_idx = set(idx[:n_test])
    X_train = [x for i, x in enumerate(data.X) if i not in test_idx]
    y_train = [v for i, v in enumerate(data.y) if i not in test_idx]
    X_test = [x for i, x in enumerate(data.X) if i in test_idx]
    y_test = [v for i, v in enumerate(data.y) if i in test_idx]
    return Split(X_train=X_train, y_train=y_train, X_test=X_test, y_test=y_test)
