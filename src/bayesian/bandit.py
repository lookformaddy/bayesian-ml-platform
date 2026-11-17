"""Bandit decisioning: Thompson sampling vs epsilon-greedy.

Bernoulli arms with Beta(1, 1) priors. Thompson sampling draws a
probability from each arm's posterior and pulls the argmax — the
principled Bayesian way to balance exploration and exploitation.
Cumulative regret is measured against the oracle arm.
"""
from __future__ import annotations

import random
from dataclasses import dataclass, field


@dataclass
class BanditResult:
    policy: str
    total_reward: float
    cumulative_regret: float
    pulls_per_arm: list[int]
    estimated_rates: list[float]

    def to_dict(self) -> dict:
        return {
            "policy": self.policy,
            "total_reward": round(self.total_reward, 2),
            "cumulative_regret": round(self.cumulative_regret, 2),
            "pulls_per_arm": self.pulls_per_arm,
            "estimated_rates": [round(r, 4) for r in self.estimated_rates],
        }


@dataclass
class BernoulliBandit:
    true_rates: list[float]
    seed: int = 42
    rng: random.Random = field(init=False)

    def __post_init__(self) -> None:
        self.rng = random.Random(self.seed)

    def pull(self, arm: int) -> int:
        return 1 if self.rng.random() < self.true_rates[arm] else 0


def run_thompson_sampling(
    true_rates: list[float], n_rounds: int = 500, seed: int = 42
) -> BanditResult:
    rng = random.Random(seed)
    bandit = BernoulliBandit(true_rates, seed=seed + 7)
    k = len(true_rates)
    alpha = [1.0] * k
    beta = [1.0] * k
    pulls = [0] * k
    rewards = [0.0] * k
    best = max(true_rates)
    regret = 0.0
    for _ in range(n_rounds):
        draws = [rng.betavariate(alpha[i], beta[i]) for i in range(k)]
        arm = max(range(k), key=lambda i: draws[i])
        r = bandit.pull(arm)
        pulls[arm] += 1
        rewards[arm] += r
        alpha[arm] += r
        beta[arm] += 1 - r
        regret += best - true_rates[arm]
    return BanditResult(
        policy="thompson_sampling",
        total_reward=sum(rewards),
        cumulative_regret=regret,
        pulls_per_arm=pulls,
        estimated_rates=[rewards[i] / pulls[i] if pulls[i] else 0.0 for i in range(k)],
    )


def run_epsilon_greedy(
    true_rates: list[float],
    n_rounds: int = 500,
    epsilon: float = 0.1,
    seed: int = 42,
) -> BanditResult:
    rng = random.Random(seed)
    bandit = BernoulliBandit(true_rates, seed=seed + 7)
    k = len(true_rates)
    pulls = [0] * k
    rewards = [0.0] * k
    best = max(true_rates)
    regret = 0.0
    for _ in range(n_rounds):
        if rng.random() < epsilon:
            arm = rng.randrange(k)
        else:
            means = [
                (rewards[i] / pulls[i] if pulls[i] else 0.0) for i in range(k)
            ]
            arm = max(range(k), key=lambda i: means[i])
        r = bandit.pull(arm)
        pulls[arm] += 1
        rewards[arm] += r
        regret += best - true_rates[arm]
    return BanditResult(
        policy="epsilon_greedy",
        total_reward=sum(rewards),
        cumulative_regret=regret,
        pulls_per_arm=pulls,
        estimated_rates=[rewards[i] / pulls[i] if pulls[i] else 0.0 for i in range(k)],
    )
