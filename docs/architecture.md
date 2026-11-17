# Architecture

## Overview

The Bayesian ML Platform turns a point prediction into a *decision with a
stated uncertainty*. Every prediction carries a predictive mean, a standard
deviation, and a 90% credible interval; serving abstains (refers to a human)
when uncertainty exceeds a threshold, and promotion requires the model to
prove both accuracy *and* calibration *and* honest uncertainty.

```
synthetic data (seeded)
      │
      ▼
BayesianLogisticRegression  ── Laplace approx ──► posterior N(mean, cov)
      │   (Newton MAP + inverse Hessian)
      ▼
MetropolisHastings (MCMC) ── cross-checks the Laplace posterior
      │
      ▼
Platt calibration ──► calibrated probabilities
      │
      ▼
Evaluation ── accuracy / AUC / Brier / log-loss / ECE / 90% CI coverage
      │
      ▼
ModelRegistry ── gates (accuracy, AUC, ECE, coverage, n) ──► ACTIVE
      │
      ▼
BayesianServer ── uncertainty-aware serving, abstention,
                  latency budget + point-model fallback
      │
      ├── ThompsonSampling bandit (decisioning over actions)
      └── DriftMonitor ── KL(posterior) + PSI(predictive) ──► recalibrate
```

All state changes (register / promote / reject / rollback / drift action)
are appended to a SHA-256 hash-chained audit log.

## Modules

| Module | Responsibility |
| --- | --- |
| `linalg.py` | Pure-stdlib dense linear algebra: solve, inverse, Cholesky, log-det, Gaussian KL, posterior sampling |
| `data.py` | Seeded synthetic binary-classification data (+ a shifted copy for drift) and train/test split |
| `models.py` | Bayesian logistic regression (Laplace), Bayesian linear regression (closed form), deterministic point logistic regression (fallback) |
| `mcmc.py` | Metropolis–Hastings sampler with acceptance rate and ESS diagnostics |
| `bandit.py` | Thompson sampling and epsilon-greedy Bernoulli bandits with regret tracking |
| `calibration.py` | ECE / Brier / log-loss, reliability bins, Platt scaling, binned recalibration |
| `decision.py` | Expected-utility thresholds under asymmetric costs; abstain rule |
| `drift.py` | KL divergence between posteriors, PSI and KS on predictive distributions |
| `evaluation.py` | Accuracy, ROC-AUC, confusion, 90% credible-interval coverage |
| `registry.py` | Versioned registry, promotion gates, rollback |
| `audit.py` | SHA-256 hash-chained audit log with verification |
| `manifest.py` | Reproducibility manifest: seed + config hash + data hash |
| `serving.py` | Uncertainty-aware serving, abstention, latency budget, point-model fallback |
| `platform.py` | Pipeline orchestration: train → MCMC → calibrate → evaluate → gate → serve; bandit + drift scenarios |
| `service.py` | FastAPI service (`/health`, `/predict`, `/predict/batch`, `/model`, `/drift`, `/registry`, `/audit`) |
| `cli.py` | `demo | train | predict | bandit | drift` |
| `demo.py` | End-to-end demo printing headline numbers |

## Key design decisions

- **Laplace approximation first, MCMC as the check.** Laplace is fast and
  closed-form (Newton MAP + inverse Hessian); the deterministic MCMC chain
  verifies the posterior mean agrees, so a bad approximation cannot ship
  silently.
- **Probit approximation for the predictive mean, Monte Carlo for the
  interval.** The probit map is cheap and accurate for the mean; the 90%
  credible interval needs the full posterior, so it is sampled (seeded).
- **Calibration is measured on a hold-out half.** Fitting Platt scaling on
  the same rows it is evaluated on would flatter the ECE; the split keeps
  the report honest.
- **Cost-aware thresholds, not 0.5.** With a missed positive costing 5× a
  false alarm, the optimal threshold is 1/6. The demo reports realised
  cost vs the naive 0.5 baseline.
- **Abstention is a first-class action.** When predictive std exceeds the
  threshold the cheapest action is a human review, so the server returns
  `abstain` instead of a low-confidence guess.
- **Latency budget with a point-model fallback.** If the Bayesian path
  exceeds the budget, the deterministic point model answers instead of
  failing the request — flagged `fallback: true` so callers know the
  uncertainty fields are unavailable.
- **Drift needs no labels to fire.** PSI on predictive probabilities and KL
  between posteriors detect a shift before outcomes arrive; a critical
  signal triggers recalibration (and refit in production).
- **Promotion is fail-closed and audited.** Accuracy, AUC, ECE, coverage,
  and minimum-eval-cases gates must all pass; every decision is hash-chained.
- **Pure stdlib core.** No numpy / sklearn / torch anywhere in the modeling
  code — every number is auditable in source. FastAPI is only the serving
  surface and is optional for the core library and its tests.

## Trade-offs

- Laplace assumes a unimodal, roughly Gaussian posterior; genuinely
  multimodal posteriors need the MCMC path (here only a cross-check).
- The probit approximation slightly shrinks extreme probabilities; the
  Monte-Carlo interval is the authoritative uncertainty.
- MCMC is pure Python and therefore slow on large data; the default chain
  (2,000 samples) is sized for the demo's 1,200 rows.
- The point-model fallback is a deliberate capability downgrade under
  latency pressure — availability over uncertainty, explicitly flagged.
