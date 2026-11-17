# Bayesian ML Platform

**A point prediction is a bet. A Bayesian prediction is a bet with the odds
attached.** An uncertainty-aware modeling and serving platform built from
scratch (no sklearn, no numpy, no torch): Bayesian logistic regression via
the Laplace approximation, a deterministic MCMC cross-check, Bayesian
linear regression in closed form, Thompson-sampling bandits, calibration,
cost-aware decisioning, drift monitoring, and a gated model registry — all
in pure stdlib Python so every number in the system is auditable in source.

## The enterprise problem

Most production models ship a single number and hide how sure they are:

- **Silent overconfidence.** A model that is 97% sure and wrong costs the
  same as one that is 51% sure and wrong — until the wrong one is a
  high-stakes decision. Without uncertainty, there is no way to know which
  predictions to trust.
- **Thresholds picked by habit.** Teams default to 0.5 even when a missed
  positive costs 5× a false alarm. The threshold is a business decision,
  not a modeling default.
- **Calibration is assumed, never measured.** Accuracy looks fine while the
  probabilities are systematically overconfident, so every downstream
  expected-value calculation is wrong.
- **Drift is discovered by damage.** By the time labelled outcomes show a
  model has gone stale, the bad decisions have already shipped.
- **Promotion is a vibe.** A model goes live because it looked good on one
  split, with no check that its uncertainty is honest or its calibration
  survives.

This platform makes each failure mode explicit and gated: a model is
promoted only if it is accurate, calibrated, *and* its credible intervals
actually cover the truth; serving abstains to a human when uncertainty is
high; drift fires on unlabeled data; and every decision is hash-chained
into an audit log.

## What it demonstrates

- **Bayesian logistic regression (Laplace)** — Newton MAP estimate under a
  Gaussian prior, posterior covariance = inverse Hessian at the MAP;
  predictive probabilities via the probit approximation and seeded Monte
  Carlo draws for 90% credible intervals.
- **Deterministic MCMC cross-check** — Metropolis–Hastings on a seeded RNG
  verifies the Laplace posterior mean (acceptance 0.27, min ESS 700 in the
  demo); a bad approximation cannot ship silently.
- **Bayesian linear regression** — closed-form Gaussian posterior
  (precision = αI + βXᵀX).
- **Thompson-sampling bandit** — Beta-posterior decisioning with cumulative
  regret tracking, benchmarked against epsilon-greedy.
- **Calibration** — reliability bins, ECE, Brier, log-loss; Platt scaling
  and binned recalibration, fit on training predictions and reported on a
  held-out set so the numbers stay honest.
- **Cost-aware decisioning** — expected-utility threshold from asymmetric
  costs (FN = 5× FP → threshold 0.167, not 0.5), with realised-cost
  reporting against the naive 0.5 baseline.
- **Uncertainty-aware serving** — every prediction returns mean, std, and
  a 90% credible interval; the server abstains (refer to human) when
  predictive std exceeds the threshold, and falls back to a deterministic
  point model when the Bayesian path blows the latency budget.
- **Posterior drift monitoring** — KL divergence between deployed and
  refit posteriors plus PSI / KS on predictive distributions; a critical
  signal triggers recalibration.
- **Gated registry** — versioned models (DRAFT → ACTIVE) with five
  promotion gates (accuracy, AUC, ECE ceiling, uncertainty-coverage
  floor, minimum eval cases) and rollback to the previous ACTIVE version.
- **Reproducibility manifests** — seed + config hash + data hash per
  version, so two runs with the same manifest hash are the same run.
- **Audit** — SHA-256 hash-chained log of every registry and drift
  decision, verifiable.

## Quickstart

```bash
make install   # create .venv and install dependencies
make test      # run the test suite
make demo      # full end-to-end demo (see below)
make serve     # FastAPI service on :8000
```

Or without make:

```bash
python -m venv .venv && .venv/bin/pip install -r requirements.txt
PYTHONPATH=src python -m pytest -q
PYTHONPATH=src python scripts/demo.py
PYTHONPATH=src python -m uvicorn bayesian.service:app --port 8000
```

## The demo

`scripts/demo.py` runs the full pipeline on seeded synthetic data (1,200
rows, 6 features, planted linear signal + one weak interaction term):

```
== Bayesian ML Platform — demo ==
accuracy 0.7633 | AUC 0.8556 | log-loss 0.4602
ECE before calibration 0.0705 -> after 0.0730
Brier before 0.1534 -> after 0.1534
90% credible-interval coverage 0.9133 (mean width 0.114)
MCMC cross-check: acceptance 0.271, min ESS 700
Cost-aware threshold 0.167: total cost 97.0 vs 0.5-baseline 235.0 (saved 138.0)
Bandit regret (500 rounds): Thompson 6.7 vs epsilon-greedy 10.4 (saved 3.8)
Drift event: KL 26.302 (critical), PSI 0.132 (watch) -> action recalibrate_and_refit
  recalibration after drift: ECE 0.1078 -> 0.0676
Promotion: bayesian-logistic@v1 promoted to ACTIVE
```

The demo also serves five test rows through the uncertainty-aware server
(mean, std, 90% credible interval, action, fallback flag).

## CLI

```bash
PYTHONPATH=src python -m bayesian.cli demo
PYTHONPATH=src python -m bayesian.cli train --version v1
PYTHONPATH=src python -m bayesian.cli predict 0.1 -0.2 0.3 0.0 0.5 -0.1
PYTHONPATH=src python -m bayesian.cli bandit
PYTHONPATH=src python -m bayesian.cli drift
```

## API

| Endpoint | Description |
| --- | --- |
| `GET /health` | Liveness + active model version and stage |
| `POST /predict` | One uncertainty-aware prediction + decision (`simulate_slow` forces the point-model fallback) |
| `POST /predict/batch` | Batch predictions + serving stats |
| `GET /model` | Active model metadata, evaluation, calibration, MCMC, manifest |
| `GET /drift` | Latest drift report (KL / PSI / KS + recommended action) |
| `GET /registry` | All versions, stages, and gate results |
| `GET /audit` | Hash-chained audit log with verification result |

## Key design decisions and trade-offs

- **Laplace first, MCMC as the check.** Laplace is fast and closed-form;
  the deterministic MCMC chain verifies the posterior mean agrees, so a
  bad approximation cannot ship silently.
- **Probit approximation for the predictive mean, Monte Carlo for the
  interval.** The probit map is cheap and accurate for the mean; the 90%
  credible interval needs the full posterior, so it is sampled (seeded).
- **Calibration is measured on data the calibrator never saw.** Platt
  scaling is fit on training predictions and evaluated on the test set;
  fitting and evaluating on the same rows would flatter the ECE.
- **Coverage is measured against the true probability.** For a classifier
  the honest coverage question is "does the 90% interval contain the true
  P(y=1)?" — the synthetic generator knows it, so the demo reports real
  coverage (0.9133), not a proxy.
- **Cost-aware thresholds, not 0.5.** With a missed positive costing 5× a
  false alarm, the optimal threshold is 1/6; the demo saves 138 cost units
  vs the naive 0.5 threshold.
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

Trade-offs: Laplace assumes a unimodal, roughly Gaussian posterior;
genuinely multimodal posteriors need the MCMC path (here a cross-check).
MCMC is pure Python and therefore slow on large data; the default chain
(2,000 samples) is sized for the demo's 1,200 rows. The point-model
fallback is a deliberate capability downgrade under latency pressure —
availability over uncertainty, explicitly flagged.

## Project layout

```
src/bayesian/
  config.py       settings (BMP_* env vars)
  linalg.py       pure-stdlib linear algebra + Gaussian KL + sampling
  data.py         seeded synthetic data (+ shifted copy) and splits
  models.py       Laplace logistic, closed-form Bayesian linear, point model
  mcmc.py         Metropolis-Hastings sampler + ESS diagnostics
  bandit.py       Thompson sampling vs epsilon-greedy
  calibration.py  ECE / Brier / log-loss, Platt + binned recalibration
  decision.py     expected-utility thresholds, abstain rule
  drift.py        posterior KL, PSI, KS
  evaluation.py   accuracy / AUC / confusion / interval coverage
  registry.py     versioned registry, promotion gates, rollback
  audit.py        SHA-256 hash-chained audit log
  manifest.py     reproducibility manifests (seed + config + data hash)
  serving.py      uncertainty-aware serving, abstention, fallback
  platform.py     pipeline orchestration + bandit / drift scenarios
  service.py      FastAPI service
  cli.py          demo | train | predict | bandit | drift
  demo.py         end-to-end demo
tests/            pytest suite (FastAPI tests auto-skip without FastAPI)
scripts/demo.py   demo entry point
docs/architecture.md
docs/github-actions-ci.yml   (move to .github/workflows/ci.yml to activate)
```

## Configuration

See `.env.example` for every `BMP_*` override: gates
(`BMP_ACCURACY_FLOOR`, `BMP_AUC_FLOOR`, `BMP_ECE_CEILING`,
`BMP_COVERAGE_FLOOR`), serving (`BMP_ABSTAIN_STD`, `BMP_COST_FP`,
`BMP_COST_FN`, `BMP_SERVE_BUDGET_MS`), drift (`BMP_KL_WARN`,
`BMP_KL_CRITICAL`, `BMP_PSI_WARN`, `BMP_PSI_CRITICAL`), and MCMC /
reproducibility settings.
