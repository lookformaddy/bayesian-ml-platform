"""Platform orchestration: train → calibrate → evaluate → gate → serve.

:func:`run_pipeline` is the single entry point the demo, CLI, and service
share. It fits the Laplace logistic model, cross-checks the posterior with
MCMC, calibrates, evaluates (including credible-interval coverage),
registers the version with a reproducibility manifest, attempts promotion,
and returns everything the caller needs to report or serve.
"""
from __future__ import annotations

from dataclasses import dataclass, field

from .bandit import run_epsilon_greedy, run_thompson_sampling
from .calibration import calibrate_and_report, calibrate_with_reference
from .config import Settings
from .data import (
    generate_dataset,
    generate_shifted_dataset,
    train_test_split,
    true_probabilities,
)
from .decision import expected_cost_of_policy
from .drift import drift_report
from .evaluation import evaluate
from .manifest import build_manifest
from .mcmc import MetropolisHastings
from .models import BayesianLogisticRegression, PointLogisticRegression
from .registry import ModelRegistry
from .serving import BayesianServer


@dataclass
class PipelineResult:
    model: BayesianLogisticRegression
    point_model: PointLogisticRegression
    mcmc: dict
    calibration: dict
    evaluation: dict
    cost_policy: dict
    registry: ModelRegistry
    promoted: bool
    promotion_message: str
    server: BayesianServer
    manifest: dict
    bandit: dict = field(default_factory=dict)
    drift: dict = field(default_factory=dict)

    def summary(self) -> dict:
        return {
            "evaluation": self.evaluation,
            "calibration": self.calibration,
            "mcmc": self.mcmc,
            "cost_policy": self.cost_policy,
            "promoted": self.promoted,
            "promotion_message": self.promotion_message,
            "bandit": self.bandit,
            "drift": self.drift,
        }


class BayesianPlatform:
    def __init__(self, settings: Settings | None = None):
        self.settings = settings or Settings()
        self.registry = ModelRegistry(self.settings)

    def run_pipeline(self, version: str = "v1") -> PipelineResult:
        s = self.settings
        data = generate_dataset(n=s.n_samples, n_features=s.n_features, seed=s.seed)
        split = train_test_split(data, test_fraction=s.test_fraction, seed=s.seed)

        model = BayesianLogisticRegression(
            prior_var=s.prior_var,
            max_iters=s.max_newton_iters,
            tol=s.newton_tol,
            n_posterior_samples=s.n_posterior_samples,
            seed=s.seed,
        ).fit(split.X_train, split.y_train)
        point_model = PointLogisticRegression(
            prior_var=s.prior_var, max_iters=s.max_newton_iters, tol=s.newton_tol
        ).fit(split.X_train, split.y_train)

        # MCMC cross-check of the Laplace posterior.
        mcmc = MetropolisHastings(
            n_samples=s.mcmc_samples,
            burn_in=s.mcmc_burn_in,
            proposal_std=s.mcmc_proposal_std,
            prior_var=s.prior_var,
            seed=s.seed,
        ).sample(split.X_train, split.y_train, init=model.mean_)

        probs = model.predict_proba(split.X_test)
        intervals = model.predict_proba_with_uncertainty(split.X_test)
        true_probs = true_probabilities(split.X_test)

        # Calibration: fit Platt scaling on the training predictions and
        # serve calibrated probabilities, so the evaluation report — and
        # the promotion gates — are measured on what actually ships.
        train_probs = model.predict_proba(split.X_train)
        cal_probs, cal_report = calibrate_with_reference(
            train_probs, split.y_train, probs, split.y_test, method="platt"
        )
        report = evaluate(cal_probs, intervals, split.y_test, true_probs=true_probs)

        cost_policy = expected_cost_of_policy(
            cal_probs, split.y_test, s.cost_false_positive, s.cost_false_negative
        )

        manifest = build_manifest(
            "bayesian-logistic", version, s, split.X_train, split.y_train
        )
        entry = self.registry.register(
            "bayesian-logistic", version, report, manifest
        )
        promoted, message = self.registry.promote("bayesian-logistic", version)

        server = BayesianServer(model, point_model, s)
        return PipelineResult(
            model=model,
            point_model=point_model,
            mcmc=mcmc.to_dict(),
            calibration=cal_report.to_dict(),
            evaluation=report.to_dict(),
            cost_policy=cost_policy,
            registry=self.registry,
            promoted=promoted,
            promotion_message=message,
            server=server,
            manifest=manifest.to_dict(),
        )

    def run_bandit_comparison(self, n_rounds: int = 500) -> dict:
        # Three generic arms with planted true rates.
        rates = [0.25, 0.60, 0.40]
        thompson = run_thompson_sampling(rates, n_rounds=n_rounds, seed=self.settings.seed)
        greedy = run_epsilon_greedy(rates, n_rounds=n_rounds, seed=self.settings.seed)
        return {
            "thompson_sampling": thompson.to_dict(),
            "epsilon_greedy": greedy.to_dict(),
            "regret_saved": round(greedy.cumulative_regret - thompson.cumulative_regret, 2),
        }

    def run_drift_scenario(self, result: PipelineResult) -> dict:
        """Fit a fresh posterior on shifted data and report drift vs the
        deployed posterior; a critical signal triggers recalibration."""
        s = self.settings
        shifted = generate_shifted_dataset(
            n=600, n_features=s.n_features, seed=s.seed + 99, shift=1.0
        )
        new_model = BayesianLogisticRegression(
            prior_var=s.prior_var,
            max_iters=s.max_newton_iters,
            tol=s.newton_tol,
            seed=s.seed,
        ).fit(shifted.X, shifted.y)
        old_probs = result.model.predict_proba(shifted.X)
        new_probs = new_model.predict_proba(shifted.X)
        report = drift_report(
            result.model.mean_,
            result.model.cov_,
            new_model.mean_,
            new_model.cov_,
            old_probs,
            new_probs,
            kl_warn=s.kl_warn,
            kl_critical=s.kl_critical,
            psi_warn=s.psi_warn,
            psi_critical=s.psi_critical,
        )
        recalibrated = None
        if report.action == "recalibrate_and_refit":
            # The deployed (old) model is miscalibrated under drift:
            # recalibrate it on the shifted data and report the gain.
            _, cal = calibrate_and_report(old_probs, shifted.y, method="platt")
            recalibrated = cal.to_dict()
        return {
            "drift": report.to_dict(),
            "recalibration_after_drift": recalibrated,
        }
