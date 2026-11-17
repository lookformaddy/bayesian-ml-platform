"""FastAPI service for the Bayesian ML Platform.

Endpoints:
- GET  /health            — liveness + active model version
- POST /predict           — uncertainty-aware prediction with decision
- POST /predict/batch     — batch predictions
- GET  /model             — active model metadata + evaluation
- GET  /drift             — latest drift report (recomputed on shifted data)
- GET  /registry          — all registered versions and stages
- GET  /audit             — hash-chained audit log (verified)
"""
from __future__ import annotations

from fastapi import FastAPI, HTTPException
from pydantic import BaseModel, Field

from .config import Settings
from .platform import BayesianPlatform

_platform: BayesianPlatform | None = None
_result = None


def _get_platform() -> BayesianPlatform:
    global _platform, _result
    if _platform is None:
        _platform = BayesianPlatform(Settings())
        _result = _platform.run_pipeline(version="v1")
    return _platform


class PredictRequest(BaseModel):
    features: list[float] = Field(..., min_length=1)
    simulate_slow: bool = False


class BatchPredictRequest(BaseModel):
    rows: list[list[float]] = Field(..., min_length=1)


def create_app() -> FastAPI:
    app = FastAPI(title="Bayesian ML Platform", version="1.0.0")

    @app.get("/health")
    def health() -> dict:
        platform = _get_platform()
        active = platform.registry.get_active()
        return {
            "status": "ok",
            "active_model": active.name if active else None,
            "active_version": active.version if active else None,
            "stage": active.stage if active else None,
        }

    @app.post("/predict")
    def predict(req: PredictRequest) -> dict:
        platform = _get_platform()
        if len(req.features) != platform.settings.n_features:
            raise HTTPException(
                status_code=422,
                detail=f"expected {platform.settings.n_features} features",
            )
        return _result.server.predict_one(
            req.features, simulate_slow=req.simulate_slow
        ).to_dict()

    @app.post("/predict/batch")
    def predict_batch(req: BatchPredictRequest) -> dict:
        platform = _get_platform()
        results = []
        for row in req.rows:
            if len(row) != platform.settings.n_features:
                raise HTTPException(
                    status_code=422,
                    detail=f"expected {platform.settings.n_features} features",
                )
            results.append(_result.server.predict_one(row).to_dict())
        return {"predictions": results, "serving_stats": _result.server.stats()}

    @app.get("/model")
    def model() -> dict:
        platform = _get_platform()
        active = platform.registry.get_active()
        return {
            "model": "bayesian-logistic",
            "version": active.version if active else None,
            "stage": active.stage if active else None,
            "evaluation": _result.evaluation,
            "calibration": _result.calibration,
            "mcmc": _result.mcmc,
            "manifest": _result.manifest,
        }

    @app.get("/drift")
    def drift() -> dict:
        platform = _get_platform()
        return platform.run_drift_scenario(_result)["drift"]

    @app.get("/registry")
    def registry() -> dict:
        platform = _get_platform()
        return {
            "active_version": platform.registry.active_version,
            "versions": [v.to_dict() for v in platform.registry.versions.values()],
        }

    @app.get("/audit")
    def audit() -> dict:
        platform = _get_platform()
        return {
            "verified": platform.registry.audit.verify(),
            "entries": platform.registry.audit.to_dict(),
        }

    return app


app = create_app()
