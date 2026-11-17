import pytest

from bayesian.evaluation import EvaluationReport
from bayesian.manifest import Manifest


def _report(**over):
    base = dict(
        accuracy=0.85,
        auc=0.90,
        brier=0.10,
        logloss=0.35,
        ece=0.03,
        coverage_90=0.88,
        mean_width_90=0.40,
        n=300,
        confusion={"tp": 100, "fp": 20, "tn": 150, "fn": 30},
    )
    base.update(over)
    return EvaluationReport(**base)


def _manifest():
    return Manifest(
        model_name="bayesian-logistic",
        version="v1",
        seed=42,
        config_hash="a" * 64,
        data_version="synthetic-v1",
        data_hash="b" * 64,
    )


def test_promote_passes_all_gates(registry):
    entry = registry.register("bayesian-logistic", "v1", _report(), _manifest())
    ok, msg = registry.promote("bayesian-logistic", "v1")
    assert ok is True
    assert entry.stage == "ACTIVE"
    assert registry.active_version == "bayesian-logistic@v1"
    assert all(g.passed for g in entry.gates)


def test_promote_rejected_when_ece_too_high(registry):
    registry.register("bayesian-logistic", "v1", _report(ece=0.20), _manifest())
    ok, msg = registry.promote("bayesian-logistic", "v1")
    assert ok is False
    assert "ece_ceiling" in msg
    assert registry.active_version is None


def test_promote_rejected_when_coverage_too_low(registry):
    registry.register("bayesian-logistic", "v1", _report(coverage_90=0.50), _manifest())
    ok, msg = registry.promote("bayesian-logistic", "v1")
    assert ok is False
    assert "coverage_floor" in msg


def test_promote_unknown_version(registry):
    ok, msg = registry.promote("bayesian-logistic", "v99")
    assert ok is False
    assert "unknown" in msg


def test_rollback_restores_previous(registry):
    registry.register("bayesian-logistic", "v1", _report(), _manifest())
    registry.promote("bayesian-logistic", "v1")
    registry.register("bayesian-logistic", "v2", _report(), _manifest())
    registry.promote("bayesian-logistic", "v2")
    assert registry.active_version == "bayesian-logistic@v2"
    ok, msg = registry.rollback("bayesian-logistic")
    assert ok is True
    assert registry.active_version == "bayesian-logistic@v1"
    assert registry.get_active().stage == "ACTIVE"


def test_rollback_with_no_previous_fails(registry):
    ok, _ = registry.rollback("bayesian-logistic")
    assert ok is False


def test_registry_audit_is_verifiable(registry):
    registry.register("bayesian-logistic", "v1", _report(), _manifest())
    registry.promote("bayesian-logistic", "v1")
    assert registry.audit.verify() is True
    events = [e.event for e in registry.audit.entries]
    assert events == ["model_registered", "model_promoted"]
